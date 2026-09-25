"""Gymnasium-style local document-edit MDP with a real retrieval terminal check."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np

from attack.harness import stage_document_attack
from pipeline.secure_rag import _select_answer
from poison.actions import EditAction, OPERATIONS, PAYLOADS, POSITIONS
from poison.edit_ops import apply_edit
from poison.reward import RewardWeights, step_reward, terminal_reward
from retrieval.hybrid_retriever import HybridRetriever, TextEmbedder

STATE_VERSION = "edit-mdp-v3-hashing-774"


class DocumentEditEnv:
    """One episode edits one seed passage against one frozen local corpus.

    ``reset`` and ``step`` follow Gymnasium's return convention. ``action_mask``
    is a three-axis boolean tensor for operation, position, and payload. Invalid
    actions consume a step and receive a penalty; they never mutate the text.
    """

    def __init__(
        self, corpus_path: Path, query: dict, wrong_answer: str, *,
        seed_doc_id: int | None = None, replace_existing: bool = False,
        max_steps: int = 4, max_words: int = 512,
        min_seed_similarity: float = 0.35,
        detector=None, reward_weights: RewardWeights = RewardWeights(),
    ) -> None:
        if max_steps < 1 or max_words < 1:
            raise ValueError("max_steps and max_words must be positive")
        if not 0 <= min_seed_similarity <= 1:
            raise ValueError("min_seed_similarity must be in [0, 1]")
        self.corpus_path = Path(corpus_path)
        self.query = query
        self.wrong_answer = wrong_answer.strip()
        self.answer_aliases = list(query["answer_aliases"])
        self.seed_doc_id = seed_doc_id if seed_doc_id is not None else query["support_doc_ids"][0]
        self.replace_existing = replace_existing
        self.max_steps = max_steps
        self.max_words = max_words
        self.min_seed_similarity = min_seed_similarity
        self.detector = detector
        self.reward_weights = reward_weights
        self.docs = [json.loads(line) for line in self.corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.seed_text = next(doc["text"] for doc in self.docs if doc["doc_id"] == self.seed_doc_id)
        self.attack_doc_id = self.seed_doc_id if replace_existing else max(doc["doc_id"] for doc in self.docs) + 1
        self.embedder = TextEmbedder(preferred_backend="hashing")
        self.query_vector = self.embedder.encode(query["question"])[0]
        self.seed_vector = self.embedder.encode(self.seed_text)[0]
        self.observation_shape = (774,)
        self.text = self.seed_text
        self.steps = 0
        self.edits = 0
        self.done = False
        self.last_operation = -1

    def _observation(self) -> np.ndarray:
        vector = self.embedder.encode(self.text)[0]
        query_words = set(re.findall(r"\w+", self.query["question"].casefold()))
        doc_words = set(re.findall(r"\w+", self.text.casefold()))
        scalar = np.asarray([
            float(self.query_vector @ vector),
            len(query_words & doc_words) / max(len(query_words), 1),
            min(len(self.text.split()) / self.max_words, 1.0),
            self.edits / self.max_steps,
            (self.last_operation + 1) / len(OPERATIONS),
            self.steps / self.max_steps,
        ], dtype=np.float32)
        return np.concatenate([self.query_vector, vector, scalar]).astype(np.float32)

    def _candidate(self, action: EditAction) -> tuple[str | None, str | None]:
        if action.operation == "STOP":
            if action.position or action.payload:
                return None, "STOP uses position=0 and payload=0"
            if self.edits == 0 and self.steps < self.max_steps - 1:
                return None, "At least one edit is required before STOP"
            return self.text, None
        if action.operation != "INSERT" and action.payload:
            return None, "Payload is only used by INSERT"
        if action.operation == "PARAPHRASE" and action.position:
            return None, "PARAPHRASE uses position=0"
        try:
            candidate = apply_edit(
                self.text, action, question=self.query["question"],
                wrong_answer=self.wrong_answer, answer_aliases=self.answer_aliases,
            )
        except ValueError as error:
            return None, str(error)
        if candidate == self.text:
            return None, "Edit made no change"
        if len(candidate.split()) > self.max_words:
            return None, "Document exceeds max_words"
        similarity = float(self.seed_vector @ self.embedder.encode(candidate)[0])
        if similarity < self.min_seed_similarity:
            return None, "Document changed too far from seed"
        return candidate, None

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((len(OPERATIONS), len(POSITIONS), len(PAYLOADS)), dtype=np.bool_)
        if self.done:
            return mask
        for op_index, operation in enumerate(OPERATIONS):
            for position in range(len(POSITIONS)):
                for payload in range(len(PAYLOADS)):
                    action = EditAction(operation, position, payload)
                    if self.steps < self.max_steps - 1 or operation == "STOP":
                        _, reason = self._candidate(action)
                        mask[op_index, position, payload] = reason is None
        return mask

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("Episode options are fixed at construction")
        self.rng = np.random.default_rng(seed)
        self.text = self.seed_text
        self.steps = 0
        self.edits = 0
        self.done = False
        self.last_operation = -1
        return self._observation(), {"state_version": STATE_VERSION, "action_mask": self.action_mask()}

    def _terminal(self) -> dict:
        with tempfile.TemporaryDirectory(prefix="sentinel-mdp-") as temporary:
            base = Path(temporary)
            staged = base / "staged.jsonl"
            stage_document_attack(
                self.corpus_path, staged, doc_id=self.attack_doc_id,
                text=self.text, replace_existing=self.replace_existing,
            )
            retriever = HybridRetriever(staged, artifact_dir=base / "index", force_rebuild=True)
            candidates = retriever.retrieve(self.query["question"], top_k=5)
            rank = next((i for i, doc in enumerate(candidates, 1) if doc["doc_id"] == self.attack_doc_id), None)
            answer, source_doc_id, _ = _select_answer(self.query["question"], candidates, retriever)
            attack_risk = None
            detected = None
            defended_answer = None
            defended_source = None
            if self.detector is not None:
                kept = []
                for doc in candidates:
                    risk = self.detector.score(self.query["question"], doc["text"])
                    if doc["doc_id"] == self.attack_doc_id:
                        attack_risk = risk["risk_score"]
                        detected = risk["decision"] == "quarantine"
                    if risk["decision"] != "quarantine":
                        kept.append(doc)
                defended_answer, defended_source, _ = _select_answer(self.query["question"], kept, retriever)
        success = source_doc_id == self.attack_doc_id and self.wrong_answer.casefold() in answer.casefold()
        defended_success = (
            defended_source == self.attack_doc_id and self.wrong_answer.casefold() in defended_answer.casefold()
            if self.detector is not None else None
        )
        reward_success = defended_success if defended_success is not None else success
        return {
            "retrieved": rank is not None, "top5_rank": rank,
            "attack_success": success, "answer": answer, "source_doc_id": source_doc_id,
            "attack_doc_risk": attack_risk, "attack_doc_detected": detected,
            "defended_answer": defended_answer, "defended_source_doc_id": defended_source,
            "defended_attack_success": defended_success,
            "reward": terminal_reward(
                retrieved=rank is not None, attack_success=reward_success,
                valid_attack=self.edits > 0, weights=self.reward_weights,
            ),
        }

    def step(self, action: EditAction | dict):
        if self.done:
            raise RuntimeError("Episode is finished; call reset()")
        if isinstance(action, dict):
            action = EditAction(**action)
        if not isinstance(action, EditAction):
            raise TypeError("Action must be EditAction or a matching dict")
        old_relevance = float(self.query_vector @ self.embedder.encode(self.text)[0])
        candidate, reason = self._candidate(action)
        valid = reason is None
        if valid and action.operation != "STOP":
            self.text = candidate
            self.edits += 1
        self.steps += 1
        self.last_operation = OPERATIONS.index(action.operation)
        terminated = valid and action.operation == "STOP"
        truncated = self.steps >= self.max_steps and not terminated
        self.done = terminated or truncated
        relevance = float(self.query_vector @ self.embedder.encode(self.text)[0])
        risk = (
            self.detector.score(self.query["question"], self.text)["risk_score"]
            if self.detector and valid and action.operation != "STOP" else None
        )
        components = step_reward(
            relevance_delta=relevance - old_relevance, detector_risk=risk,
            valid=valid, weights=self.reward_weights,
        )
        info = {
            "valid": valid, "reason": reason, "reward_components": components,
            "steps": self.steps, "edits": self.edits, "document": self.text,
            "state_version": STATE_VERSION,
        }
        reward = components["total"]
        if self.done:
            info["terminal"] = self._terminal()
            reward += info["terminal"]["reward"]["total"]
        info["action_mask"] = self.action_mask()
        return self._observation(), reward, terminated, truncated, info
