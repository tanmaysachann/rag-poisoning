"""Compare retrieved documents with an independent, pinned evidence corpus.

The source decision is an authorization decision. A changed or newly inserted
document is unreviewed; its text is not thereby proven factually false. The
optional factual model scores a changed claim for research, never overriding
the source decision or treating a hash as a truth label.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from security.integrity import IntegrityManifest


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _containing_sentence(text: str, offset: int) -> str:
    """Return the sentence containing an edit without using an answer key."""
    start = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        if offset < boundary.start():
            return text[start:boundary.start()].strip()
        start = boundary.end()
    return text[start:].strip()


def changed_claim_pair(original: str, candidate: str) -> dict[str, str] | None:
    """Expose the first changed source and candidate sentences for review."""
    if original == candidate:
        return None
    edits = [part for part in difflib.SequenceMatcher(
        None, original, candidate, autojunk=False,
    ).get_opcodes() if part[0] != "equal"]
    if not edits:
        return None
    _, old_start, old_end, new_start, new_end = edits[0]
    old_sentence = _containing_sentence(original, old_start)
    new_sentence = _containing_sentence(candidate, new_start)
    return {
        "source_sentence": old_sentence,
        "candidate_sentence": new_sentence,
        "source_edit": original[old_start:old_end],
        "candidate_edit": candidate[new_start:new_end],
    }


@dataclass(frozen=True)
class EvidenceScreen:
    """Screen a retrieved batch against a separately loaded clean snapshot."""

    snapshot: IntegrityManifest
    source_documents: dict[int, str]

    @classmethod
    def from_retriever(cls, snapshot: IntegrityManifest, retriever: Any) -> "EvidenceScreen":
        if snapshot.legacy or not snapshot.corpus_sha256:
            raise ValueError("A versioned pinned evidence manifest is required")
        if getattr(retriever, "corpus_digest", None) != snapshot.corpus_sha256:
            raise ValueError("The evidence retriever is not bound to the pinned manifest")
        source_documents = {int(doc["doc_id"]): str(doc["text"])
                            for doc in retriever.documents}
        if len(source_documents) != len(retriever.documents):
            raise ValueError("Duplicate document IDs in the evidence corpus")
        if set(map(str, source_documents)) != set(snapshot.hashes):
            raise ValueError("Evidence corpus and manifest document IDs differ")
        for doc_id, text in source_documents.items():
            if snapshot.check(doc_id, text)["status"] != "verified":
                raise ValueError("Evidence corpus contains an unverified document")
        return cls(snapshot, source_documents)

    def assess(self, document: dict) -> dict:
        doc_id = int(document["doc_id"])
        candidate = str(document["text"])
        check = self.snapshot.check(doc_id, candidate)
        status = check["status"]
        source_text = self.source_documents.get(doc_id)
        if status == "verified" and source_text is None:
            raise ValueError("Verified document is missing from the evidence corpus")
        comparison = (changed_claim_pair(source_text, candidate)
                      if source_text is not None and status != "verified" else None)
        return {
            "decision": "accept" if status == "verified" else "quarantine",
            "reason": ("pinned_source_verified" if status == "verified"
                       else "source_changed_since_review" if source_text is not None
                       else "source_not_reviewed"),
            "snapshot_status": status,
            "source_sha256": check["expected_hash"],
            "candidate_sha256": check["actual_hash"],
            "changed_claim_pair": comparison,
            "factual_verdict": "unassessed",
        }


class ChangedClaimModel:
    """Research scorer for a source sentence and its changed counterpart.

    Uses the VitaminC authors' pretrained model in its documented
    evidence-then-claim order. Its probability is diagnostic until calibrated
    on legitimate revisions and an unseen poisoning collection.
    """

    def __init__(self, model_path: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True,
        ).to(self.device).eval()
        label_id = self.model.config.label2id.get("REFUTES")
        if label_id is None:
            raise ValueError("Factual model has no REFUTES label")
        self.refutes_id = int(label_id)

    def score(self, pairs: list[dict[str, str]], *, batch_size: int = 16) -> list[float]:
        if not pairs:
            return []
        probabilities: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                tokens = self.tokenizer(
                    [row["source_sentence"] for row in batch],
                    [row["candidate_sentence"] for row in batch],
                    padding=True, truncation=True, max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                scores = self.torch.softmax(
                    self.model(**tokens).logits.float(), dim=1,
                )[:, self.refutes_id]
                probabilities.extend(float(value) for value in scores.cpu().tolist())
        return probabilities


class QuestionClaimModel:
    """Extract the answer claim from a candidate and independent source.

    The source document must be selected and verified outside this class. A
    different extracted span is a review signal, not a proof of poisoning.
    """

    def __init__(self, model_path: Path):
        from generation.trusted_reference import ReferenceReader

        self.reader = ReferenceReader(model_path)

    def compare(self, rows: list[dict[str, str]], *, batch_size: int = 16) -> list[dict]:
        from generation.trusted_reference import answer_overlap

        if not rows:
            return []
        questions = [row["question"] for row in rows]
        source_predictions = self.reader.extract_batch(
            questions, [row["source_text"] for row in rows], batch_size=batch_size,
        )
        candidate_predictions = self.reader.extract_batch(
            questions, [row["candidate_text"] for row in rows], batch_size=batch_size,
        )
        result = []
        for row, source, candidate in zip(rows, source_predictions, candidate_predictions):
            source_span = row["source_text"][source.start:source.end]
            candidate_span = row["candidate_text"][candidate.start:candidate.end]
            valid = (bool(source_span and candidate_span)
                     and source_span == source.answer
                     and candidate_span == candidate.answer)
            result.append({
                "source_claim": source_span if valid else None,
                "candidate_claim": candidate_span if valid else None,
                "source_confidence": source.score,
                "candidate_confidence": candidate.score,
                "claim_disagrees": bool(valid and not answer_overlap(source_span, candidate_span)),
                "status": "compared" if valid else "unassessable_span",
            })
        return result
