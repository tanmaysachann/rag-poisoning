"""Local v5 research gate combining the frozen detector and trained MiniLM.

Both components must be available. No silent fallback is allowed: a missing or
changed model artifact raises an error so a run cannot claim the v5 policy while
executing v3 alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from detect.research_detector import ResearchDetector


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CombinedGate:
    def __init__(self, old: ResearchDetector, model_path: Path, *, new_threshold: float,
                 max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.old = old
        self.new_threshold = new_threshold
        self.threshold = 0.5  # threshold-normalized gate margin; not a probability
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device).eval()
        self.max_length = max_length

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "CombinedGate":
        root = manifest_path.resolve().parents[1]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, item in manifest["inputs"].items():
            path = root / item["path"]
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise ValueError(f"Combined gate input missing or changed: {name}")
        confirmation = json.loads((root / manifest["inputs"]["confirmation_report"]["path"]).read_text(encoding="utf-8"))
        if not confirmation["combined_adoption_gate"]["pass"]:
            raise ValueError("Combined gate did not pass the frozen confirmation rule")
        training = json.loads((root / manifest["inputs"]["training_report"]["path"]).read_text(encoding="utf-8"))
        old = joblib.load(root / manifest["inputs"]["old_detector"]["path"])
        if not isinstance(old, ResearchDetector):
            raise ValueError("Wrong frozen detector type")
        if old.threshold != manifest["old_threshold"] or training["threshold"] != manifest["new_threshold"]:
            raise ValueError("Detector threshold mismatch")
        return cls(old, root / manifest["model_directory"],
                   new_threshold=manifest["new_threshold"],
                   max_length=training["max_length"])

    def _new_scores(self, query: str, texts: list[str]) -> list[float]:
        import torch

        if not texts:
            return []
        tokens = self.tokenizer(
            [query] * len(texts), texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**tokens).logits.float(), dim=1)[:, 1]
        return probabilities.cpu().tolist()

    def score_batch(self, query: str, texts: list[str]) -> list[dict]:
        old_scores = [self.old.score(query, text) for text in texts]
        new_scores = self._new_scores(query, texts)
        outputs = []
        for old, new in zip(old_scores, new_scores):
            old_hit = old["risk_score"] >= self.old.threshold
            new_hit = new >= self.new_threshold
            # A score of 0.5 is exactly at either component's threshold.
            normalized = 0.5 * max(old["risk_score"] / self.old.threshold,
                                   new / self.new_threshold)
            outputs.append({
                "decision": "quarantine" if old_hit or new_hit else "accept",
                "risk_score": min(1.0, normalized), "threshold": self.threshold,
                "risk_score_kind": "threshold_normalized_margin_not_probability",
                "triggered_by": [name for name, hit in (("v3", old_hit), ("trained", new_hit)) if hit],
                "features": {**old["features"],
                             "v3_risk": old["risk_score"],
                             "v3_threshold": self.old.threshold,
                             "trained_risk": new,
                             "trained_threshold": self.new_threshold},
            })
        return outputs

    def score(self, query: str, doc_text: str) -> dict:
        return self.score_batch(query, [doc_text])[0]
