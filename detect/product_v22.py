"""Local research scanner: learned source risk plus pinned-corpus integrity."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import torch

from detect.reference_conditioned_v9 import ReferenceConditionedScorer
from detect.research_detector import ResearchDetector
from detect.snapshot_near_duplicate import SnapshotNearDuplicate
from detect.snapshot_containment_v23 import SnapshotContainment


def clean_tail_probability(scores: np.ndarray, clean_reference: np.ndarray) -> np.ndarray:
    """The frozen fusion rule, kept in serving code without training imports."""
    sorted_clean = np.sort(np.asarray(clean_reference, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    rank_below = np.searchsorted(sorted_clean, scores, side="left")
    return (len(sorted_clean) - rank_below + 1) / (len(sorted_clean) + 1)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PoisonDocumentScanner:
    """Score a supplied question, reference, and candidate for local review.

    The caller must establish reference authenticity. Model probabilities are
    source-label scores, and the snapshot is an integrity baseline, not truth.
    """

    def __init__(self, *, root: Path = Path("."), use_snapshot: bool = True,
                 snapshot_directory: Path | None = None,
                 snapshot_manifest_sha256: str | None = None,
                 integrity_mode: str = "near_copy") -> None:
        self.root = Path(root)
        if integrity_mode not in {"near_copy", "containment"}:
            raise ValueError("Unknown integrity mode")
        self.integrity_mode = integrity_mode
        artifact_name = ("product_v24_calibration.joblib" if integrity_mode == "containment"
                         else "product_v22_calibration.joblib")
        artifact_path = self.root / "artifacts" / artifact_name
        self.calibration = joblib.load(artifact_path)
        expected_schema = ("product-v24-calibration" if integrity_mode == "containment"
                           else "product-v22-calibration")
        if self.calibration.get("schema") != expected_schema:
            raise ValueError("Unexpected calibration schema")
        if integrity_mode == "containment" and self.calibration[
            "containment_report_sha256"] != file_sha256(
                self.root / "results/snapshot_containment_v23.json"):
            raise ValueError("Containment calibration report changed")
        neural_path = self.root / "artifacts/multifamily_v20"
        signal_path = self.root / "artifacts/fluency_v21.joblib"
        v3_path = self.root / "artifacts/research_detector_v3_hashing.joblib"
        required = {
            "neural_model_sha256": neural_path / "model.safetensors",
            "signal_model_sha256": signal_path,
            "v3_model_sha256": v3_path,
            "fusion_report_sha256": self.root / "results/fusion_v21.json",
            "data_manifest_sha256": self.root / "data/multifamily_v20/manifest.json",
        }
        for field, path in required.items():
            if self.calibration[field] != file_sha256(path):
                raise ValueError(f"Pinned scanner input changed: {path}")
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.neural = ReferenceConditionedScorer(
            neural_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.signal = joblib.load(signal_path)
        portable = self.root / "artifacts/research_detector_v3_hashing_web.joblib"
        portable_manifest = self.root / "research_protocols/v3_web_detector.json"
        if os.name == "nt" and os.getenv("V24_PORTABLE_V3") != "1":
            self.v3 = joblib.load(v3_path)
        else:
            binding = json.loads(portable_manifest.read_text(encoding="utf-8"))
            if (binding["source_sha256"] != self.calibration["v3_model_sha256"] or
                    binding["serving_sha256"] != file_sha256(portable)):
                raise ValueError("Portable V3 detector does not match frozen source")
            self.v3 = joblib.load(portable)
        if (not isinstance(self.v3, ResearchDetector) or
                self.signal["v3_artifact_sha256"] != self.calibration["v3_model_sha256"]):
            raise ValueError("Signal extractor differs from trained artifact")
        self.snapshot = None
        if use_snapshot:
            snapshot_path = (self.root / "data/trusted_snapshot_v12" if
                             snapshot_directory is None else Path(snapshot_directory))
            expected_snapshot_sha = (self.calibration["snapshot_manifest_sha256"] if
                                     snapshot_directory is None else snapshot_manifest_sha256)
            if expected_snapshot_sha is None:
                raise ValueError("Custom snapshot requires its pinned manifest SHA-256")
            actual_snapshot_sha = file_sha256(snapshot_path / "manifest.json")
            if expected_snapshot_sha != actual_snapshot_sha:
                if os.getenv("V24_REBUILT_SNAPSHOT") != "1" or integrity_mode != "containment" or snapshot_directory is not None:
                    raise ValueError("Snapshot manifest changed")
                canonical_path = snapshot_path / "calibration_manifest.json"
                if file_sha256(canonical_path) != expected_snapshot_sha:
                    raise ValueError("Frozen source snapshot manifest changed")
                canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
                rebuilt = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
                source_fields = ("archive_md5", "archive_sha256", "member", "passages", "schema")
                if any(rebuilt.get(field) != canonical.get(field) for field in source_fields):
                    raise ValueError("Rebuilt snapshot does not match the frozen source corpus")
            scanner_class = (SnapshotContainment if integrity_mode == "containment"
                             else SnapshotNearDuplicate)
            self.snapshot = scanner_class(snapshot_path)

    def close(self) -> None:
        if self.snapshot is not None:
            self.snapshot.close()

    def assess(self, query: str, reference: str, candidate: str, *,
               expected_sha256: str | None = None) -> dict:
        if not all(part.strip() for part in (query, reference, candidate)):
            raise ValueError("Question, supplied reference and candidate are required")
        neural = float(self.neural.score([{
            "query": query, "reference": reference, "candidate": candidate,
        }], batch_size=24)[0])
        base_features = self.v3.features(query, candidate)
        v3_risk = float(self.v3.classifier.predict_proba(
            base_features.reshape(1, -1))[0, 1])
        features = np.concatenate([
            base_features, [v3_risk], self.signal["fluency"].features(candidate),
        ]).reshape(1, -1)
        signal = float(self.signal["model"].predict_proba(features)[0, 1])
        neural_tail = float(clean_tail_probability(
            np.asarray([neural]), self.calibration["neural_clean_scores"])[0])
        signal_tail = float(clean_tail_probability(
            np.asarray([signal]), self.calibration["signal_clean_scores"])[0])
        fused = float(-np.log(min(neural_tail, signal_tail)))
        learned_flag = fused >= self.calibration["fusion_threshold_1pct"]
        integrity = self.snapshot.assess(candidate) if self.snapshot is not None else None
        integrity_flag = (integrity is not None and integrity["near_copy_score"] >=
                          self.calibration["snapshot_threshold"])
        identity = {"status": "not_supplied", "flag": False}
        if expected_sha256 is not None:
            expected = expected_sha256.lower()
            if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError("expected_sha256 must be a 64-character hex digest")
            actual = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            changed = actual != expected
            identity = {"status": "mismatch" if changed else "match",
                        "flag": changed, "expected_sha256": expected,
                        "actual_sha256": actual}
        return {
            "decision": "review" if learned_flag or integrity_flag or identity["flag"] else "pass",
            "learned": {
                "flag": bool(learned_flag),
                "fusion_score": fused,
                "threshold": float(self.calibration["fusion_threshold_1pct"]),
                "reference_model_poison_probability": neural,
                "signal_model_poison_probability": signal,
                "v3_risk": v3_risk,
                "neural_clean_tail": neural_tail,
                "signal_clean_tail": signal_tail,
            },
            "integrity": ({**integrity, "flag": bool(integrity_flag),
                           "threshold": self.calibration["snapshot_threshold"],
                           "snapshot_source": self.snapshot.snapshot.manifest.get("source")}
                          if integrity is not None else {"status": "disabled"}),
            "identity": identity,
            "scope": ("research_source_label_screen; supplied_reference_and_expected_hash_"
                      "must_be_authenticated_by_caller"),
        }
