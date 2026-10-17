"""Runtime implementation of the presentation's four answer-source probes.

The gate evaluates the selected answer source only. Its four anomaly scores are
rank-calibrated together on clean training examples, then the combined score
can quarantine that source and force abstention. SHA-256 and signed provenance
remain separate, mandatory trust controls in the strict serving path.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from detect.model_probes import MiniLMModelProbes, ProbeReference
from detect.research_detector import threshold_for_fpr
from detect.slm_srq import LocalSLMProbe, score_srq
from retrieval.hybrid_retriever import TextEmbedder


PROBE_NAMES = ("s1_hidden_state_mahalanobis", "s2_attention_deviation", "s3_slm_srq", "s4_loo_influence")
CALIBRATION_VERSION = "four-probe-joint-clean-rank-v2"


class CachedWordEmbedder:
    """Cache MiniLM word vectors used by the SRQ computation."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.model_name = backend.model_name
        self._vectors: dict[str, np.ndarray] = {}

    def encode(self, words: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(word for word in words if word not in self._vectors))
        if missing:
            encoded = self.backend.encode(missing)
            self._vectors.update(zip(missing, encoded))
        if not words:
            return np.empty((0, self.backend.encode(["sentinel"])[0].shape[0]), dtype=np.float32)
        return np.stack([self._vectors[word] for word in words]).astype(np.float32)


def _midrank_percentile(value: float, reference: np.ndarray) -> float:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    left = int(np.searchsorted(ordered, value, side="left"))
    right = int(np.searchsorted(ordered, value, side="right"))
    return float((left + right) / (2.0 * max(len(ordered), 1)))


def build_calibration(raw_rows: list[dict[str, float]], target_fpr: float = 0.05) -> dict:
    """Create a joint max-rank gate using leave-one-out clean calibration ranks."""
    if len(raw_rows) < 20:
        raise ValueError("Four-probe calibration requires at least 20 clean training rows")
    columns = {name: np.asarray([row[name] for row in raw_rows], dtype=np.float64) for name in PROBE_NAMES}
    if any(not np.isfinite(values).all() for values in columns.values()):
        raise ValueError("Probe calibration contains non-finite scores")
    joint_clean = []
    for index, row in enumerate(raw_rows):
        component_ranks = []
        for name in PROBE_NAMES:
            reference_without_row = np.delete(columns[name], index)
            component_ranks.append(_midrank_percentile(row[name], reference_without_row))
        joint_clean.append(max(component_ranks))
    return {
        "calibration_version": CALIBRATION_VERSION,
        "calibration_method": "leave-one-out-clean-rank-threshold-v1",
        "probe_names": list(PROBE_NAMES),
        "target_clean_fpr": float(target_fpr),
        "rows": len(raw_rows),
        "reference_scores": {name: np.sort(values) for name, values in columns.items()},
        "joint_threshold": threshold_for_fpr(np.asarray(joint_clean), target_fpr),
        "joint_calibration_scores": np.asarray(joint_clean, dtype=np.float64),
    }


class FourProbeGate:
    """Run S1-S4 on a candidate answer source and return a calibrated gate result."""

    def __init__(self, bundle: dict, *, local_files_only: bool = True) -> None:
        if bundle.get("calibration_version") != CALIBRATION_VERSION:
            raise ValueError("Unsupported four-probe calibration version")
        if tuple(bundle.get("probe_names", ())) != PROBE_NAMES:
            raise ValueError("Calibration does not contain the four expected probes")
        self.bundle = bundle
        from config import LLM_MODEL

        if Path(str(LLM_MODEL)).resolve() != Path(str(bundle["qwen_model"])).resolve():
            raise ValueError("Set RAG_LLM_MODEL to the calibrated local Qwen model before starting the API")
        ref = bundle["minilm_reference"]
        self.reference = ProbeReference(
            np.asarray(ref["mean"], dtype=np.float64),
            np.asarray(ref["precision"], dtype=np.float64),
            float(ref["attention_mean"]), float(ref["attention_std"]),
        )
        self.minilm = MiniLMModelProbes(local_files_only=local_files_only)
        self.word_embedder = CachedWordEmbedder(TextEmbedder(preferred_backend="sentence-transformers"))
        if self.word_embedder.backend.backend != "sentence-transformers":
            raise RuntimeError("SRQ requires the calibrated MiniLM sentence encoder")
        self.qwen = LocalSLMProbe(self.word_embedder, model_name=bundle["qwen_model"])
        model_revision = self.minilm.model_revision
        if bundle.get("minilm_model_revision") != model_revision:
            raise ValueError("MiniLM probe model revision differs from calibration")
        qwen_weight = Path(bundle["qwen_model"]) / "model.safetensors"
        if not qwen_weight.is_file() or hashlib.sha256(qwen_weight.read_bytes()).hexdigest() != bundle.get("qwen_weight_sha256"):
            raise ValueError("Qwen probe weights differ from calibration or are unavailable")

    def score(self, query: str, document: str, *, loo_influence: float) -> dict:
        started = time.perf_counter()
        vector, concentration = self.minilm.extract(query, document)
        model_probe = self.reference.score(vector, concentration)
        minilm_ms = (time.perf_counter() - started) * 1000
        srq = score_srq(query, document, self.qwen.preliminary_response(query, document), self.word_embedder)
        qwen_ms = (time.perf_counter() - started) * 1000 - minilm_ms
        raw = {
            "s1_hidden_state_mahalanobis": float(model_probe["layer3_mahalanobis"]),
            "s2_attention_deviation": float(model_probe["attention_deviation"]),
            "s3_slm_srq": float(srq["raw_srq"]),
            "s4_loo_influence": float(loo_influence),
        }
        components = {
            name: _midrank_percentile(value, self.bundle["reference_scores"][name])
            for name, value in raw.items()
        }
        risk = max(components.values())
        threshold = float(self.bundle["joint_threshold"])
        return {
            "calibration_version": CALIBRATION_VERSION,
            "raw_scores": raw,
            "rank_risks": components,
            "combined_risk": risk,
            "threshold": threshold,
            "decision": "quarantine" if risk >= threshold else "accept",
            "triggered_probes": [name for name, value in components.items() if value >= threshold],
            "timings_ms": {
                "s1_s2_minilm": round(minilm_ms, 2),
                "s3_qwen_srq": round(qwen_ms, 2),
            },
        }


def load_bundle(path: Path) -> dict:
    """Load a locally built, train-calibrated four-probe bundle."""
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError("Invalid four-probe calibration artifact")
    return bundle


def save_bundle(path: Path, bundle: dict) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def write_calibration_summary(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: value for key, value in bundle.items()
        if key not in {"reference_scores", "joint_calibration_scores", "minilm_reference"}
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
