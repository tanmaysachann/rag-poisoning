"""Serve the frozen V24 research scanner without loading it on other routes."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
_SCAN_LOCK = Lock()
_SCANNER = None

REQUIRED = (
    "artifacts/multifamily_v20/model.safetensors",
    "artifacts/multifamily_v20/config.json",
    "artifacts/multifamily_v20/tokenizer.json",
    "artifacts/fluency_v21.joblib",
    "artifacts/research_detector_v3_hashing.joblib",
    "artifacts/research_detector_v3_hashing_web.joblib",
    "research_protocols/v3_web_detector.json",
    "artifacts/product_v24_calibration.joblib",
    "data/multifamily_v20/manifest.json",
    "data/multifamily_v20/evaluation.jsonl",
    "data/trusted_snapshot_v12/manifest.json",
    "data/trusted_snapshot_v12/beir_nq_fts5.sqlite",
    "results/fusion_v21.json",
    "results/snapshot_containment_v23.json",
    "results/product_v24.json",
    "results/adaptive_ppo_v24.json",
)


def _remote(path: str, body: dict | None = None) -> dict:
    base = os.getenv("V24_REMOTE_URL", "").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query:
        raise ValueError("V24_REMOTE_URL must be an HTTPS origin")
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(base + path, data=payload,
                      headers={"Content-Type": "application/json"} if payload else {},
                      method="POST" if payload else "GET")
    with urlopen(request, timeout=55) as response:
        return json.load(response)


def status() -> dict:
    if os.getenv("V24_REMOTE_URL"):
        try:
            result = _remote("/api/v24/status")
            return {**result, "deployment": "separate_v24_service"}
        except (OSError, ValueError, json.JSONDecodeError):
            return {"version": "v24", "available": False,
                    "missing_assets": ["remote_scanner_unavailable"],
                    "mode": "research_scanner", "live_rag_defense": False}
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    return {
        "version": "v24",
        "available": not missing,
        "model_and_snapshot_present": not missing,
        "snapshot_passages": 2_681_468 if not missing else None,
        "missing_assets": missing,
        "mode": "research_scanner",
        "live_rag_defense": False,
    }


def evidence() -> dict:
    paths = [ROOT / "results/product_v24.json", ROOT / "results/adaptive_ppo_v24.json"]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("V24 evaluation reports are unavailable")
    product, adaptive = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    return {
        "split": "previously_inspected_development",
        "product": {
            "combined": product["combined"],
            "classification": product["classification"],
            "calibration_clean_total": product["calibration_clean_total"],
            "calibration_clean_flags": product["calibration_clean_flags"],
            "identity_rule": product["identity_rule"],
        },
        "adaptive": {
            "training_questions_per_round": adaptive["training_questions_per_round"],
            "validation_questions": adaptive["validation_questions"],
            "rounds": adaptive["rounds"],
            "clean_answer_replay": adaptive["clean_answer_replay"],
        },
        "limitations": product["limits"],
    }


def sample() -> dict:
    """One pinned, published attack row for a reproducible UI demonstration."""
    manifest = json.loads((ROOT / "data/multifamily_v20/manifest.json").read_text(encoding="utf-8"))
    path = ROOT / "data/multifamily_v20/evaluation.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["splits"]["evaluation"]["sha256"]:
        raise ValueError("V24 example data changed")
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["query_id"] == "test415" and row["family"] == "poisonedrag_black":
                return {"query": row["query"], "reference": row["reference"],
                        "candidate": row["candidate"], "family": row["family"],
                        "split": "previously_inspected_development"}
    raise ValueError("Pinned V24 example is unavailable")


def scan(query: str, reference: str, candidate: str,
         expected_sha256: str | None = None) -> dict:
    """Serialize inference because the cached snapshot holds one SQLite handle."""
    global _SCANNER
    if os.getenv("V24_REMOTE_URL"):
        try:
            result = _remote("/api/v24/scan", {
                "query": query, "reference": reference, "candidate": candidate,
                "expected_sha256": expected_sha256,
            })
            if result.get("version") != "v24" or result.get("decision") not in {"pass", "review"}:
                raise ValueError("Unexpected remote scanner response")
            return result
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise FileNotFoundError("V24 remote scanner is unavailable") from error
    if not status()["available"]:
        raise FileNotFoundError("V24 model or pinned snapshot is unavailable")
    with _SCAN_LOCK:
        if _SCANNER is None:
            from detect.product_v22 import PoisonDocumentScanner
            _SCANNER = PoisonDocumentScanner(root=ROOT, integrity_mode="containment")
        return _SCANNER.assess(query, reference, candidate,
                               expected_sha256=expected_sha256)
