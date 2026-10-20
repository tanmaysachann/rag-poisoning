"""Optional V25 scanner service; deployment requires a completed holdout report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from backend import v24_service


ROOT = Path(__file__).resolve().parents[1]
_LOCK = Lock()
_SCANNER = None
REQUIRED_REPORTS = ("results/v25_rule_frozen.json", "results/v25_nfcorpus_holdout.json")
SERVING_INPUTS = (
    "detect/product_v25.py", "detect/product_v22.py",
    "artifacts/multifamily_v20/model.safetensors",
    "artifacts/product_v24_calibration.joblib",
    "artifacts/research_detector_v3_hashing_web.joblib",
)


def _remote(path: str, body: dict | None = None) -> dict:
    base = os.getenv("V25_REMOTE_URL", "").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query:
        raise ValueError("V25_REMOTE_URL must be an HTTPS origin")
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(base + path, data=payload,
                      headers={"Content-Type": "application/json"} if payload else {},
                      method="POST" if payload else "GET")
    with urlopen(request, timeout=55) as response:
        return json.load(response)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify() -> dict:
    rule = json.loads((ROOT / REQUIRED_REPORTS[0]).read_text(encoding="utf-8"))
    if rule["protocol"] != "v25-reference-adaptive-frozen-rule-v1":
        raise ValueError("V25 rule manifest changed")
    for name in SERVING_INPUTS:
        if _sha(ROOT / name) != rule["input_sha256"][name]:
            raise ValueError(f"Pinned V25 serving input changed: {name}")
    report = json.loads((ROOT / REQUIRED_REPORTS[1]).read_text(encoding="utf-8"))
    if report["rule_sha256"] != _sha(ROOT / REQUIRED_REPORTS[0]):
        raise ValueError("V25 holdout was not scored with this frozen rule")
    return report


def status() -> dict:
    if os.getenv("V25_REMOTE_URL"):
        try:
            return {**_remote("/api/v25/status"),
                    "deployment": "separate_v25_service"}
        except (OSError, ValueError, json.JSONDecodeError):
            return {"version": "v25", "available": False,
                    "missing_assets": ["remote_scanner_unavailable"]}
    missing = [name for name in (*REQUIRED_REPORTS, *SERVING_INPUTS)
               if not (ROOT / name).is_file()]
    missing.extend(v24_service.status()["missing_assets"])
    if not missing:
        try:
            _verify()
        except (ValueError, KeyError, json.JSONDecodeError):
            missing = ["v25_pinned_evidence_mismatch"]
    return {"version": "v25", "available": not missing,
            "missing_assets": missing, "mode": "reference_adaptive_research_scanner"}


def evidence() -> dict:
    if os.getenv("V25_REMOTE_URL"):
        return _remote("/api/v25/evidence")
    report = _verify()
    return {"rule": json.loads((ROOT / REQUIRED_REPORTS[0]).read_text(encoding="utf-8")),
            "holdout": report}


def scan(query: str, reference: str, candidate: str,
         expected_sha256: str | None = None) -> dict:
    global _SCANNER
    if os.getenv("V25_REMOTE_URL"):
        try:
            result = _remote("/api/v25/scan", {
                "query": query, "reference": reference, "candidate": candidate,
                "expected_sha256": expected_sha256,
            })
            if result.get("version") != "v25" or result.get("decision") not in {"pass", "review"}:
                raise ValueError("Unexpected V25 remote response")
            return result
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise FileNotFoundError("V25 remote scanner is unavailable") from error
    with _LOCK:
        if _SCANNER is None:
            if not status()["available"]:
                raise FileNotFoundError("V25 model or frozen evidence is unavailable")
            from detect.product_v25 import ReferenceAdaptiveScanner
            _SCANNER = ReferenceAdaptiveScanner(root=ROOT)
        result = _SCANNER.assess(query, reference, candidate,
                                 expected_sha256=expected_sha256)
        return {**result, "version": "v25"}
