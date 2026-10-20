"""Pinned V26 scanner service and optional HTTPS proxy for the small site."""

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
REQUIRED_REPORTS = ("results/v26_rule_frozen.json", "results/v26_fiqa_holdout.json",
                    "results/v26_fiqa_answer_path.json",
                    "results/v26_fiqa_role_stress_frozen.json",
                    "results/v26_fiqa_role_stress.json",
                    "results/v26_nfcorpus_development_replay.json")
SERVING_INPUTS = (
    "detect/product_v26.py", "detect/role_boundary_v26.py",
    "detect/product_v25.py", "detect/product_v22.py",
    "artifacts/role_boundary_v26.joblib",
    "artifacts/multifamily_v20/model.safetensors",
    "artifacts/product_v24_calibration.joblib",
    "artifacts/research_detector_v3_hashing_web.joblib",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remote(path: str, body: dict | None = None) -> dict:
    base = os.getenv("V26_REMOTE_URL", "").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query:
        raise ValueError("V26_REMOTE_URL must be an HTTPS origin")
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(base + path, data=payload,
                      headers={"Content-Type": "application/json"} if payload else {},
                      method="POST" if payload else "GET")
    with urlopen(request, timeout=55) as response:
        return json.load(response)


def _verify() -> dict:
    rule = json.loads((ROOT / REQUIRED_REPORTS[0]).read_text(encoding="utf-8"))
    if rule["protocol"] != "v26-role-adaptive-frozen-rule-v1":
        raise ValueError("V26 rule manifest changed")
    for name in SERVING_INPUTS:
        if _sha(ROOT / name) != rule["input_sha256"][name]:
            raise ValueError(f"Pinned V26 serving input changed: {name}")
    holdout = json.loads((ROOT / REQUIRED_REPORTS[1]).read_text(encoding="utf-8"))
    if holdout["rule_sha256"] != _sha(ROOT / REQUIRED_REPORTS[0]):
        raise ValueError("V26 holdout does not match frozen rule")
    answer = json.loads((ROOT / REQUIRED_REPORTS[2]).read_text(encoding="utf-8"))
    answer_inputs = {name.replace("\\", "/"): digest
                     for name, digest in answer["inputs_sha256"].items()}
    if answer_inputs["results/v26_fiqa_holdout.json"] != _sha(ROOT / REQUIRED_REPORTS[1]):
        raise ValueError("V26 answer replay does not match scanner holdout")
    stress_schedule = json.loads((ROOT / REQUIRED_REPORTS[3]).read_text(encoding="utf-8"))
    stress = json.loads((ROOT / REQUIRED_REPORTS[4]).read_text(encoding="utf-8"))
    if stress["schedule_sha256"] != _sha(ROOT / REQUIRED_REPORTS[3]) or stress_schedule[
            "input_sha256"]["results/v26_rule_frozen.json"] != _sha(ROOT / REQUIRED_REPORTS[0]):
        raise ValueError("V26 follow-up stress does not match pinned inputs")
    development = json.loads((ROOT / REQUIRED_REPORTS[5]).read_text(encoding="utf-8"))
    if development["status"] != "development_replay_not_blind":
        raise ValueError("V26 NFCorpus development replay changed")
    return {"rule": rule, "holdout": holdout, "answer_path": answer,
            "targeted_stress": stress, "development_replay": development}


def status() -> dict:
    if os.getenv("V26_REMOTE_URL"):
        try:
            return {**_remote("/api/v26/status"), "deployment": "separate_v26_service"}
        except (OSError, ValueError, json.JSONDecodeError):
            return {"version": "v26", "available": False,
                    "missing_assets": ["remote_scanner_unavailable"]}
    missing = [name for name in (*REQUIRED_REPORTS, *SERVING_INPUTS)
               if not (ROOT / name).is_file()]
    missing.extend(v24_service.status()["missing_assets"])
    if not missing:
        try:
            _verify()
        except (ValueError, KeyError, json.JSONDecodeError):
            missing = ["v26_pinned_evidence_mismatch"]
    return {"version": "v26", "available": not missing,
            "missing_assets": missing, "mode": "role_adaptive_research_scanner"}


def evidence() -> dict:
    if os.getenv("V26_REMOTE_URL"):
        return _remote("/api/v26/evidence")
    return _verify()


def scan(query: str, reference: str, candidate: str,
         expected_sha256: str | None = None) -> dict:
    global _SCANNER
    if os.getenv("V26_REMOTE_URL"):
        try:
            result = _remote("/api/v26/scan", {
                "query": query, "reference": reference, "candidate": candidate,
                "expected_sha256": expected_sha256,
            })
            if result.get("version") != "v26" or result.get("decision") not in {"pass", "review"}:
                raise ValueError("Unexpected V26 remote response")
            return result
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise FileNotFoundError("V26 remote scanner is unavailable") from error
    with _LOCK:
        if _SCANNER is None:
            if not status()["available"]:
                raise FileNotFoundError("V26 model or frozen evidence is unavailable")
            from detect.product_v26 import RoleAdaptiveScanner
            _SCANNER = RoleAdaptiveScanner(root=ROOT)
        return {**_SCANNER.assess(query, reference, candidate,
                                  expected_sha256=expected_sha256), "version": "v26"}
