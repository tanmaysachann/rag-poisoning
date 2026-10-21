"""Pin one previously scored V25-pass/V26-review case for the website demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/v26_fiqa_role_stress/cases.jsonl"
TRACE = ROOT / "results/v26_fiqa_role_stress.cases.jsonl"
REPORT = ROOT / "results/v26_fiqa_role_stress.json"
OUTPUT = ROOT / "results/v26_presentation_sample.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["trace_sha256"] != sha(TRACE):
        raise ValueError("Targeted stress trace changed")
    cases = [json.loads(line) for line in CASES.open(encoding="utf-8") if line.strip()]
    trace = [json.loads(line) for line in TRACE.open(encoding="utf-8") if line.strip()]
    if len(cases) != len(trace) or len(cases) != 360:
        raise ValueError("Targeted stress schedule changed")
    chosen = None
    for row, score in zip(cases, trace):
        if (row["query_id"], row["candidate_id"], hashlib.sha256(row["candidate"].encode()).hexdigest()) != (
                score["query_id"], score["candidate_id"], score["candidate_sha256"]):
            raise ValueError("Scored trace and cases differ")
        if (chosen is None and row["family"] == "role_delimiter"
                and not score["v25_flag"] and score["v26_flag"]):
            chosen = row
    if chosen is None:
        raise ValueError("No verified V26-specific role example exists")
    clean = next(row for row in cases if row["query_id"] == chosen["query_id"]
                 and row["family"] == "reference")
    clean_score = next(score for score in trace if score["query_id"] == chosen["query_id"]
                       and score["family"] == "reference")
    if clean_score["v25_flag"] or clean_score["v26_flag"]:
        raise ValueError("Chosen reference did not pass both scanners")
    sample = {
        "protocol": "v26-pinned-presentation-sample-v1",
        "evidence_status": "post_hoc_known_pattern_targeted_stress",
        "query_id": chosen["query_id"], "family": chosen["family"],
        "query": chosen["query"], "reference": chosen["reference"],
        "attack": chosen["candidate"], "clean": clean["candidate"],
        "expected": {"attack_v25": "pass", "attack_v26": "review",
                     "clean_v25": "pass", "clean_v26": "pass"},
        "report_sha256": sha(REPORT), "cases_sha256": sha(CASES),
        "trace_sha256": sha(TRACE),
        "note": "Known NFCorpus role-delimiter pattern on another FiQA question; selected after first FiQA aggregate inspection.",
    }
    OUTPUT.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"query_id": sample["query_id"], "family": sample["family"],
            "sample_sha256": sha(OUTPUT), "report_sha256": sample["report_sha256"]}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
