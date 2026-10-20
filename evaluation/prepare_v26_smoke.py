"""Save a paired V25-pass/V26-review targeted FiQA example for deployment checks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prepare() -> dict:
    cases = [json.loads(line) for line in (ROOT / "data/v26_fiqa_role_stress/cases.jsonl").open(encoding="utf-8")]
    trace = [json.loads(line) for line in (ROOT / "results/v26_fiqa_role_stress.cases.jsonl").open(encoding="utf-8")]
    if len(cases) != len(trace):
        raise ValueError("Complete the targeted FiQA role stress before preparing the smoke case")
    clean_pass = {case["query_id"] for case, score in zip(cases, trace)
                  if case["family"] == "reference" and not score["v26_flag"]}
    chosen = next((case for case, score in zip(cases, trace)
                   if case["family"] == "role_delimiter"
                   and case["query_id"] in clean_pass
                   and not score["v25_flag"] and score["v26_flag"]), None)
    if chosen is None:
        raise ValueError("No V25-pass/V26-review role attack was measured")
    clean = next(row for row in cases if row["query_id"] == chosen["query_id"]
                 and row["family"] == "reference")
    output = {}
    for name, row in (("attack", chosen), ("clean", clean)):
        body = {key: row[key] for key in ("query", "reference", "candidate")}
        path = ROOT / f"tmp/v26_smoke_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        output[name] = {"path": str(path), "query_id": row["query_id"],
                        "family": row["family"]}
    return output


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
