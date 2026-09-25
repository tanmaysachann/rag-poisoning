"""Report 95% intervals from saved paired validation or frozen test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import RESULTS_DIR
from evaluation.metrics import paired_bootstrap_difference, wilson_interval


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(values: list[bool]) -> dict:
    count = sum(values)
    lower, upper = wilson_interval(count, len(values))
    return {"count": count, "total": len(values), "rate": count / len(values),
            "wilson_95": [lower, upper]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--profiles", nargs="+", choices=("hashing", "minilm_ranker"),
                        default=["hashing", "minilm_ranker"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profiles = {}
    for strategy in ("greedy", "stealth"):
        for answer_profile, suffix in (("hashing", ""), ("minilm_ranker", "_minilm_ranker")):
            if answer_profile not in args.profiles:
                continue
            path = RESULTS_DIR / f"defense_{args.split}_{strategy}_accepted_ingest{suffix}.cases.jsonl"
            rows = _read_jsonl(path)
            if len(rows) != 75 or len({row["qid"] for row in rows}) != 75:
                raise ValueError(f"Expected 75 unique {args.split} queries in {path}")
            before = [bool(row["undefended_attack_success"]) for row in rows]
            after = [bool(row["defended_attack_success"]) for row in rows]
            profiles[f"{strategy}_{answer_profile}"] = {
                "undefended_attack_success": _rate(before),
                "defended_attack_success": _rate(after),
                "absolute_attack_success_reduction": sum(b - a for b, a in zip(before, after)) / len(rows),
                "paired_bootstrap_reduction_95": paired_bootstrap_difference(before, after),
                "clean_alias_recovery": _rate([bool(row["clean_answer_recovered"]) for row in rows]),
                "attack_quarantine": _rate([bool(row["attack_quarantined"]) for row in rows]),
                "abstention": _rate([bool(row["abstained"]) for row in rows]),
            }
    output = {
        "split": args.split, "cases_per_profile": 75,
        "confidence_level": 0.95,
        "note": (
            "Descriptive intervals after validation-stage development; not final test generalization"
            if args.split == "validation" else
            "Frozen v1 test protocol; intervals describe sampling uncertainty for this 75-query corpus, not all RAG settings"
        ),
        "profiles": profiles,
    }
    path = args.output or RESULTS_DIR / f"{args.split}_intervals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
