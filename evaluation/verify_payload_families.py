"""Verify saved payload-family probe summaries against raw validation rows."""

from __future__ import annotations

import json
from pathlib import Path

from attack.payload_families import FAMILIES, family_success
from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark
from data.validate_multisupport_benchmark import read_jsonl


def verify(root: Path = ROOT_DIR) -> dict:
    validate_benchmark(root / "data/benchmark")
    queries = {row["qid"]: row for row in read_jsonl(root / "data/benchmark/validation/queries.jsonl")}
    summary = json.loads((root / "results/payload_families_validation.json").read_text(encoding="utf-8"))
    if summary["split"] != "validation" or set(summary["families"]) != set(FAMILIES):
        raise ValueError("Payload-family summary has wrong split or families")
    checked = {}
    for family in FAMILIES:
        rows = read_jsonl(root / f"results/payload_families_validation_{family}.cases.jsonl")
        if len(rows) != len(queries) or {row["qid"] for row in rows} != set(queries):
            raise ValueError(f"Family {family}: missing or duplicate validation questions")
        if len({row["attack_doc_id"] for row in rows}) != len(rows):
            raise ValueError(f"Family {family}: attack document IDs are not unique")
        for row in rows:
            if row["family"] != family:
                raise ValueError(f"Family {family}: mislabeled row {row['qid']}")
            if row["undefended_attack_success"] != family_success(
                    family, row["undefended_answer"], row["undefended_source"], row["attack_doc_id"]):
                raise ValueError(f"Family {family}: undefended success flag differs: {row['qid']}")
            if row["defended_attack_success"] != family_success(
                    family, row["defended_answer"], row["defended_source"], row["attack_doc_id"]):
                raise ValueError(f"Family {family}: defended success flag differs: {row['qid']}")
            if row["attack_quarantined"] and not row["attack_retrieved"]:
                raise ValueError(f"Family {family}: quarantined without retrieval: {row['qid']}")
        block = summary["families"][family]
        expected = {
            "cases": len(rows),
            "attack_retrieved": sum(row["attack_retrieved"] for row in rows),
            "attack_quarantined": sum(row["attack_quarantined"] for row in rows),
            "undefended_attack_successes": sum(row["undefended_attack_success"] for row in rows),
            "defended_attack_successes": sum(row["defended_attack_success"] for row in rows),
            "clean_answer_recovered": sum(row["clean_answer_recovered"] for row in rows),
            "abstentions": sum(row["abstained"] for row in rows),
        }
        if any(block[key] != value for key, value in expected.items()):
            raise ValueError(f"Family {family}: summary disagrees with its cases")
        checked[family] = expected["defended_attack_successes"]
    return {"status": "verified", "families": checked, "test_attacks_evaluated": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
