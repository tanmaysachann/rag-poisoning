"""Check the saved stealth poison-budget study against raw validation rows."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from backend.research_lab import _inputs
from config import ROOT_DIR


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path = ROOT_DIR) -> dict:
    summary_path = root / "results/poison_budget_stealth_validation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in summary_path.with_suffix(".cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queries, _, attacks, _ = _inputs()
    expected = {(budget, qid) for budget in (1, 2, 3) for qid in queries}
    observed = Counter((row["budget"], row["qid"]) for row in cases)
    if (summary["split"] != "validation" or summary["surface"] != "accepted_ingest"
            or summary["retrieval_backend"] != "hashing"
            or set(summary["budgets"]) != {"1", "2", "3"}
            or summary["corpus_sha256"] != _hash(root / "data/benchmark/validation/corpus.jsonl")
            or summary["attack_schedule_sha256"] != _hash(root / "results/attack_baseline_validation_stealth.jsonl")
            or set(observed) != expected or any(count != 1 for count in observed.values())):
        raise ValueError("Poison-budget inputs, split, or paired cases differ")
    for row in cases:
        qid, budget = row["qid"], row["budget"]
        attack = attacks["stealth"][qid]
        ids = set(row["attack_doc_ids"])
        retrieved = set(row["retrieved_attack_doc_ids"])
        quarantined = set(row["quarantined_attack_doc_ids"])
        wrong = attack["wrong_answer"].casefold()
        if (len(ids) != budget or int(attack["attack_doc_id"]) not in ids
                or row["wrong_answer"] != attack["wrong_answer"]
                or not quarantined.issubset(retrieved) or not retrieved.issubset(ids)
                or set(map(int, row["attack_risks"])) != retrieved
                or any(not 0 <= risk <= 1 for risk in row["attack_risks"].values())
                or row["undefended_attack_success"] != (row["undefended_source_doc_id"] in ids
                                                      and wrong in row["undefended_answer"].casefold())
                or row["defended_attack_success"] != (row["defended_source_doc_id"] in ids
                                                     and wrong in row["defended_answer"].casefold())
                or row["clean_alias_recovered"] != any(alias.casefold() in row["defended_answer"].casefold()
                                                        for alias in queries[qid]["answer_aliases"])
                or row["latency_ms"] < 0):
            raise ValueError(f"Poison-budget row is inconsistent: {budget} {qid}")
    for budget in (1, 2, 3):
        rows = [row for row in cases if row["budget"] == budget]
        counts = {
            "cases": len(rows),
            "attack_retrieved_cases": sum(bool(row["retrieved_attack_doc_ids"]) for row in rows),
            "retrieved_attack_documents": sum(len(row["retrieved_attack_doc_ids"]) for row in rows),
            "quarantined_attack_documents": sum(len(row["quarantined_attack_doc_ids"]) for row in rows),
            "undefended_attack_successes": sum(row["undefended_attack_success"] for row in rows),
            "defended_attack_successes": sum(row["defended_attack_success"] for row in rows),
            "clean_alias_recovered": sum(row["clean_alias_recovered"] for row in rows),
            "median_case_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 2),
        }
        if summary["budgets"][str(budget)] != counts:
            raise ValueError(f"Poison-budget {budget} aggregate differs")
    original = json.loads((root / "results/defense_validation_stealth_accepted_ingest.json").read_text(encoding="utf-8"))
    one = summary["budgets"]["1"]
    if (one["undefended_attack_successes"] != round(original["undefended_attack_success_rate"] * 75)
            or one["defended_attack_successes"] != round(original["defended_attack_success_rate"] * 75)
            or one["quarantined_attack_documents"] != round(original["attack_quarantine_rate"] * 75)):
        raise ValueError("One-copy case disagrees with the saved stealth baseline")
    return {"status": "verified", "budgets": 3, "paired_validation_cases": len(cases)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
