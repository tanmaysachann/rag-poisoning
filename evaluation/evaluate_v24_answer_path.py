"""Paired admitted-source V24 answer replay on inspected v3 validation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from backend.research_lab import ROOT, case_catalog, run_case


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    def counts(subset: list[dict]) -> dict:
        return {
            "cases": len(subset),
            "retrieved_attacks": sum(row["attack_retrieved"] for row in subset),
            "undefended_attack_successes": sum(row["undefended_attack_success"] for row in subset),
            "v3_attack_successes": sum(row["v3_attack_success"] for row in subset),
            "v24_attack_successes": sum(row["v24_attack_success"] for row in subset),
            "v3_clean_alias_recovered": sum(row["v3_clean_alias_recovered"] for row in subset),
            "v24_clean_alias_recovered": sum(row["v24_clean_alias_recovered"] for row in subset),
            "v24_abstentions": sum(row["v24_answer_source"] is None for row in subset),
            "v24_retrieved_attack_reviews": sum(row["v24_attack_reviews"] for row in subset),
            "v24_scanner_attack_reviews": sum(row["v24_scanner_attack_reviews"] for row in subset),
            "v24_snapshot_recoveries": sum(row["v24_snapshot_recovered"] for row in subset),
        }
    return {"overall": counts(rows), **{
        strategy: counts([row for row in rows if row["strategy"] == strategy])
        for strategy in ("stealth", "greedy")
    }}


def paired_interval(rows: list[dict], *, seed: int = 20261020,
                    draws: int = 2000) -> list[float]:
    groups = {}
    for row in rows:
        groups.setdefault(row["qid"], []).append(row)
    qids = sorted(groups)
    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        sample = [row for _ in qids for row in groups[rng.choice(qids)]]
        deltas.append((sum(row["v24_attack_success"] for row in sample)
                       - sum(row["v3_attack_success"] for row in sample)) / len(sample))
    deltas.sort()
    return [deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]]


def evaluate(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Answer report already exists: {output}")
    source_names = (
        "data/benchmark_v3/manifest.json",
        "results/v3_attack_validation_stealth.jsonl",
        "results/v3_attack_validation_greedy.jsonl",
        "artifacts/multifamily_v20/model.safetensors",
        "artifacts/product_v24_calibration.joblib",
    )
    inputs = {name: sha(ROOT / name) for name in source_names}
    cases = case_catalog()["cases"]
    qids = [case["qid"] for case in cases]
    if len(qids) != 150 or len(set(qids)) != 150:
        raise ValueError("Unexpected v3 validation question inventory")
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = output.with_suffix(".cases.jsonl")
    prior = [json.loads(line) for line in trace.open(encoding="utf-8") if line.strip()] if trace.exists() else []
    schedule = [(strategy, qid) for strategy in ("stealth", "greedy") for qid in qids]
    if [(row["strategy"], row["qid"]) for row in prior] != schedule[:len(prior)]:
        raise ValueError("Existing replay trace does not match the frozen schedule")
    with trace.open("a", encoding="utf-8", newline="\n") as sink:
        for index, (strategy, qid) in enumerate(schedule[len(prior):], len(prior) + 1):
            case = run_case(qid, strategy, "accepted_ingest", v24_gate=True)
            v24 = case["v24_answer_path"]
            record = {
                "qid": qid, "strategy": strategy,
                "attack_retrieved": case["attack_retrieved"],
                "undefended_attack_success": case["undefended_attack_success"],
                "v3_attack_success": case["defended_attack_success"],
                "v24_attack_success": v24["attack_source_success"],
                "v3_clean_alias_recovered": case["clean_alias_recovered"],
                "v24_clean_alias_recovered": v24["clean_alias_recovered"],
                "v3_answer_source": case["defended"]["source_doc_id"],
                "v24_answer_source": v24["defended"]["source_doc_id"],
                "v24_attack_reviews": v24["retrieved_attack_docs_reviewed"],
                "v24_scanner_attack_reviews": sum(
                    row["doc_id"] in case["attack_doc_ids"] and row["scanner_decision"] == "review"
                    for row in v24["documents"]),
                "v24_snapshot_recovered": v24["snapshot_recovery"]["decision"] == "recover",
                "v24_review_doc_ids": v24["review_doc_ids"],
                "v24_answer": v24["defended"]["answer"],
                "v24_gate_latency_ms": v24["latency_ms"],
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            prior.append(record)
            if index % 10 == 0:
                print(f"Replayed {index}/{len(schedule)} paired answer cases", flush=True)
    if any(sha(ROOT / name) != expected for name, expected in inputs.items()):
        raise ValueError("Input changed during answer replay")
    report = {
        "protocol": "v24-admitted-source-paired-v3-validation-answer-replay-v2",
        "split": "previously_inspected_validation",
        "revision_status": "admission_policy_added_after_inspecting_initial_v24_answer_replay",
        "answer_backend": "hashing_extractive",
        "attack_surface": "accepted_ingest; stealth replaces a known document, greedy inserts a new one",
        "reference_source": "benchmark support passage checked against pre-attack corpus SHA-256 manifest",
        "inputs_sha256": inputs, "cases_sha256": sha(trace),
        "summary": summarize(prior),
        "paired_question_cluster_95pct_ci_v24_minus_v3_attack_success_rate": paired_interval(prior),
        "limits": [
            "The same 150 validation questions were inspected during prior development.",
            "The benchmark support passage is supplied by a qrel; a public RAG service needs its own authenticated reference selection.",
            "The extractor emits source spans and is not a hosted LLM answer-quality test.",
            "Any later detector tuning invalidates this as a fresh evaluation.",
            "Only source bytes admitted to the pinned snapshot may supply primary answers; newly inserted clean documents also require admission.",
            "This revision was designed after the initial V24 answer replay exposed a new-document failure and is not blind evidence.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/v24_answer_path_validation_admitted.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output), indent=2))
