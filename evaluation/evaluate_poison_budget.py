"""Measure one-to-three accepted stealth passage copies on v1 validation."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from backend.research_lab import _detector, _inputs
from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark
from pipeline.research_inference import infer_research
from retrieval.hybrid_retriever import HybridRetriever
from security.integrity import load_runtime_manifest


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    os.environ["RAG_USE_MINILM"] = "0"
    validate_benchmark(ROOT_DIR / "data/benchmark")
    queries, corpus, attacks, _ = _inputs()
    if len(queries) != 75 or set(queries) != set(attacks["stealth"]):
        raise ValueError("Expected all 75 v1 validation stealth schedules")
    detector = _detector()
    source_path = ROOT_DIR / "data/benchmark/validation/corpus.jsonl"
    attacks_path = ROOT_DIR / "results/attack_baseline_validation_stealth.jsonl"
    next_id = max(corpus) + 1
    records = []
    with tempfile.TemporaryDirectory(prefix="sentinel-poison-budget-") as directory:
        work = Path(directory)
        for budget in (1, 2, 3):
            for index, (qid, query) in enumerate(queries.items(), 1):
                attack = attacks["stealth"][qid]
                target_id = int(attack["attack_doc_id"])
                if target_id not in query["support_doc_ids"]:
                    raise ValueError(f"Attack target is not labeled support: {qid}")
                poison = attack["text"]
                copy_ids = [next_id + offset for offset in range(budget - 1)]
                attack_ids = {target_id, *copy_ids}
                staged_rows = [
                    {**doc, "text": poison} if doc["doc_id"] == target_id else doc
                    for doc in corpus.values()
                ]
                staged_rows.extend({"doc_id": doc_id, "text": poison} for doc_id in copy_ids)
                case_path = work / f"budget-{budget}-{index}"
                case_path.mkdir()
                staged = case_path / "staged.jsonl"
                with staged.open("w", encoding="utf-8", newline="\n") as handle:
                    for doc in staged_rows:
                        handle.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
                started = time.perf_counter()
                retriever = HybridRetriever(staged, artifact_dir=case_path / "index",
                                            write_integrity_manifest=True)
                if retriever.embedder.backend != "hashing":
                    raise RuntimeError("Budget probe requires deterministic hashing retrieval")
                manifest = load_runtime_manifest(retriever.integrity_path)
                if manifest is None:
                    raise RuntimeError("Accepted-ingest case has no manifest")
                result = infer_research(query["question"], retriever, manifest, detector)
                observed = [row for row in result["documents"] if row["doc_id"] in attack_ids]
                off = result["undefended"]
                on = result["defended"]
                wrong = attack["wrong_answer"].casefold()
                records.append({
                    "qid": qid, "budget": budget, "attack_doc_ids": sorted(attack_ids),
                    "wrong_answer": attack["wrong_answer"],
                    "retrieved_attack_doc_ids": [row["doc_id"] for row in observed],
                    "quarantined_attack_doc_ids": [row["doc_id"] for row in observed
                                                    if row["decision"] == "quarantine"],
                    "attack_risks": {str(row["doc_id"]): row["risk"] for row in observed},
                    "undefended_source_doc_id": off["source_doc_id"],
                    "defended_source_doc_id": on["source_doc_id"],
                    "undefended_answer": off["answer"], "defended_answer": on["answer"],
                    "undefended_attack_success": off["source_doc_id"] in attack_ids and wrong in off["answer"].casefold(),
                    "defended_attack_success": on["source_doc_id"] in attack_ids and wrong in on["answer"].casefold(),
                    "clean_alias_recovered": any(alias.casefold() in on["answer"].casefold()
                                                 for alias in query["answer_aliases"]),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                })
                if index % 25 == 0:
                    print(f"budget {budget}: {index}/75", flush=True)
    summaries = {}
    for budget in (1, 2, 3):
        rows = [row for row in records if row["budget"] == budget]
        summaries[str(budget)] = {
            "cases": len(rows),
            "attack_retrieved_cases": sum(bool(row["retrieved_attack_doc_ids"]) for row in rows),
            "retrieved_attack_documents": sum(len(row["retrieved_attack_doc_ids"]) for row in rows),
            "quarantined_attack_documents": sum(len(row["quarantined_attack_doc_ids"]) for row in rows),
            "undefended_attack_successes": sum(row["undefended_attack_success"] for row in rows),
            "defended_attack_successes": sum(row["defended_attack_success"] for row in rows),
            "clean_alias_recovered": sum(row["clean_alias_recovered"] for row in rows),
            "median_case_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 2),
        }
    output = ROOT_DIR / "results/poison_budget_stealth_validation.json"
    summary = {
        "split": "validation", "surface": "accepted_ingest", "retrieval_backend": "hashing",
        "attack_family": "saved stealth answer substitution replicated verbatim",
        "corpus_sha256": _hash(source_path), "attack_schedule_sha256": _hash(attacks_path),
        "budgets": summaries,
        "note": "One original support passage is replaced; budgets two and three add identical copies with new document IDs. Copies share one attacker origin and are not independent evidence. The v1 frozen test was not reopened.",
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
