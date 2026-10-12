"""Benchmark-scale payload-family probe on validation only.

Stages fixed-template instruction-takeover, URL-insertion, and refusal-jamming
insertions (one new document per query) and runs the hashing/extractive
profile defense OFF/ON over the same candidates. Never reads the test split.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import joblib

from attack.harness import stage_document_attack
from attack.payload_families import FAMILIES, attach_payload, family_success
from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from detect.research_detector import ResearchDetector
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import HybridRetriever
from security.integrity import load_runtime_manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=75)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    validate_benchmark(ROOT_DIR / "data/benchmark")
    detector = joblib.load(ARTIFACTS_DIR / "research_detector_hashing.joblib")
    if not isinstance(detector, ResearchDetector) or detector.embedding_model != "sklearn-hashing-384":
        raise ValueError("This probe requires the frozen hashing detector")
    os.environ["RAG_USE_MINILM"] = "0"
    benchmark = ROOT_DIR / "data/benchmark"
    corpus_path = benchmark / "validation/corpus.jsonl"
    docs = _read_jsonl(corpus_path)
    by_id = {doc["doc_id"]: doc for doc in docs}
    queries = _read_jsonl(benchmark / "validation/queries.jsonl")[:args.limit]
    attack_base = max(by_id) + 1
    families = {}
    temp_parent = ROOT_DIR / "tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="payload-families-", dir=temp_parent) as temporary:
        temporary_path = Path(temporary)
        for family in FAMILIES:
            records = []
            for index, query in enumerate(queries):
                seed_doc = by_id[query["support_doc_ids"][0]]
                attack_id = attack_base + index
                text = attach_payload(seed_doc["text"], family, query["question"])
                started = time.perf_counter()
                with tempfile.TemporaryDirectory(prefix="case-", dir=temporary_path) as case_dir:
                    case_path = Path(case_dir)
                    staged_path = case_path / "staged.jsonl"
                    stage_document_attack(corpus_path, staged_path, doc_id=attack_id, text=text)
                    retriever = HybridRetriever(
                        staged_path, artifact_dir=case_path / "index",
                        write_integrity_manifest=True,
                    )
                    candidates = retriever.retrieve(query["question"], top_k=5)
                    manifest = load_runtime_manifest(retriever.integrity_path)
                    assert manifest is not None
                    undefended, undefended_source, _ = _select_answer(query["question"], candidates, retriever)
                    kept, filtered = [], []
                    for doc in candidates:
                        integrity = manifest.check(doc["doc_id"], doc["text"])
                        score = detector.score(query["question"], doc["text"])
                        doc["risk_score"] = score["risk_score"]
                        if integrity["status"] != "verified" or score["decision"] == "quarantine":
                            filtered.append(doc["doc_id"])
                        else:
                            kept.append(doc)
                    defended, defended_source, _ = _select_answer(query["question"], kept, retriever)
                    correct = [alias.casefold() for alias in query["answer_aliases"]]
                    records.append({
                        "qid": query["qid"], "family": family,
                        "attack_doc_id": attack_id,
                        "attack_retrieved": attack_id in [doc["doc_id"] for doc in candidates],
                        "attack_quarantined": attack_id in filtered,
                        "undefended_answer": undefended, "undefended_source": undefended_source,
                        "defended_answer": defended, "defended_source": defended_source,
                        "undefended_attack_success": family_success(family, undefended, undefended_source, attack_id),
                        "defended_attack_success": family_success(family, defended, defended_source, attack_id),
                        "clean_answer_recovered": any(alias in defended.casefold() for alias in correct),
                        "abstained": defended_source is None,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    })
            n = len(records)
            latencies = [row["latency_ms"] for row in records]
            families[family] = {
                "cases": n,
                "attack_retrieved": sum(row["attack_retrieved"] for row in records),
                "attack_quarantined": sum(row["attack_quarantined"] for row in records),
                "undefended_attack_successes": sum(row["undefended_attack_success"] for row in records),
                "defended_attack_successes": sum(row["defended_attack_success"] for row in records),
                "clean_answer_recovered": sum(row["clean_answer_recovered"] for row in records),
                "abstentions": sum(row["abstained"] for row in records),
                "median_latency_ms": round(statistics.median(latencies), 2),
            }
            out_cases = RESULTS_DIR / f"payload_families_validation_{family}.cases.jsonl"
            with out_cases.open("w", encoding="utf-8", newline="\n") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{family}: {json.dumps(families[family], indent=2)}", flush=True)
    summary = {
        "benchmark": "msmarco-v1.1-frozen", "split": "validation",
        "surface": "accepted_ingest", "answer_backend": "extractive",
        "retrieval_backend": "sklearn-hashing-384",
        "payload_position": "end", "payload_shape": "query_aware_lead_in",
        "detector_threshold": detector.threshold,
        "families": families,
        "note": "Query-aware templates appended to a copy of the labeled support passage. Success is a family-specific lexical goal on the selected extractive answer, not a model-following judgment.",
    }
    output = RESULTS_DIR / "payload_families_validation.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
