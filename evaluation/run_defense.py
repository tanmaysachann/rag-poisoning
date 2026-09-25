"""Paired end-to-end defense experiment for frozen validation attack cases."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack.harness import stage_document_attack
from config import ARTIFACTS_DIR, RESULTS_DIR
from data.validate_benchmark import validate_benchmark
from detect.research_detector import ResearchDetector
from generation.sentence_ranker import SentenceRanker
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import HybridRetriever
from security.integrity import load_runtime_manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--strategy", choices=("random", "greedy", "stealth"), required=True)
    parser.add_argument(
        "--surface", choices=("accepted_ingest", "post_index_tamper"),
        default="accepted_ingest",
        help="Whether the poisoned content was sealed at ingest or edited after clean indexing",
    )
    parser.add_argument("--detector", type=Path, default=ARTIFACTS_DIR / "research_detector_hashing.joblib")
    parser.add_argument("--minilm-ranker", action="store_true", help="Evaluate transfer to cached MiniLM and the learned answer selector")
    parser.add_argument("--attacks", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_benchmark(args.benchmark)
    detector = joblib.load(args.detector)
    if not isinstance(detector, ResearchDetector):
        raise ValueError("Research detector artifact has the wrong type")
    if detector.embedding_model != "sklearn-hashing-384":
        raise ValueError("This research experiment currently requires the hashing detector")
    ranker = None
    if args.minilm_ranker:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        ranker = joblib.load(ARTIFACTS_DIR / "sentence_ranker_minilm_strict.joblib")
        if not isinstance(ranker, SentenceRanker):
            raise ValueError("Wrong sentence-ranker artifact type")
    attack_path = args.attacks or RESULTS_DIR / f"attack_baseline_{args.split}_{args.strategy}.jsonl"
    attacks = _read_jsonl(attack_path)
    if any(row["split"] != args.split or row["strategy"] != args.strategy for row in attacks):
        raise ValueError("Attack records do not match the requested split and strategy")
    corpus_path = args.benchmark / args.split / "corpus.jsonl"
    os.environ["RAG_USE_MINILM"] = "0"
    clean_retriever = HybridRetriever(corpus_path)
    clean_manifest_path = clean_retriever.integrity_path
    if not clean_manifest_path.is_file():
        raise FileNotFoundError("Clean benchmark integrity manifest is missing; run scripts/index_benchmark.py")
    queries = {row["qid"]: row for row in _read_jsonl(args.benchmark / args.split / "queries.jsonl")}
    if {row["qid"] for row in attacks} != set(queries):
        raise ValueError("Attack records must cover exactly the selected split")

    output = args.output or RESULTS_DIR / f"defense_{args.split}_{args.strategy}_{args.surface}{'_minilm_ranker' if ranker else ''}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    os.environ["RAG_USE_MINILM"] = "1" if ranker else "0"
    temp_parent = ROOT / "tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sentinel-defense-", dir=temp_parent) as temporary:
        for attack in attacks:
            query = queries[attack["qid"]]
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="case-", dir=temporary) as case_dir:
                case_path = Path(case_dir)
                staged_path = case_path / "staged.jsonl"
                stage_document_attack(
                    corpus_path, staged_path,
                    doc_id=attack["attack_doc_id"], text=attack["text"],
                    replace_existing=args.strategy == "stealth",
                )
                retriever = HybridRetriever(
                    staged_path,
                    artifact_dir=case_path / "index",
                    integrity_path=clean_manifest_path if args.surface == "post_index_tamper" else None,
                    write_integrity_manifest=args.surface == "accepted_ingest",
                )
                if ranker and retriever.embedder.backend != "sentence-transformers":
                    raise RuntimeError("MiniLM could not be loaded for transfer evaluation")
                candidates = retriever.retrieve(query["question"], top_k=5)
                manifest = load_runtime_manifest(retriever.integrity_path)
                assert manifest is not None
                select_answer = ranker.select if ranker else _select_answer
                undefended, undefended_source, _ = select_answer(query["question"], candidates, retriever)
                kept: list[dict] = []
                filtered: list[dict] = []
                for doc in candidates:
                    integrity = manifest.check(doc["doc_id"], doc["text"])
                    score = detector.score(query["question"], doc["text"])
                    doc["risk_score"] = score["risk_score"]
                    doc["integrity_status"] = integrity["status"]
                    if integrity["status"] != "verified" or score["decision"] == "quarantine":
                        filtered.append(doc)
                    else:
                        kept.append(doc)
                defended, defended_source, _ = select_answer(query["question"], kept, retriever)
                # LOO is diagnostic here: influence alone cannot establish that
                # a document is malicious, especially with single-source QA.
                loo_started = time.perf_counter()
                answer_source = next((doc for doc in kept if doc["doc_id"] == defended_source), None)
                loo_triggered = bool(
                    answer_source is not None
                    and 0.25 <= answer_source["risk_score"] < detector.threshold
                )
                loo_answer = None
                loo_changed_answer = None
                if loo_triggered:
                    loo_answer, _, _ = select_answer(
                        query["question"],
                        [doc for doc in kept if doc["doc_id"] != defended_source],
                        retriever,
                    )
                    loo_changed_answer = loo_answer.casefold() != defended.casefold()
                loo_ms = (time.perf_counter() - loo_started) * 1000 if loo_triggered else 0.0
                wrong = attack["wrong_answer"].casefold()
                correct = [alias.casefold() for alias in query["answer_aliases"]]
                records.append(
                    {
                        "qid": attack["qid"],
                        "attack_doc_id": attack["attack_doc_id"],
                        "attack_top5_rank": next((i for i, doc in enumerate(candidates, 1) if doc["doc_id"] == attack["attack_doc_id"]), None),
                        "original_hashing_attack_rank": attack["top5_rank"],
                        "undefended_answer": undefended,
                        "undefended_source": undefended_source,
                        "defended_answer": defended,
                        "defended_source": defended_source,
                        "undefended_attack_success": undefended_source == attack["attack_doc_id"] and wrong in undefended.casefold(),
                        "defended_attack_success": defended_source == attack["attack_doc_id"] and wrong in defended.casefold(),
                        "clean_answer_recovered": any(alias in defended.casefold() for alias in correct),
                        "abstained": defended_source is None,
                        "attack_quarantined": attack["attack_doc_id"] in {doc["doc_id"] for doc in filtered},
                        "filtered_doc_ids": [doc["doc_id"] for doc in filtered],
                        "attack_integrity_status": next(
                            (doc["integrity_status"] for doc in candidates if doc["doc_id"] == attack["attack_doc_id"]),
                            "not_retrieved",
                        ),
                        "loo_triggered": loo_triggered,
                        "loo_changed_answer": loo_changed_answer,
                        "loo_answer": loo_answer,
                        "loo_extra_extractor_calls": int(loo_triggered),
                        "loo_ms": loo_ms,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                )
    n = len(records)
    successful_before = [row for row in records if row["undefended_attack_success"]]
    latencies = [row["latency_ms"] for row in records]
    summary = {
        "split": args.split,
        "strategy": args.strategy,
        "surface": args.surface,
        "answer_backend": "learned_sentence_ranker" if ranker else "extractive",
        "retrieval_backend": "sentence-transformers/all-MiniLM-L6-v2" if ranker else "sklearn-hashing-384",
        "detector_threshold": detector.threshold,
        "cases": n,
        "undefended_attack_success_rate": sum(row["undefended_attack_success"] for row in records) / n,
        "defended_attack_success_rate": sum(row["defended_attack_success"] for row in records) / n,
        "attack_quarantine_rate": sum(row["attack_quarantined"] for row in records) / n,
        "clean_answer_recovery_rate": sum(row["clean_answer_recovered"] for row in records) / n,
        "abstention_rate": sum(row["abstained"] for row in records) / n,
        "selective_loo_rate": sum(row["loo_triggered"] for row in records) / n,
        "selective_loo_changed_answer_rate": sum(row["loo_changed_answer"] is True for row in records) / n,
        "mean_loo_ms": sum(row["loo_ms"] for row in records) / n,
        "defense_success_given_prior_attack_success": (
            sum(not row["defended_attack_success"] for row in successful_before) / len(successful_before)
            if successful_before else None
        ),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
