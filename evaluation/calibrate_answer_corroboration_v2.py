"""Calibrate answer-c corroboration threshold on v2 train only.

Reads v2 train queries/corpus, retrieves top-5 with cached MiniLM, selects a
full-context answer, finds its query-group peer, selects an independent peer
answer from that peer document alone, and sweeps token-F1 thresholds.

Never reads v2 validation or v2 test. The output threshold is consumed by
``evaluation/evaluate_answer_corroboration_v2.py``.
"""

from __future__ import annotations

import json
import math
import os

import joblib

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from evaluation.evaluate_paired_gate_v2 import paired_decision
from generation.answer_corroboration import is_abstention, token_f1
from generation.sentence_ranker import SentenceRanker, _contains_alias


def threshold_for_clean_rejection(scores: list[float | None], target_rate: float) -> tuple[float, int]:
    if not scores or not 0 <= target_rate < 1:
        raise ValueError("Need nonempty scores and target_rate in [0, 1)")
    missing = sum(value is None for value in scores)
    allowed = math.floor(target_rate * len(scores))
    values = sorted(value for value in scores if value is not None)
    if not values:
        raise ValueError("No query has a usable corroboration score")
    if missing > allowed:
        return float(values[0]), missing
    threshold = values[allowed - missing]
    rejected = sum(value is None or value < threshold for value in scores)
    return float(threshold), rejected


def main() -> None:
    benchmark = ROOT_DIR / "data/benchmark_v2"
    validate(benchmark, ROOT_DIR / "data/benchmark")
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        benchmark / "train/corpus.jsonl",
        artifact_dir=ROOT_DIR / "artifacts/multisupport_v2/train",
    )
    ranker = joblib.load(ROOT_DIR / "artifacts/sentence_ranker_minilm_strict.joblib")
    if not isinstance(ranker, SentenceRanker) or retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("MiniLM ranker required")
    cases = []
    for row in read_jsonl(benchmark / "train/queries.jsonl"):
        docs = retriever.retrieve(row["question"], top_k=5)
        full = paired_decision(row["question"], docs, retriever, ranker, -1.0)
        peer_answer = None
        if full["peer_doc_id"] is not None:
            by_id = {doc["doc_id"]: doc for doc in retriever.documents}
            peer_doc = by_id.get(full["peer_doc_id"])
            if peer_doc is not None:
                peer_answer, peer_source, _ = ranker.select(row["question"], [peer_doc], retriever)
                if peer_source is None:
                    peer_answer = None
        score = None
        if not is_abstention(full["answer"]) and peer_answer is not None and not is_abstention(peer_answer):
            score = token_f1(full["answer"], peer_answer)
        cases.append({
            "qid": row["qid"],
            "selected_answer": full["answer"],
            "selected_source": full["source_doc_id"],
            "peer_doc_id": full["peer_doc_id"],
            "peer_answer": peer_answer,
            "token_f1": score,
            "alias_match": _contains_alias(full["answer"], row["answer_aliases"]),
        })
    scores = [row["token_f1"] for row in cases]
    threshold, rejected = threshold_for_clean_rejection(scores, 0.05)
    sweep = {}
    for candidate in [round(value / 10, 1) for value in range(0, 11)]:
        kept = sum(score is not None and score >= candidate for score in scores)
        sweep[f"{candidate:.1f}"] = {
            "clean_kept": kept,
            "clean_rejected_or_missing": len(scores) - kept,
        }
    summary = {
        "benchmark": "two-support-msmarco-v2",
        "split": "train",
        "cases": len(cases),
        "target_clean_rejection": 0.05,
        "calibrated_token_f1_threshold": threshold,
        "train_clean_rejected_or_missing": rejected,
        "target_feasible": rejected <= int(0.05 * len(cases)),
        "train_clean_alias_match": sum(row["alias_match"] for row in cases),
        "missing_usable_corroboration_score": sum(row["token_f1"] is None for row in cases),
        "missing_peer_document": sum(row["peer_doc_id"] is None for row in cases),
        "threshold_sweep": sweep,
        "peer_group_contract": "non-null source_query_id equality for retrieved IDs via corpus metadata",
        "note": "Token-F1 agreement between full-context selected answer and independent peer-doc answer. Two passages share an MS MARCO query and are not independent origins.",
    }
    output = ROOT_DIR / "results/answer_corroboration_v2_train_calibration.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
