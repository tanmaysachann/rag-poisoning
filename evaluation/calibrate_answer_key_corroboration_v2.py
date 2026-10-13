"""Calibrate the answer-key containment gate on v2 train only.

Compares strict vs content-only key modes on full-context selected answers
against the query-group peer passage. Selects the mode meeting the 5%
clean-rejection budget with the most retained clean alias matches. Never
reads v2 validation or v2 test.
"""

from __future__ import annotations

import json
import os

import joblib

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from evaluation.evaluate_paired_gate_v2 import paired_decision
from generation.answer_key_corroboration import containment_decision, is_abstention
from generation.sentence_ranker import SentenceRanker, _contains_alias


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
    by_id = {doc["doc_id"]: doc for doc in retriever.documents}
    rows = []
    for row in read_jsonl(benchmark / "train/queries.jsonl"):
        docs = retriever.retrieve(row["question"], top_k=5)
        full = paired_decision(row["question"], docs, retriever, ranker, -1.0)
        peer_text = None
        if full["peer_doc_id"] is not None and full["peer_doc_id"] in by_id:
            peer_text = by_id[full["peer_doc_id"]]["text"]
        rows.append({
            "qid": row["qid"], "answer": full["answer"], "source_doc_id": full["source_doc_id"],
            "peer_doc_id": full["peer_doc_id"], "peer_text": peer_text,
            "alias_match": _contains_alias(full["answer"], row["answer_aliases"]),
        })
    modes = {}
    for content_only in (False, True):
        name = "content_only" if content_only else "strict"
        decisions = [containment_decision(
            row["answer"], row["source_doc_id"], row["peer_text"], row["peer_doc_id"],
            content_only=content_only) for row in rows]
        kept = sum(decision["gate_keeps_answer"] for decision in decisions)
        modes[name] = {
            "clean_kept": kept,
            "clean_rejected_or_missing": len(rows) - kept,
            "clean_alias_retained": sum(
                row["alias_match"] and decision["gate_keeps_answer"]
                for row, decision in zip(rows, decisions)),
        }
    feasible = {name: mode for name, mode in modes.items()
                if mode["clean_rejected_or_missing"] <= int(0.05 * len(rows))}
    if feasible:
        # Most retained clean alias matches; tie-break toward content_only,
        # which ignores function-word mismatches between passages.
        selected = sorted(feasible, key=lambda name: (
            -feasible[name]["clean_alias_retained"], name != "content_only"))[0]
    else:
        selected = sorted(modes, key=lambda name: modes[name]["clean_rejected_or_missing"])[0]
    summary = {
        "benchmark": "two-support-msmarco-v2", "split": "train", "cases": len(rows),
        "target_clean_rejection": 0.05, "target_budget": int(0.05 * len(rows)),
        "modes": modes, "selected_mode": selected,
        "selected_content_only": selected == "content_only",
        "target_feasible": modes[selected]["clean_rejected_or_missing"] <= int(0.05 * len(rows)),
        "train_clean_alias_match": sum(row["alias_match"] for row in rows),
        "missing_peer_document": sum(row["peer_doc_id"] is None for row in rows),
        "peer_group_contract": "non-null source_query_id equality for retrieved IDs via corpus metadata",
        "note": "Answer-key containment in the full peer passage text, not sentence similarity. Two passages share an MS MARCO query and are not independent origins.",
    }
    output = ROOT_DIR / "results/answer_key_corroboration_v2_train_calibration.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    slim = [{key: row[key] for key in ("qid", "answer", "source_doc_id", "peer_doc_id", "alias_match")} for row in rows]
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in slim:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
