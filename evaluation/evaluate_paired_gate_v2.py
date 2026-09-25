"""Evaluate a paired-passage consistency gate on v2 validation only.

The attacker replaces one answer-bearing passage before ingest. The second
passage is left clean. Two passages from one MS MARCO row are not proven
independent sources; this is a bounded robustness experiment.
"""

from __future__ import annotations

import json
import os
import tempfile
import argparse
from pathlib import Path

import joblib

from attack.baselines import choose_wrong_answer, substitute_answer
from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from generation.sentence_ranker import SentenceRanker, _contains_alias


def paired_decision(query: str, retrieved: list[dict], retriever, ranker: SentenceRanker,
                    threshold: float) -> dict:
    answer, source, _ = ranker.select(query, retrieved, retriever)
    by_id = {doc["doc_id"]: doc for doc in retrieved}
    selected = by_id.get(source)
    peer = next((doc for doc in retrieved if selected is not None
                 and doc["doc_id"] != source
                 and doc.get("source_query_id") == selected.get("source_query_id")), None)
    peer_answer = None
    cosine = None
    if peer is not None:
        peer_answer, peer_source, _ = ranker.select(query, [peer], retriever)
        if peer_source is not None:
            cosine = float(retriever.encode(answer)[0] @ retriever.encode(peer_answer)[0])
    keep = cosine is not None and cosine >= threshold
    return {"answer": answer, "source_doc_id": source,
            "peer_doc_id": peer["doc_id"] if peer else None,
            "peer_answer": peer_answer, "answer_cosine": cosine,
            "gate_keeps_answer": keep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", choices=("doc_only", "full_context"), default="full_context")
    args = parser.parse_args()
    benchmark = ROOT_DIR / "data/benchmark_v2"
    validate(benchmark, ROOT_DIR / "data/benchmark")
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from retrieval.hybrid_retriever import HybridRetriever

    if args.calibration == "doc_only":
        threshold = json.loads((ROOT_DIR / "results/paired_consistency_v2_train.json").read_text(encoding="utf-8"))["cosine_p05_when_both_answered"]
    else:
        threshold = json.loads((ROOT_DIR / "results/paired_gate_v2_train_calibration.json").read_text(encoding="utf-8"))["calibrated_threshold"]
    ranker = joblib.load(ROOT_DIR / "artifacts/sentence_ranker_minilm_strict.joblib")
    if not isinstance(ranker, SentenceRanker):
        raise ValueError("Wrong ranker artifact")
    corpus_path = benchmark / "validation/corpus.jsonl"
    clean = HybridRetriever(corpus_path, artifact_dir=ROOT_DIR / "artifacts/multisupport_v2/validation")
    if clean.embedder.backend != "sentence-transformers":
        raise RuntimeError("Cached MiniLM is required")
    docs = {doc["doc_id"]: doc for doc in clean.documents}
    train_queries = read_jsonl(benchmark / "train/queries.jsonl")
    wrong_pool = [row["answer_aliases"][0] for row in train_queries]
    queries = read_jsonl(benchmark / "validation/queries.jsonl")
    cases = []
    temp_parent = ROOT_DIR / "tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paired-gate-v2-", dir=temp_parent) as temporary:
        for index, query in enumerate(queries):
            retrieved = clean.retrieve(query["question"], top_k=5)
            clean_result = paired_decision(query["question"], retrieved, clean, ranker, threshold)
            attack_id = query["support_doc_ids"][0]
            wrong = choose_wrong_answer(query["answer_aliases"], wrong_pool, 59 + index)
            poisoned_text = substitute_answer(docs[attack_id]["text"], query["answer_aliases"], wrong)
            case_path = Path(temporary) / f"case-{index}"
            case_path.mkdir()
            staged = case_path / "staged.jsonl"
            staged_docs = [{**doc, "text": poisoned_text} if doc["doc_id"] == attack_id else doc
                           for doc in clean.documents]
            with staged.open("w", encoding="utf-8", newline="\n") as handle:
                for doc in staged_docs:
                    handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            attacked = HybridRetriever(staged, artifact_dir=case_path / "index")
            attack_retrieved = attacked.retrieve(query["question"], top_k=5)
            attack_result = paired_decision(query["question"], attack_retrieved,
                                            attacked, ranker, threshold)
            before_success = (attack_result["source_doc_id"] == attack_id
                              and _contains_alias(attack_result["answer"], [wrong]))
            cases.append({
                "qid": query["qid"], "question": query["question"],
                "support_doc_ids": query["support_doc_ids"], "attack_doc_id": attack_id,
                "wrong_answer": wrong,
                "clean": clean_result, "attacked": attack_result,
                "clean_alias_before_gate": _contains_alias(clean_result["answer"], query["answer_aliases"]),
                "clean_alias_after_gate": (clean_result["gate_keeps_answer"]
                                           and _contains_alias(clean_result["answer"], query["answer_aliases"])),
                "attack_success_before_gate": before_success,
                "attack_success_after_gate": before_success and attack_result["gate_keeps_answer"],
                "attack_retrieved": attack_id in [doc["doc_id"] for doc in attack_retrieved],
            })
            if (index + 1) % 10 == 0:
                print(f"validation: {index + 1}/{len(queries)}", flush=True)
    n = len(cases)
    summary = {
        "benchmark": "two-support-msmarco-v2", "split": "validation", "cases": n,
        "calibration": args.calibration, "train_calibrated_cosine_threshold": threshold,
        "clean_answer_alias_before_gate": sum(row["clean_alias_before_gate"] for row in cases),
        "clean_answer_alias_after_gate": sum(row["clean_alias_after_gate"] for row in cases),
        "clean_answer_kept": sum(row["clean"]["gate_keeps_answer"] for row in cases),
        "attack_retrieved": sum(row["attack_retrieved"] for row in cases),
        "attack_success_before_gate": sum(row["attack_success_before_gate"] for row in cases),
        "attack_success_after_gate": sum(row["attack_success_after_gate"] for row in cases),
        "attacked_answer_kept": sum(row["attacked"]["gate_keeps_answer"] for row in cases),
        "note": "One of two selected passages replaced; similarity is not entailment, and the two passages may share source origin.",
    }
    output = ROOT_DIR / f"results/paired_gate_v2_validation_{args.calibration}.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
