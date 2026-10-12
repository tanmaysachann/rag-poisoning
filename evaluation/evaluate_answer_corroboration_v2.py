"""Evaluate disagreement-abstain corroboration on v2 validation only.

Clean path uses the frozen v2 validation corpus. Attacked path replaces the
first labeled support passage with a single wrong-answer substitution drawn
from the v2 train wrong-answer pool (same construction as the paired-gate
experiment). The gate abstains on peer disagreement; it never fails over to
the peer, so it cannot create new successes by choosing a poisoned peer.

Never reads v2 test. Threshold comes from v2 train calibration only.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib

from attack.baselines import choose_wrong_answer, substitute_answer
from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from evaluation.evaluate_paired_gate_v2 import paired_decision
from generation.answer_corroboration import corroboration_decision, is_abstention, token_f1
from generation.sentence_ranker import SentenceRanker, _contains_alias


def gate_for_context(question: str, retrieved: list[dict], retriever, ranker, threshold: float) -> dict:
    full = paired_decision(question, retrieved, retriever, ranker, -1.0)
    peer_answer = None
    if full["peer_doc_id"] is not None:
        by_id = {doc["doc_id"]: doc for doc in retriever.documents}
        peer_doc = by_id.get(full["peer_doc_id"])
        if peer_doc is not None:
            candidate, source, _ = ranker.select(question, [peer_doc], retriever)
            peer_answer = candidate if source is not None else None
    score = None
    if not is_abstention(full["answer"]) and peer_answer is not None and not is_abstention(peer_answer):
        score = token_f1(full["answer"], peer_answer)
        gate = corroboration_decision(
            full["answer"], full["source_doc_id"], peer_answer, full["peer_doc_id"], threshold
        )
    elif full["source_doc_id"] is None or is_abstention(full["answer"]):
        gate = {"gate_keeps_answer": True, "reason": "selected_abstained_no_answer", "token_f1": None}
    else:
        gate = {"gate_keeps_answer": False, "reason": "missing_peer_or_peer_abstained", "token_f1": None}
    # Leave-one-out influence of the selected source within the retrieved set.
    loo_changed = None
    if full["source_doc_id"] is not None:
        remaining = [doc for doc in retrieved if doc["doc_id"] != full["source_doc_id"]]
        without, _, _ = ranker.select(question, remaining, retriever)
        loo_changed = without != full["answer"]
    return {
        "answer": full["answer"],
        "source_doc_id": full["source_doc_id"],
        "peer_doc_id": full["peer_doc_id"],
        "peer_answer": peer_answer,
        "token_f1": score,
        "gate_keeps_answer": gate["gate_keeps_answer"],
        "gate_reason": gate["reason"],
        "loo_answer_changed": loo_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    benchmark = ROOT_DIR / "data/benchmark_v2"
    validate(benchmark, ROOT_DIR / "data/benchmark")
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from retrieval.hybrid_retriever import HybridRetriever

    calibration = json.loads(
        (ROOT_DIR / "results/answer_corroboration_v2_train_calibration.json").read_text(encoding="utf-8")
    )
    threshold = float(calibration["calibrated_token_f1_threshold"])
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
    with tempfile.TemporaryDirectory(prefix="corroboration-v2-", dir=temp_parent) as temporary:
        for index, query in enumerate(queries):
            retrieved = clean.retrieve(query["question"], top_k=args.top_k)
            clean_result = gate_for_context(query["question"], retrieved, clean, ranker, threshold)
            attack_id = query["support_doc_ids"][0]
            wrong = choose_wrong_answer(query["answer_aliases"], wrong_pool, 59 + index)
            poisoned_text = substitute_answer(docs[attack_id]["text"], query["answer_aliases"], wrong)
            case_path = Path(temporary) / f"case-{index}"
            case_path.mkdir()
            staged = case_path / "staged.jsonl"
            staged_docs = [
                {**doc, "text": poisoned_text} if doc["doc_id"] == attack_id else doc
                for doc in clean.documents
            ]
            with staged.open("w", encoding="utf-8", newline="\n") as handle:
                for doc in staged_docs:
                    handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            attacked = HybridRetriever(staged, artifact_dir=case_path / "index")
            attack_retrieved = attacked.retrieve(query["question"], top_k=args.top_k)
            attack_result = gate_for_context(query["question"], attack_retrieved, attacked, ranker, threshold)
            before_success = (
                attack_result["source_doc_id"] == attack_id
                and _contains_alias(attack_result["answer"], [wrong])
            )
            cases.append({
                "qid": query["qid"],
                "question": query["question"],
                "support_doc_ids": query["support_doc_ids"],
                "attack_doc_id": attack_id,
                "wrong_answer": wrong,
                "clean": clean_result,
                "attacked": attack_result,
                "clean_alias_before_gate": _contains_alias(clean_result["answer"], query["answer_aliases"]),
                "clean_alias_after_gate": (
                    clean_result["gate_keeps_answer"]
                    and _contains_alias(clean_result["answer"], query["answer_aliases"])
                ),
                "attack_success_before_gate": before_success,
                "attack_success_after_gate": before_success and attack_result["gate_keeps_answer"],
                "attack_retrieved": attack_id in [doc["doc_id"] for doc in attack_retrieved],
            })
            if (index + 1) % 10 == 0:
                print(f"validation: {index + 1}/{len(queries)}", flush=True)
    summary = {
        "benchmark": "two-support-msmarco-v2",
        "split": "validation",
        "cases": len(cases),
        "train_calibrated_token_f1_threshold": threshold,
        "clean_answer_alias_before_gate": sum(row["clean_alias_before_gate"] for row in cases),
        "clean_answer_alias_after_gate": sum(row["clean_alias_after_gate"] for row in cases),
        "clean_answer_kept": sum(row["clean"]["gate_keeps_answer"] for row in cases),
        "attack_retrieved": sum(row["attack_retrieved"] for row in cases),
        "attack_success_before_gate": sum(row["attack_success_before_gate"] for row in cases),
        "attack_success_after_gate": sum(row["attack_success_after_gate"] for row in cases),
        "attacked_answer_kept": sum(row["attacked"]["gate_keeps_answer"] for row in cases),
        "gate_policy": "abstain_on_disagreement_or_missing_peer_no_failover",
        "peer_group_contract": "non-null source_query_id equality for retrieved IDs via corpus metadata",
        "note": "One of two selected passages replaced; two passages may share source origin. Abstention removes a wrong answer without selecting the peer.",
    }
    output = ROOT_DIR / "results/answer_corroboration_v2_validation.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
