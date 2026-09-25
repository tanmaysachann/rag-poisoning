"""Train/calibrate paired-passage risk on v2 train; inspect v2 validation.

The evaluator receives the labeled pair for each question. It is a diagnostic
upper bound for a retrieval system that can identify the peer passage; it is
not a production provenance check or a live quarantine rule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from attack.baselines import choose_wrong_answer, substitute_answer
from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from detect.pairwise_consistency import FEATURE_NAMES, pair_features
from detect.research_detector import threshold_for_fpr


def examples(benchmark: Path, split: str, embedder, wrong_pool: list[str]) -> list[dict]:
    docs = {row["doc_id"]: row for row in read_jsonl(benchmark / split / "corpus.jsonl")}
    rows = []
    for index, query in enumerate(read_jsonl(benchmark / split / "queries.jsonl")):
        left, right = (docs[doc_id]["text"] for doc_id in query["support_doc_ids"])
        wrong = choose_wrong_answer(query["answer_aliases"], wrong_pool, 59 + index)
        attacked = substitute_answer(left, query["answer_aliases"], wrong)
        rows.append({"qid": query["qid"], "wrong_answer": wrong,
                     "clean_features": pair_features(query["question"], left, right, embedder),
                     "attack_features": pair_features(query["question"], attacked, right, embedder)})
    return rows


def main() -> None:
    benchmark = ROOT_DIR / "data/benchmark_v2"
    validate(benchmark, ROOT_DIR / "data/benchmark")
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from retrieval.hybrid_retriever import HybridRetriever

    train_retriever = HybridRetriever(benchmark / "train/corpus.jsonl",
                                      artifact_dir=ROOT_DIR / "artifacts/multisupport_v2/train")
    if train_retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("Cached MiniLM required")
    train_queries = read_jsonl(benchmark / "train/queries.jsonl")
    wrong_pool = [row["answer_aliases"][0] for row in train_queries]
    train = examples(benchmark, "train", train_retriever, wrong_pool)
    fit_rows, calibration = train[:100], train[100:]
    x = np.asarray([row["clean_features"] for row in fit_rows]
                   + [row["attack_features"] for row in fit_rows])
    y = np.asarray([0] * len(fit_rows) + [1] * len(fit_rows))
    classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=59, max_iter=1000))
    classifier.fit(x, y)
    clean_cal_scores = classifier.predict_proba(np.asarray([row["clean_features"] for row in calibration]))[:, 1]
    threshold = threshold_for_fpr(clean_cal_scores, 0.05)
    validation_retriever = HybridRetriever(benchmark / "validation/corpus.jsonl",
                                            artifact_dir=ROOT_DIR / "artifacts/multisupport_v2/validation")
    validation = examples(benchmark, "validation", validation_retriever, wrong_pool)
    gate_cases = {row["qid"]: row for row in read_jsonl(
        ROOT_DIR / "results/paired_gate_v2_validation_full_context.cases.jsonl"
    )}
    scores = []
    for row in validation:
        clean_score = float(classifier.predict_proba(row["clean_features"].reshape(1, -1))[0, 1])
        attack_score = float(classifier.predict_proba(row["attack_features"].reshape(1, -1))[0, 1])
        existing = gate_cases[row["qid"]]
        if row["wrong_answer"] != existing["wrong_answer"]:
            raise ValueError("Wrong-answer schedule differs from paired-gate validation")
        scores.append({"qid": row["qid"], "clean_risk": clean_score, "attack_risk": attack_score,
                       "clean_flagged": clean_score >= threshold,
                       "attack_flagged": attack_score >= threshold,
                       "baseline_attack_success": existing["attack_success_before_gate"],
                       "retrieved_peer": existing["attacked"]["peer_doc_id"] is not None})
    clean = np.asarray([row["clean_risk"] for row in scores])
    attacks = np.asarray([row["attack_risk"] for row in scores])
    summary = {
        "benchmark": "two-support-msmarco-v2", "split": "validation", "cases": len(scores),
        "feature_names": FEATURE_NAMES,
        "fit_train_pairs": len(fit_rows), "calibration_train_pairs": len(calibration),
        "train_calibrated_threshold": threshold,
        "calibration_clean_flags": int(np.sum(clean_cal_scores >= threshold)),
        "validation_clean_flags": sum(row["clean_flagged"] for row in scores),
        "validation_attack_flags": sum(row["attack_flagged"] for row in scores),
        "validation_auc": float(roc_auc_score(
            np.r_[np.zeros(len(clean)), np.ones(len(attacks))], np.r_[clean, attacks]
        )),
        "baseline_attack_successes": sum(row["baseline_attack_success"] for row in scores),
        "baseline_successes_with_flagged_pair_and_retrieved_peer": sum(
            row["baseline_attack_success"] and row["attack_flagged"] and row["retrieved_peer"]
            for row in scores
        ),
        "note": "Oracle labeled pairs and same-family substitution; flags cannot identify which passage is false.",
    }
    output = ROOT_DIR / "results/pairwise_detector_v2_validation.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in scores:
            handle.write(json.dumps(row) + "\n")
    joblib.dump({"classifier": classifier, "threshold": threshold, "feature_names": FEATURE_NAMES},
                ROOT_DIR / "artifacts/pairwise_detector_v2.joblib")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
