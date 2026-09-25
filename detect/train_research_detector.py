"""Fit detector on train-only data and calibrate its threshold on validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from detect.research_detector import FEATURE_NAMES, ResearchDetector, threshold_for_fpr
from retrieval.hybrid_retriever import HybridRetriever


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rows_for_queries(queries: list[dict], corpus: list[dict]) -> list[tuple[str, str, str]]:
    docs = {doc["doc_id"]: doc for doc in corpus}
    return [
        (query["qid"], query["question"], docs[query["support_doc_ids"][0]]["text"])
        for query in queries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--train-attacks", type=Path, default=RESULTS_DIR / "attack_baseline_train_random.jsonl")
    parser.add_argument("--additional-train-attacks", nargs="*", type=Path, default=[],
                        help="Other complete train-only attack families for an experimental detector")
    parser.add_argument("--validation-attacks", type=Path, default=RESULTS_DIR / "attack_baseline_validation_greedy.jsonl")
    parser.add_argument("--artifact-output", type=Path, default=ARTIFACTS_DIR / "research_detector_hashing.joblib")
    parser.add_argument("--metrics-output", type=Path, default=RESULTS_DIR / "research_detector_validation.json")
    parser.add_argument("--reference-clean", type=int, default=200)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_benchmark(args.benchmark)
    train_corpus = _read_jsonl(args.benchmark / "train" / "corpus.jsonl")
    train_queries = _read_jsonl(args.benchmark / "train" / "queries.jsonl")
    val_corpus = _read_jsonl(args.benchmark / "validation" / "corpus.jsonl")
    val_queries = _read_jsonl(args.benchmark / "validation" / "queries.jsonl")
    train_attack_paths = [args.train_attacks, *args.additional_train_attacks]
    if len({path.resolve() for path in train_attack_paths}) != len(train_attack_paths):
        raise ValueError("Training attack paths must be distinct")
    train_attack_families = [
        {row["qid"]: row for row in _read_jsonl(path)} for path in train_attack_paths
    ]
    val_attacks = {row["qid"]: row for row in _read_jsonl(args.validation_attacks)}
    if any(set(family) != {row["qid"] for row in train_queries} for family in train_attack_families):
        raise ValueError("Each training attack family must cover exactly the training questions")
    if set(val_attacks) != {row["qid"] for row in val_queries}:
        raise ValueError("Validation attacks must cover exactly the validation questions")
    if not 10 <= args.reference_clean < len(train_queries) - 10:
        raise ValueError("--reference-clean leaves too few classifier training examples")
    if any(row["split"] != "train" for family in train_attack_families for row in family.values()):
        raise ValueError("Training attacks contain a non-training split")
    if any(row["split"] != "validation" for row in val_attacks.values()):
        raise ValueError("Validation attacks contain a non-validation split")

    retriever = HybridRetriever(args.benchmark / "train" / "corpus.jsonl")
    clean_train = _rows_for_queries(train_queries, train_corpus)
    random.Random(args.seed).shuffle(clean_train)
    reference = clean_train[: args.reference_clean]
    classifier_clean = clean_train[args.reference_clean :]
    reference_ids = {qid for qid, _, _ in reference}
    classifier_ids = {qid for qid, _, _ in classifier_clean}
    if reference_ids & classifier_ids:
        raise AssertionError("Reference and classifier queries overlap")

    reference_embeddings = retriever.encode([text for _, _, text in reference])
    covariance = LedoitWolf().fit(reference_embeddings)
    iforest = IsolationForest(n_estimators=200, random_state=args.seed, contamination="auto").fit(reference_embeddings)

    detector = ResearchDetector(
        retriever, covariance.location_, covariance.precision_, iforest,
        classifier=None, threshold=0.5, embedding_model=retriever.embedder.model_name,
    )
    train_rows = [(qid, query, text, 0) for qid, query, text in classifier_clean]
    for family in train_attack_families:
        train_rows += [
            (qid, family[qid]["question"], family[qid]["text"], 1)
            for qid in sorted(classifier_ids)
        ]
    train_features = np.vstack([detector.features(query, text) for _, query, text, _ in train_rows])
    train_labels = np.asarray([label for _, _, _, label in train_rows])
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, random_state=args.seed)
    )
    classifier.fit(train_features, train_labels)
    detector.classifier = classifier

    clean_validation = _rows_for_queries(val_queries, val_corpus)
    val_rows = [(qid, query, text, 0) for qid, query, text in clean_validation]
    val_rows += [
        (row["qid"], row["question"], row["text"], 1)
        for row in _read_jsonl(args.validation_attacks)
    ]
    val_features = np.vstack([detector.features(query, text) for _, query, text, _ in val_rows])
    val_labels = np.asarray([label for _, _, _, label in val_rows])
    val_scores = classifier.predict_proba(val_features)[:, 1]
    detector.threshold = threshold_for_fpr(val_scores[val_labels == 0], args.target_fpr)
    predictions = (val_scores >= detector.threshold).astype(int)
    clean_count = int(sum(val_labels == 0))
    attack_count = int(sum(val_labels == 1))
    metrics = {
        "evaluation": "validation only; test split untouched",
        "embedding_backend": detector.embedding_model,
        "reference_clean": len(reference),
        "classifier_clean": len(classifier_clean),
        "classifier_attack": len(classifier_ids) * len(train_attack_families),
        "train_attack_files": [str(path) for path in train_attack_paths],
        "validation_attack_file": str(args.validation_attacks),
        "validation_clean": clean_count,
        "validation_attack": attack_count,
        "target_fpr": args.target_fpr,
        "threshold": detector.threshold,
        "observed_validation_fpr": float(sum((predictions == 1) & (val_labels == 0)) / clean_count),
        "validation_recall": float(recall_score(val_labels, predictions)),
        "validation_precision": float(precision_score(val_labels, predictions, zero_division=0)),
        "validation_f1": float(f1_score(val_labels, predictions, zero_division=0)),
        "validation_roc_auc": float(roc_auc_score(val_labels, val_scores)),
        "validation_pr_auc": float(average_precision_score(val_labels, val_scores)),
        "feature_names": list(FEATURE_NAMES),
        "note": "Validation-only local attack families; evaluate unseen attack families before claiming generalization.",
    }
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, args.artifact_output)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with args.metrics_output.with_name(args.metrics_output.stem + "_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for (qid, _, _, label), score, prediction in zip(val_rows, val_scores, predictions):
            handle.write(json.dumps({"qid": qid, "label": label, "risk_score": float(score),
                                     "prediction": int(prediction)}) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
