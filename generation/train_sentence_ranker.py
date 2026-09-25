"""Fit sentence-level answer selection on training questions only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from detect.research_detector import threshold_for_fpr
from generation.sentence_ranker import FEATURE_NAMES, SentenceRanker, _contains_alias, sentence_candidates
from retrieval.hybrid_retriever import HybridRetriever


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--minilm", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Calibrate abstention after removing the known support document")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_benchmark(args.benchmark)
    os.environ["RAG_USE_MINILM"] = "1" if args.minilm else "0"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    corpus = args.benchmark / "train" / "corpus.jsonl"
    retriever = HybridRetriever(corpus, artifact_dir=ARTIFACTS_DIR / "minilm_benchmark" / "train" if args.minilm else None)
    if args.minilm and retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("MiniLM was requested but could not be loaded")
    queries = _read_jsonl(args.benchmark / "train" / "queries.jsonl")
    train_x, train_y, calibration = [], [], []
    for index, query in enumerate(queries):
        docs = retriever.retrieve(query["question"], top_k=5)
        features, metadata = sentence_candidates(query["question"], docs, retriever)
        labels = np.asarray([_contains_alias(row["sentence"], query["answer_aliases"]) for row in metadata], dtype=int)
        if index < 280:
            train_x.append(features)
            train_y.append(labels)
        else:
            calibration.append((features, labels, metadata, set(query["support_doc_ids"])))
    x = np.vstack(train_x)
    y = np.concatenate(train_y)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Training data requires positive and negative answer sentences")
    classifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    classifier.fit(x, y)
    if args.strict:
        unsupported_scores = []
        for features, _, metadata, support_ids in calibration:
            unsupported = [index for index, row in enumerate(metadata) if row["doc_id"] not in support_ids]
            if unsupported:
                unsupported_scores.append(float(classifier.predict_proba(features[unsupported])[:, 1].max()))
        selected_threshold = threshold_for_fpr(np.asarray(unsupported_scores), 0.05)
    else:
        selected_threshold = None
    best = None
    for threshold in ([selected_threshold] if args.strict else np.linspace(0.05, 0.95, 19)):
        hits = abstentions = 0
        for features, labels, _, _ in calibration:
            if not len(labels):
                abstentions += 1
                continue
            scores = classifier.predict_proba(features)[:, 1]
            winner = int(np.argmax(scores))
            if scores[winner] < threshold:
                abstentions += 1
            else:
                hits += int(labels[winner])
        key = (hits, -abstentions, -threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), hits, abstentions)
    assert best is not None
    ranker = SentenceRanker(classifier, best[1], retriever.embedder.model_name)
    suffix = f"{'minilm' if args.minilm else 'hashing'}{'_strict' if args.strict else ''}"
    output = args.output or ARTIFACTS_DIR / f"sentence_ranker_{suffix}.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ranker, output)
    report = {
        "embedding_model": ranker.embedding_model,
        "feature_names": FEATURE_NAMES,
        "train_queries": 280, "calibration_queries": len(calibration),
        "positive_sentences": int(y.sum()), "negative_sentences": int(len(y) - y.sum()),
        "threshold": ranker.threshold,
        "calibration": "support_removed_5_percent_admission" if args.strict else "maximum_alias_hits",
        "calibration_alias_hits": best[2], "calibration_abstentions": best[3],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"sentence_ranker_train_{suffix}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
