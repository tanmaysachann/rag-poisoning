"""Train and evaluate the explainable phase-1 fusion classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (ARTIFACTS_DIR, BASE_CORPUS_PATH, DEMO_CORPUS_PATH,
                    POISONED_DOCS_PATH, RESULTS_DIR, ensure_project_dirs)
from detect.signals import behavioural_signal_components, isolation_forest_score, mahalanobis_score
from retrieval.hybrid_retriever import HybridRetriever

FEATURE_NAMES = ["mahalanobis", "isolation_forest", "relevance_spike",
                 "instruction_pattern", "url_pattern", "authority_cue",
                 "dense_rank_normalized"]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dense_rank(retriever: HybridRetriever, query: str, doc_id: int) -> float:
    order, _ = retriever.dense_search(query)
    index_by_id = {doc["doc_id"]: index for index, doc in enumerate(retriever.documents)}
    target_index = index_by_id[doc_id]
    rank = int(np.where(order == target_index)[0][0]) + 1
    return float(1.0 - (rank - 1) / max(len(order) - 1, 1))


def main() -> None:
    ensure_project_dirs()
    clean = _read_jsonl(BASE_CORPUS_PATH)
    poisoned = _read_jsonl(POISONED_DOCS_PATH)
    retriever = HybridRetriever(DEMO_CORPUS_PATH if DEMO_CORPUS_PATH.exists() else BASE_CORPUS_PATH)
    clean_embeddings = retriever.encode([doc["text"] for doc in clean])
    covariance = LedoitWolf().fit(clean_embeddings)
    clean_mean = covariance.location_
    clean_cov_inv = covariance.precision_
    iforest = IsolationForest(n_estimators=200, random_state=42, contamination="auto").fit(clean_embeddings)

    docs = poisoned + clean
    labels = np.asarray([1] * len(poisoned) + [0] * len(clean), dtype=int)
    embeddings = retriever.encode([doc["text"] for doc in docs])
    rows = []
    for doc, embedding in zip(docs, embeddings):
        query = doc.get("target_query") or doc.get("title") or doc["text"].split(".", 1)[0]
        behaviour = behavioural_signal_components(query, doc["text"], retriever)
        rows.append([
            mahalanobis_score(embedding, clean_mean, clean_cov_inv),
            isolation_forest_score(embedding, iforest),
            behaviour["relevance_spike"], behaviour["instruction"],
            behaviour["url"], behaviour["authority"],
            _dense_rank(retriever, query, int(doc["doc_id"])),
        ])
    features = np.asarray(rows, dtype=np.float64)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42))
    probabilities = cross_val_predict(estimator, features, labels, cv=LeaveOneOut(),
                                      method="predict_proba")[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "clean_samples": len(clean), "poisoned_samples": len(poisoned),
        "embedding_backend": retriever.embedder.model_name,
        "evaluation": "Leave-one-out cross-validation",
        "note": "Controlled small-sample Review-1 result; re-evaluate with RL-generated attacks in Phase 2.",
    }
    estimator.fit(features, labels)
    scaler, classifier = estimator.steps[0][1], estimator.steps[1][1]
    clean_distances = [mahalanobis_score(e, clean_mean, clean_cov_inv) for e in clean_embeddings]
    bundle = {"classifier": classifier, "scaler": scaler, "clean_mean": clean_mean,
              "clean_cov_inv": clean_cov_inv, "isolation_forest": iforest,
              "mahal_p95": float(np.percentile(clean_distances, 95)),
              "feature_names": FEATURE_NAMES, "metrics": metrics,
              "embedding_backend": retriever.embedder.model_name}
    backend_path = ARTIFACTS_DIR / f"fusion_classifier_{retriever.embedder.backend}.joblib"
    joblib.dump(bundle, backend_path)
    joblib.dump(bundle, ARTIFACTS_DIR / "fusion_classifier.joblib")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fpr, tpr, _ = roc_curve(labels, probabilities)
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, color="#0c6973", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="#98a7aa"); plt.xlabel("False positive rate")
    plt.ylabel("True positive rate"); plt.title("Fusion detector - LOOCV ROC")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=180); plt.close()
    matrix = confusion_matrix(labels, predictions)
    ConfusionMatrixDisplay(matrix, display_labels=["Clean", "Poisoned"]).plot(cmap="Blues", colorbar=False)
    plt.title("Fusion detector - LOOCV confusion matrix"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180); plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
