"""Build an S1-S4 gate with leave-one-out clean ranks from train queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark")
    parser.add_argument("--qwen-model", type=Path, default=ROOT / "artifacts" / "models" / "qwen2_5_0_5b")
    parser.add_argument("--reference-count", type=int, default=100)
    parser.add_argument("--calibration-count", type=int, default=50)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "four_probe_calibration.joblib")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "four_probe_calibration.json")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if args.reference_count < 5 or args.calibration_count < 20:
        raise ValueError("Use at least five reference and twenty calibration examples")
    weight_path = args.qwen_model / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Local Qwen weights are unavailable: {weight_path}")
    if args.output.exists() and not args.replace_existing:
        raise FileExistsError(f"Calibration already exists: {args.output}; pass --replace-existing to rebuild")

    os.environ["RAG_USE_MINILM"] = "0"
    os.environ["RAG_LLM_MODEL"] = str(args.qwen_model.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from data.validate_benchmark import validate_benchmark
    from detect.model_probes import MiniLMModelProbes, ProbeReference
    from detect.slm_srq import LocalSLMProbe, score_srq
    from pipeline.secure_rag import _counterfactual_influence, _select_answer
    from retrieval.hybrid_retriever import HybridRetriever, TextEmbedder
    from security.four_probe_gate import build_calibration, save_bundle, write_calibration_summary

    validate_benchmark(args.benchmark)
    torch.set_num_threads(1)
    train_dir = args.benchmark / "train"
    queries = _jsonl(train_dir / "queries.jsonl")
    documents = {int(row["doc_id"]): row for row in _jsonl(train_dir / "corpus.jsonl")}
    if args.reference_count + args.calibration_count > len(queries):
        raise ValueError("Reference and calibration counts exceed the clean train split")
    reference_queries = queries[:args.reference_count]
    calibration_queries = queries[args.reference_count:args.reference_count + args.calibration_count]

    minilm = MiniLMModelProbes(local_files_only=True)
    reference_hidden, reference_attention = [], []
    for index, row in enumerate(reference_queries, 1):
        source = documents[int(row["support_doc_ids"][0])]
        hidden, attention = minilm.extract(row["question"], source["text"])
        reference_hidden.append(hidden)
        reference_attention.append(attention)
        if index % 25 == 0:
            print(f"S1/S2 clean reference {index}/{len(reference_queries)}", flush=True)
    probe_reference = ProbeReference.fit(np.asarray(reference_hidden), np.asarray(reference_attention))

    encoder = TextEmbedder(preferred_backend="sentence-transformers")
    if encoder.backend != "sentence-transformers":
        raise RuntimeError("The joint probe gate requires the pinned MiniLM encoder")

    class WordCache:
        def __init__(self, backend):
            self.backend = backend
            self.model_name = backend.model_name
            self.vectors = {}

        def encode(self, words):
            missing = list(dict.fromkeys(word for word in words if word not in self.vectors))
            if missing:
                vectors = self.backend.encode(missing)
                self.vectors.update(zip(missing, vectors))
            return np.stack([self.vectors[word] for word in words]).astype(np.float32)

    word_embedder = WordCache(encoder)
    slm = LocalSLMProbe(word_embedder, model_name=str(args.qwen_model.resolve()))
    retriever = HybridRetriever(train_dir / "corpus.jsonl")
    raw_rows = []
    for index, row in enumerate(calibration_queries, 1):
        support_id = int(row["support_doc_ids"][0])
        support = documents[support_id]
        hidden, attention = minilm.extract(row["question"], support["text"])
        model_scores = probe_reference.score(hidden, attention)
        response = slm.preliminary_response(row["question"], support["text"])
        srq_score = score_srq(row["question"], support["text"], response, word_embedder)["raw_srq"]

        candidates = retriever.retrieve(row["question"], top_k=5)
        baseline_answer, baseline_source, baseline_span = _select_answer(
            row["question"], candidates, retriever
        )
        loo_influence = 0.0
        if any(int(doc["doc_id"]) == support_id for doc in candidates):
            ablated = [doc for doc in candidates if int(doc["doc_id"]) != support_id]
            after_answer, after_source, after_span = _select_answer(row["question"], ablated, retriever)
            loo_influence = _counterfactual_influence(
                next(doc for doc in candidates if int(doc["doc_id"]) == support_id),
                retriever,
                (baseline_answer, baseline_source, baseline_span),
                (after_answer, after_source, after_span),
            )
        raw_rows.append({
            "s1_hidden_state_mahalanobis": float(model_scores["layer3_mahalanobis"]),
            "s2_attention_deviation": float(model_scores["attention_deviation"]),
            "s3_slm_srq": float(srq_score),
            "s4_loo_influence": float(loo_influence),
        })
        if index % 5 == 0:
            print(f"Joint clean calibration {index}/{len(calibration_queries)}", flush=True)

    bundle = build_calibration(raw_rows, args.target_fpr)
    bundle.update({
        "minilm_reference": {
            "mean": probe_reference.mean,
            "precision": probe_reference.precision,
            "attention_mean": probe_reference.attention_mean,
            "attention_std": probe_reference.attention_std,
        },
        "minilm_model": minilm.model_name,
        "minilm_model_revision": minilm.model_revision,
        "qwen_model": str(args.qwen_model.resolve()),
        "qwen_weight_sha256": _sha256(weight_path),
        "reference_qids": [row["qid"] for row in reference_queries],
        "calibration_qids": [row["qid"] for row in calibration_queries],
        "training_corpus_sha256": _sha256(train_dir / "corpus.jsonl"),
        "training_queries_sha256": _sha256(train_dir / "queries.jsonl"),
        "probe_reference_count": len(reference_queries),
        "calibration_count": len(calibration_queries),
        "calibration_rows": raw_rows,
    })
    save_bundle(args.output, bundle)
    write_calibration_summary(args.summary, bundle)
    print(json.dumps({
        "calibration_version": bundle["calibration_version"],
        "rows": bundle["rows"], "threshold": bundle["joint_threshold"],
        "target_clean_fpr": bundle["target_clean_fpr"],
        "artifact": str(args.output), "summary": str(args.summary),
    }, indent=2))


if __name__ == "__main__":
    main()
