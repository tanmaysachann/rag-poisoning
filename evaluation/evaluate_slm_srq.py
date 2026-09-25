"""Measure the slide-7 SRQ with a real local SLM on train/validation only.

This is an optional, expensive research probe. It never changes the frozen
serving detector and never reads the benchmark test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CachedWordEmbedder:
    """Batch only new vocabulary terms; share vectors across SRQ cases."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.model_name = backend.model_name
        self.vectors: dict[str, np.ndarray] = {}

    def encode(self, words: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(word for word in words if word not in self.vectors))
        if missing:
            vectors = self.backend.encode(missing)
            self.vectors.update(zip(missing, vectors))
        return np.stack([self.vectors[word] for word in words]).astype(np.float32)


def _support_cases(benchmark: Path, split: str) -> list[dict]:
    docs = {row["doc_id"]: row for row in _read_jsonl(benchmark / split / "corpus.jsonl")}
    return [
        {"qid": query["qid"], "question": query["question"],
         "text": docs[query["support_doc_ids"][0]]["text"]}
        for query in _read_jsonl(benchmark / split / "queries.jsonl")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--model-path", type=Path, default=ROOT_DIR / "artifacts/models/qwen2_5_0_5b")
    parser.add_argument("--calibration-limit", type=int, default=20)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.calibration_limit, args.validation_limit, args.max_new_tokens) <= 0:
        raise ValueError("Limits and generation length must be positive")
    weight_path = args.model_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Local SLM weights are missing: {args.model_path}")
    weight_sha256 = _sha256(weight_path)
    os.environ["RAG_LLM_MODEL"] = str(args.model_path.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from data.validate_benchmark import validate_benchmark
    from detect.slm_srq import LocalSLMProbe, calibrate_srq
    from retrieval.hybrid_retriever import TextEmbedder

    validate_benchmark(args.benchmark)

    encoder = TextEmbedder(preferred_backend="sentence-transformers")
    if encoder.backend != "sentence-transformers":
        raise RuntimeError("Cached MiniLM is required for the semantic SRQ probe")
    probe = LocalSLMProbe(
        CachedWordEmbedder(encoder), model_name=os.environ["RAG_LLM_MODEL"],
        max_new_tokens=args.max_new_tokens,
    )

    calibration = _support_cases(args.benchmark, "train")
    validation = _support_cases(args.benchmark, "validation")
    random.Random(args.seed).shuffle(calibration)
    random.Random(args.seed + 1).shuffle(validation)
    calibration = calibration[:args.calibration_limit]
    validation = validation[:args.validation_limit]
    if len(calibration) != args.calibration_limit or len(validation) != args.validation_limit:
        raise ValueError("Requested more cases than the benchmark split contains")
    selected_qids = {row["qid"] for row in validation}
    attacks = {}
    for family in ("greedy", "stealth"):
        path = RESULTS_DIR / f"attack_baseline_validation_{family}.jsonl"
        rows = {row["qid"]: row for row in _read_jsonl(path)}
        if not selected_qids <= rows.keys() or any(rows[qid]["split"] != "validation" for qid in selected_qids):
            raise ValueError(f"Missing or wrong-split {family} validation attack")
        attacks[family] = rows

    cases: list[dict] = []
    for label, rows in (("calibration_clean", calibration), ("validation_clean", validation)):
        for row in rows:
            cases.append({"group": label, "qid": row["qid"],
                          **probe.score(row["question"], row["text"])})
    for family in ("greedy", "stealth"):
        for row in validation:
            attack = attacks[family][row["qid"]]
            cases.append({"group": f"validation_{family}", "qid": row["qid"],
                          **probe.score(row["question"], attack["text"])})

    clean_train = [row["raw_srq"] for row in cases if row["group"] == "calibration_clean"]
    threshold = calibrate_srq(clean_train, target_fpr=0.05)
    clean_val = [row["raw_srq"] for row in cases if row["group"] == "validation_clean"]
    results = {
        "model": str(args.model_path), "model_weight_sha256": weight_sha256,
        "embedding_model": encoder.model_name,
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "calibration_clean": len(clean_train), "validation_clean": len(clean_val),
        "threshold": threshold,
        "clean_validation_fpr": sum(score >= threshold for score in clean_val) / len(clean_val),
        "families": {},
        "note": "Train-clean threshold; small validation subset, real local SLM, not wired into quarantine",
    }
    for family in ("greedy", "stealth"):
        scores = [row["raw_srq"] for row in cases if row["group"] == f"validation_{family}"]
        results["families"][family] = {
            "cases": len(scores),
            "recall": sum(score >= threshold for score in scores) / len(scores),
            "roc_auc": float(roc_auc_score([0] * len(clean_val) + [1] * len(scores), clean_val + scores)),
            "mean_clean_srq": float(np.mean(clean_val)),
            "mean_attack_srq": float(np.mean(scores)),
        }
    output = args.output or RESULTS_DIR / f"slm_srq_validation_{args.calibration_limit}_{args.validation_limit}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
