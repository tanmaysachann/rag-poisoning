"""Evaluate real Qwen decoder hidden-state and attention signals on validation attacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from detect.decoder_probes import DecoderProbeReference, PROBE_VERSION, QwenDecoderProbes
from detect.research_detector import threshold_for_fpr


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure(probe, rows: list[dict], *, label: str):
    hidden, attention, token_counts, truncations = [], [], [], []
    for index, row in enumerate(rows, 1):
        vector, concentration, count, truncated = probe.extract(row["question"], row["text"])
        hidden.append(vector)
        attention.append(concentration)
        token_counts.append(count)
        truncations.append(truncated)
        if index % 25 == 0:
            print(f"{label}: {index}/{len(rows)}", flush=True)
    return np.asarray(hidden), np.asarray(attention), token_counts, truncations


def _clean_rows(benchmark: Path, split: str) -> list[dict]:
    docs = {row["doc_id"]: row for row in _read_jsonl(benchmark / split / "corpus.jsonl")}
    return [
        {"qid": query["qid"], "question": query["question"],
         "text": docs[query["support_doc_ids"][0]]["text"]}
        for query in _read_jsonl(benchmark / split / "queries.jsonl")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data/benchmark")
    parser.add_argument("--model-path", type=Path, default=ROOT_DIR / "artifacts/models/qwen2_5_0_5b")
    parser.add_argument("--reference-count", type=int, default=50)
    parser.add_argument("--calibration-count", type=int, default=25)
    parser.add_argument("--limit", type=int, default=75)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "decoder_probes_validation.json")
    args = parser.parse_args()
    if args.reference_count < 5 or args.calibration_count < 5 or args.limit < 1:
        raise ValueError("Reference, calibration, and validation counts are too small")
    validate_benchmark(args.benchmark)
    train = _clean_rows(args.benchmark, "train")
    validation = _clean_rows(args.benchmark, "validation")[:args.limit]
    if args.reference_count + args.calibration_count > len(train) or len(validation) != args.limit:
        raise ValueError("Requested more rows than the benchmark split contains")
    model_weight = args.model_path / "model.safetensors"
    if not model_weight.is_file():
        raise FileNotFoundError(model_weight)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    torch.set_num_threads(1)
    probe = QwenDecoderProbes(args.model_path, max_length=args.max_length)
    ref_hidden, ref_attention, _, ref_truncated = _measure(
        probe, train[:args.reference_count], label="reference"
    )
    reference = DecoderProbeReference.fit(ref_hidden, ref_attention)
    cal_hidden, cal_attention, _, cal_truncated = _measure(
        probe, train[args.reference_count:args.reference_count + args.calibration_count],
        label="calibration",
    )
    keys = ("decoder_layer_mahalanobis", "decoder_attention_deviation")
    cal_scores = [reference.score(vector, attention)
                  for vector, attention in zip(cal_hidden, cal_attention)]
    thresholds = {key: threshold_for_fpr(np.asarray([row[key] for row in cal_scores]), args.target_fpr)
                  for key in keys}
    clean_hidden, clean_attention, clean_tokens, clean_truncated = _measure(
        probe, validation, label="validation clean"
    )
    clean_scores = [reference.score(vector, attention)
                    for vector, attention in zip(clean_hidden, clean_attention)]
    families = {}
    cases = []
    for family in ("greedy", "stealth"):
        attack_path = RESULTS_DIR / f"attack_baseline_validation_{family}.jsonl"
        attacks = {row["qid"]: row for row in _read_jsonl(attack_path)}
        if set(attacks) != {row["qid"] for row in _clean_rows(args.benchmark, "validation")}:
            raise ValueError(f"Incomplete {family} validation attack family")
        attack_rows = [{"qid": row["qid"], "question": row["question"],
                        "text": attacks[row["qid"]]["text"]} for row in validation]
        attack_hidden, attack_attention, attack_tokens, attack_truncated = _measure(
            probe, attack_rows, label=f"validation {family}"
        )
        attack_scores = [reference.score(vector, attention)
                         for vector, attention in zip(attack_hidden, attack_attention)]
        family_metrics = {}
        for key in keys:
            clean = np.asarray([row[key] for row in clean_scores])
            attack = np.asarray([row[key] for row in attack_scores])
            family_metrics[key] = {
                "train_calibrated_threshold": thresholds[key],
                "validation_clean_fpr": float(np.mean(clean >= thresholds[key])),
                "validation_attack_recall": float(np.mean(attack >= thresholds[key])),
                "validation_roc_auc": float(roc_auc_score(
                    np.r_[np.zeros(len(clean)), np.ones(len(attack))], np.r_[clean, attack]
                )),
            }
        families[family] = family_metrics
        for index, query in enumerate(validation):
            cases.append({
                "qid": query["qid"], "family": family,
                "clean": clean_scores[index], "attack": attack_scores[index],
                "clean_tokens": clean_tokens[index], "attack_tokens": attack_tokens[index],
                "clean_truncated": clean_truncated[index],
                "attack_truncated": attack_truncated[index],
            })
    summary = {
        "probe_version": PROBE_VERSION,
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "model_weight_sha256": _sha256(model_weight),
        "layer_index": probe.layer_index, "max_length": probe.max_length,
        "reference_train_cases": args.reference_count,
        "calibration_train_cases": args.calibration_count,
        "validation_cases_per_family": len(validation),
        "target_train_fpr": args.target_fpr,
        "truncated_reference_cases": sum(ref_truncated),
        "truncated_calibration_cases": sum(cal_truncated),
        "truncated_clean_validation_cases": sum(clean_truncated),
        "families": families,
        "note": "Decoder prefill internals, not generated-response activations; research-only signal",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
