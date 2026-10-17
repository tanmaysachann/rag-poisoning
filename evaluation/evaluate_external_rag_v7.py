"""Frozen, reference-aware detector evaluation on an external RAG attack archive.

Calibration reads only clean calibration candidates. Attack documents are never
used for threshold selection. Sealed test scores are computed afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detect.research_detector import ResearchDetector, threshold_for_fpr
from detect.signals import split_sentences
from detect.train_factual_verifier_v6 import LABEL_TO_ID, predict
from pipeline.secure_rag import _effective_query_terms


def read_cases(path: Path, expected_hash: str) -> list[dict]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"External case file changed: {path}")
    return [json.loads(line) for line in data.splitlines()]


def claim_sentence(query: str, text: str) -> str:
    """Pick one relevant factual sentence without answer or attack labels."""
    terms = _effective_query_terms(query)
    sentences = [part.strip() for line in text.splitlines()
                 for part in split_sentences(line) if part.strip()]
    viable = [sentence for sentence in sentences
              if len(re.findall(r"\w+", sentence)) >= 6]
    if not viable:
        viable = sentences or [text]

    def rank(index_sentence: tuple[int, str]) -> tuple[float, int, int, int]:
        index, sentence = index_sentence
        words = _effective_query_terms(sentence)
        overlap = len(terms & words)
        length = len(re.findall(r"\w+", sentence))
        return (overlap / max(len(terms), 1), overlap, -abs(length - 25), -index)

    return max(enumerate(viable), key=rank)[1]


def candidates(cases: list[dict]) -> dict[tuple[str, str], dict]:
    rows = {}
    for case in cases:
        if not case["reference_id"]:
            continue
        for retrieval in case["retrievals"].values():
            for candidate in retrieval["candidates"]:
                key = (case["query_id"], candidate["id"])
                if key in rows:
                    if rows[key]["text"] != candidate["text"]:
                        raise ValueError(f"Different text for {key}")
                    continue
                rows[key] = {"claim": claim_sentence(case["query"], candidate["text"]),
                             "evidence": case["reference_text"],
                             "query": case["query"], "text": candidate["text"]}
    return rows


def model_scores(path: Path, rows: dict, evidence_first: bool, device: torch.device,
                 batch_size: int) -> tuple[dict, float]:
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                              use_fast=not evidence_first)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True
    ).to(device)
    model.eval()
    if {int(key): value for key, value in model.config.id2label.items()} != {
        index: label for label, index in LABEL_TO_ID.items()
    }:
        raise ValueError(f"Unexpected label order: {path}")
    keys = list(rows)
    scores = {}
    started = time.perf_counter()
    for start in range(0, len(keys), 1024):
        part = keys[start:start + 1024]
        probabilities = predict(model, tokenizer, [rows[key] for key in part],
                                device, batch_size, 256,
                                evidence_first=evidence_first)
        scores.update(zip(part, probabilities[:, LABEL_TO_ID["REFUTES"]].astype(float)))
        print(f"{path.name}: {start + len(part)}/{len(keys)} pairs", flush=True)
    seconds = time.perf_counter() - started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, seconds


def clean_scores(cases: list[dict], scores: dict) -> np.ndarray:
    return np.asarray([scores[case["query_id"], candidate["id"]]
                       for case in cases if case["reference_id"]
                       for candidate in case["retrievals"]["clean"]["candidates"]],
                      dtype=np.float64)


def summary(cases: list[dict], scores: dict, threshold: float) -> dict:
    clean_flags = 0
    clean_total = 0
    reference_flags = 0
    reference_total = 0
    families = {family: {"own_poison_caught": 0, "own_poison_retrieved": 0,
                         "any_poison_flags": 0, "any_poison_total": 0}
                for family in ("naive", "corruptrag_ak")}
    for case in cases:
        if not case["reference_id"]:
            continue
        qid = case["query_id"]
        for candidate in case["retrievals"]["clean"]["candidates"]:
            flag = scores[qid, candidate["id"]] >= threshold
            clean_flags += flag
            clean_total += 1
            if candidate["id"] == case["reference_id"]:
                reference_flags += flag
                reference_total += 1
        for family in families:
            attack = case["retrievals"][family]
            if attack["own_poison_retrieved"]:
                own_id = (f"poisoned-naive-q:{qid}" if family == "naive"
                          else f"poisoned-corruptrag-ak-q:{qid}")
                families[family]["own_poison_retrieved"] += 1
                families[family]["own_poison_caught"] += scores[qid, own_id] >= threshold
            for candidate in attack["candidates"]:
                if candidate["id"].startswith("poisoned-"):
                    families[family]["any_poison_total"] += 1
                    families[family]["any_poison_flags"] += (
                        scores[qid, candidate["id"]] >= threshold
                    )
    return {
        "clean_flags": int(clean_flags), "clean_total": clean_total,
        "clean_false_quarantine_rate": clean_flags / clean_total,
        "reference_flags": int(reference_flags), "reference_total": reference_total,
        "other_clean_flags": int(clean_flags - reference_flags),
        "other_clean_total": clean_total - reference_total,
        "families": {family: {**{key: int(value) for key, value in values.items()},
                    "conditional_own_poison_catch_rate":
                    values["own_poison_caught"] / values["own_poison_retrieved"]}
                    for family, values in families.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/external_rag_v7"))
    parser.add_argument("--output", type=Path, default=Path("results/external_rag_v7.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    calibration = read_cases(args.data / "calibration.jsonl",
                             manifest["file_sha256"]["calibration.jsonl"])
    sealed = read_cases(args.data / "test.jsonl",
                        manifest["file_sha256"]["test.jsonl"])
    if {x["query_id"] for x in calibration} & {x["query_id"] for x in sealed}:
        raise ValueError("Calibration and sealed questions overlap")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"source_manifest_sha256": hashlib.sha256(
        (args.data / "manifest.json").read_bytes()
    ).hexdigest(), "calibration_questions": len(calibration),
              "test_questions": len(sealed), "device": str(device), "models": {}}
    for name, path, evidence_first in (
        ("locally_trained_minilm", Path("artifacts/factual_verifier_v6"), False),
        ("published_vitaminc_albert", Path("artifacts/vendor/albert-base-vitaminc"), True),
    ):
        cal_rows = candidates(calibration)
        cal_scores, cal_seconds = model_scores(path, cal_rows, evidence_first,
                                               device, args.batch_size)
        threshold = threshold_for_fpr(clean_scores(calibration, cal_scores), 0.01)
        cal_summary = summary(calibration, cal_scores, threshold)
        print(f"{name}: frozen threshold={threshold:.8f}; "
              f"clean calibration flags={cal_summary['clean_flags']}/"
              f"{cal_summary['clean_total']}", flush=True)
        del cal_rows, cal_scores
        test_rows = candidates(sealed)
        test_scores, test_seconds = model_scores(path, test_rows, evidence_first,
                                                 device, args.batch_size)
        test_summary = summary(sealed, test_scores, threshold)
        report["models"][name] = {
            "threshold": threshold, "calibration": cal_summary, "test": test_summary,
            "calibration_scoring_seconds": round(cal_seconds, 2),
            "test_scoring_seconds": round(test_seconds, 2),
            "test_unique_pairs": len(test_rows),
            "milliseconds_per_test_pair": round(test_seconds / len(test_rows) * 1000, 2),
        }
        del test_rows, test_scores
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    old = joblib.load("artifacts/research_detector_v3_hashing.joblib")
    if not isinstance(old, ResearchDetector):
        raise TypeError("Unexpected v3 detector artifact")
    old_scores = {}
    started = time.perf_counter()
    for index, (key, row) in enumerate(candidates(sealed).items(), 1):
        old_scores[key] = old.score(row["query"], row["text"])["risk_score"]
        if index % 1000 == 0:
            print(f"v3: {index} pairs", flush=True)
    seconds = time.perf_counter() - started
    report["models"]["frozen_v3_content_detector"] = {
        "threshold": old.threshold, "test": summary(sealed, old_scores, old.threshold),
        "test_scoring_seconds": round(seconds, 2),
        "test_unique_pairs": len(old_scores),
        "milliseconds_per_test_pair": round(seconds / len(old_scores) * 1000, 2),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
