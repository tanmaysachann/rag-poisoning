"""Calibration-only exploration of multi-sentence factual verification.

This changes the v7 representation after its test result was seen. The output
is development evidence only; it is not a new sealed-test claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging

from detect.research_detector import threshold_for_fpr
from detect.signals import split_sentences
from detect.train_factual_verifier_v6 import LABEL_TO_ID, predict
from evaluation.evaluate_external_rag_v7 import candidates, clean_scores, read_cases, summary
from pipeline.secure_rag import _effective_query_terms


def ranked_sentences(query: str, text: str, limit: int = 3) -> list[str]:
    terms = _effective_query_terms(query)
    sentences = [part.strip() for line in text.splitlines()
                 for part in split_sentences(line) if len(re.findall(r"\w+", part)) >= 6]
    if not sentences:
        return [text]

    def key(item: tuple[int, str]) -> tuple[int, float, int, int]:
        index, sentence = item
        words = _effective_query_terms(sentence)
        overlap = len(terms & words)
        return (overlap, overlap / max(len(words), 1),
                -abs(len(re.findall(r"\w+", sentence)) - 25), -index)

    return [sentence for _, sentence in sorted(enumerate(sentences), key=key,
                                              reverse=True)[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/external_rag_v7"))
    parser.add_argument("--model", type=Path,
                        default=Path("artifacts/vendor/albert-base-vitaminc"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/external_rag_v7_multisentence_dev.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    cases = read_cases(args.data / "calibration.jsonl",
                       manifest["file_sha256"]["calibration.jsonl"])
    rows = candidates(cases)
    keys = list(rows)
    flat = []
    ranges = {}
    for key in keys:
        row = rows[key]
        start = len(flat)
        flat.extend({"claim": sentence, "evidence": row["evidence"]}
                    for sentence in ranked_sentences(row["query"], row["text"]))
        ranges[key] = (start, len(flat))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformers_logging.set_verbosity_error()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True
    ).to(device)
    probabilities = []
    started = time.perf_counter()
    for start in range(0, len(flat), 512):
        part = predict(model, tokenizer, flat[start:start + 512], device, 16, 256,
                       evidence_first=True)
        probabilities.extend(part[:, LABEL_TO_ID["REFUTES"]].astype(float))
        print(f"pairs: {start + len(part)}/{len(flat)}", flush=True)
    report = {"scope": "calibration-only; post-v7 exploratory representation change",
              "source_manifest_sha256": hashlib.sha256(
                  (args.data / "manifest.json").read_bytes()
              ).hexdigest(), "cases": len(cases), "unique_documents": len(rows),
              "sentence_pairs": len(flat), "elapsed_seconds": round(time.perf_counter()-started, 2),
              "variants": {}}
    for count in (1, 2, 3):
        scores = {key: max(probabilities[start:min(end, start + count)])
                  for key, (start, end) in ranges.items()}
        threshold = threshold_for_fpr(clean_scores(cases, scores), 0.01)
        report["variants"][f"top_{count}_max"] = {
            "threshold": threshold, "calibration": summary(cases, scores, threshold)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
