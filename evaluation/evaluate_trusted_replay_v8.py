"""Replay one QA reader with and without a benchmark trusted-reference policy.

The all-passages answer picks the highest-scoring span from the logged top 10.
The reference policy emits only from the BEIR relevance-judged clean passage
when that passage is present in the logged top 10. Neither is the archived LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from evaluation.evaluate_external_rag_v7 import candidates, read_cases
from evaluation.evaluate_trusted_reference_v8 import MODELS, choose_threshold, outcomes
from generation.trusted_reference import ReferenceReader, answer_overlap


def best_predictions(cases: list[dict], scores: dict, family: str) -> list[dict]:
    result = []
    for case in cases:
        qid = case["query_id"]
        docs = case["retrievals"][family]["candidates"]
        best = max(docs, key=lambda doc: scores[qid, doc["id"]]["score"])
        result.append(scores[qid, best["id"]])
    return result


def reference_predictions(cases: list[dict], scores: dict) -> list[dict]:
    return [scores[case["query_id"], case["reference_id"]] for case in cases]


def inline_reference_predictions(cases: list[dict], scores: dict,
                                 family: str) -> list[dict]:
    result = []
    for case in cases:
        ids = {doc["id"] for doc in case["retrievals"][family]["candidates"]}
        result.append(scores[case["query_id"], case["reference_id"]]
                      if case["reference_id"] in ids else {"answer": "", "score": 0.0})
    return result


def exposed_metrics(cases: list[dict], predictions: list[dict],
                    threshold: float, family: str) -> dict:
    selected = [(case, prediction) for case, prediction in zip(cases, predictions)
                if case["retrievals"][family]["own_poison_retrieved"]]
    return outcomes([case for case, _ in selected],
                    [prediction for _, prediction in selected], threshold)


def disagreement(cases: list[dict], scores: dict, reference_threshold: float,
                 candidate_threshold: float) -> dict:
    result = {}
    for family in ("clean", "naive", "corruptrag_ak"):
        counts = {"reference_answer_available": 0, "candidate_answer_available": 0,
                  "disagreements": 0, "candidate_total": 0,
                  "own_poison_disagreements": 0, "own_poison_total": 0}
        for case in cases:
            qid = case["query_id"]
            reference = scores[qid, case["reference_id"]]
            if reference["score"] < reference_threshold:
                continue
            counts["reference_answer_available"] += 1
            for doc in case["retrievals"][family]["candidates"]:
                if doc["id"] == case["reference_id"]:
                    continue
                counts["candidate_total"] += 1
                row = scores[qid, doc["id"]]
                if row["score"] < candidate_threshold:
                    continue
                counts["candidate_answer_available"] += 1
                different = not answer_overlap(row["answer"], reference["answer"])
                counts["disagreements"] += different
                if doc["id"] == f"poisoned-naive-q:{qid}" or doc["id"] == f"poisoned-corruptrag-ak-q:{qid}":
                    counts["own_poison_total"] += 1
                    counts["own_poison_disagreements"] += different
        result[family] = counts
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/external_rag_v7"))
    parser.add_argument("--reader", choices=tuple(MODELS), default="roberta_nq")
    parser.add_argument("--reader-calibration", type=Path,
                        default=Path("results/trusted_reference_v8_roberta_nq_calibration.json"))
    parser.add_argument("--calibration-report", type=Path,
                        default=Path("results/trusted_replay_v8_calibration.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trusted_replay_v8_test.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    cases = read_cases(args.data / f"{args.split}.jsonl",
                       manifest["file_sha256"][f"{args.split}.jsonl"])
    reader_cal = json.loads(args.reader_calibration.read_text(encoding="utf-8"))
    if reader_cal["model"] != MODELS[args.reader]["name"]:
        raise ValueError("Reader and reference calibration differ")
    model = MODELS[args.reader]
    if hashlib.sha256((model["path"] / model["weight"]).read_bytes()).hexdigest() != model["sha256"]:
        raise ValueError("Pinned QA model hash mismatch")
    reader = ReferenceReader(MODELS[args.reader]["path"])
    rows = candidates(cases)
    keys = list(rows)
    started = time.perf_counter()
    scores = {}
    for start in range(0, len(keys), 512):
        part = keys[start:start + 512]
        extracted = reader.extract_batch([rows[key]["query"] for key in part],
                                         [rows[key]["text"] for key in part])
        scores.update((key, {"answer": pred.answer, "score": pred.score})
                      for key, pred in zip(part, extracted))
        print(f"QA pairs: {start + len(part)}/{len(keys)}", flush=True)
    if args.split == "calibration":
        clean_best = best_predictions(cases, scores, "clean")
        all_threshold, clean_metrics = choose_threshold(cases, clean_best)
        output = args.calibration_report
    else:
        calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
        if calibration["source_manifest_sha256"] != hashlib.sha256(
            (args.data / "manifest.json").read_bytes()
        ).hexdigest():
            raise ValueError("Replay calibration source changed")
        all_threshold = calibration["all_passages_min_score"]
        clean_metrics = None
        output = args.output
    reference_threshold = reader_cal["min_reader_score"]
    summary = {}
    for family in ("clean", "naive", "corruptrag_ak"):
        all_predictions = best_predictions(cases, scores, family)
        inline_predictions = inline_reference_predictions(cases, scores, family)
        entry = {
            "all_passages": outcomes(cases, all_predictions, all_threshold),
            "all_passages_without_abstention_cutoff": outcomes(cases, all_predictions, 0.0),
            "inline_reference_only": outcomes(cases, inline_predictions, reference_threshold),
            "separate_reference_channel_oracle": outcomes(
                cases, reference_predictions(cases, scores), reference_threshold
            ),
        }
        if family != "clean":
            entry["all_passages_when_own_poison_retrieved"] = exposed_metrics(
                cases, all_predictions, all_threshold, family
            )
            entry["inline_reference_when_own_poison_retrieved"] = exposed_metrics(
                cases, inline_predictions, reference_threshold, family
            )
        summary[family] = entry
    report = {"split": args.split,
              "source_manifest_sha256": hashlib.sha256(
                  (args.data / "manifest.json").read_bytes()
              ).hexdigest(), "reader": MODELS[args.reader]["name"],
              "all_passages_min_score": all_threshold,
              "reference_min_score": reference_threshold,
              "selection": "highest QA span score from logged top10 vs benchmark reference if present",
              "calibration_clean_all_passages": clean_metrics,
              "families": summary,
              "disagreement": disagreement(cases, scores, reference_threshold,
                                           all_threshold),
              "unique_pairs": len(keys), "elapsed_seconds": round(time.perf_counter()-started, 2),
              "limitation": "Benchmark qrel reference is an oracle-like proxy, not a live authenticated origin."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
