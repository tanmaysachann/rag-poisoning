"""Reference-only QA replay with a calibration-frozen abstention threshold.

The source is a benchmark relevance judgment, not a live authentication proof.
No attack or answer key is passed to the QA reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from evaluation.evaluate_external_rag_v7 import read_cases
from generation.answer_corroboration import token_f1
from generation.trusted_reference import ReferenceReader, answer_overlap


MODELS = {
    "tinyroberta": {
        "path": Path("artifacts/vendor/tinyroberta-squad2"),
        "weight": "model.safetensors",
        "sha256": "39955cbf5f636762a2ef642b718c8c6e172d153c94a13de40b37efd00822cd80",
        "name": "deepset/tinyroberta-squad2",
    },
    "roberta_nq": {
        "path": Path("artifacts/vendor/roberta-base-squad2-nq"),
        "weight": "pytorch_model.bin",
        "sha256": "b2f61c7b1710fcc6826fc0cc12dca8c59c0fbdadd82c05bb6ebff2028583fcf6",
        "name": "nlpconnect/roberta-base-squad2-nq",
    },
}


def outcomes(cases: list[dict], predictions: list[dict], threshold: float) -> dict:
    result = {"questions": len(cases), "correct_match": 0, "wrong_emitted": 0,
              "target_match": 0, "abstained": 0, "token_f1_sum": 0.0,
              "token_f1_at_least_half": 0}
    for case, prediction in zip(cases, predictions):
        answer = prediction["answer"] if prediction["score"] >= threshold else ""
        if not answer:
            result["abstained"] += 1
            continue
        correct = answer_overlap(answer, case["correct_answer"])
        overlap_score = token_f1(answer, case["correct_answer"])
        result["correct_match"] += correct
        result["wrong_emitted"] += not correct
        result["target_match"] += answer_overlap(answer, case["target_answer"])
        result["token_f1_sum"] += overlap_score
        result["token_f1_at_least_half"] += overlap_score >= 0.5
    result["correct_match_rate"] = result["correct_match"] / len(cases)
    result["wrong_emitted_rate"] = result["wrong_emitted"] / len(cases)
    result["abstention_rate"] = result["abstained"] / len(cases)
    result["mean_token_f1_all_questions"] = result["token_f1_sum"] / len(cases)
    return result


def choose_threshold(cases: list[dict], predictions: list[dict]) -> tuple[float, dict]:
    """Maximize correct minus twice wrong, breaking ties toward coverage."""
    candidates = sorted({0.0, 1.0, *(row["score"] for row in predictions)})
    ranked = []
    for threshold in candidates:
        result = outcomes(cases, predictions, threshold)
        utility = result["correct_match"] - 2 * result["wrong_emitted"]
        ranked.append((utility, result["correct_match"], -threshold, threshold, result))
    _, _, _, threshold, metrics = max(ranked)
    return threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--reader", choices=tuple(MODELS), default="tinyroberta")
    parser.add_argument("--data", type=Path, default=Path("data/external_rag_v7"))
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = MODELS[args.reader]
    calibration_path = args.calibration_report or Path(
        f"results/trusted_reference_v8_{args.reader}_calibration.json"
    )
    output_path = args.output or Path(f"results/trusted_reference_v8_{args.reader}_test.json")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    cases = read_cases(args.data / f"{args.split}.jsonl",
                       manifest["file_sha256"][f"{args.split}.jsonl"])
    if hashlib.sha256((model["path"] / model["weight"]).read_bytes()).hexdigest() != model["sha256"]:
        raise ValueError("Pinned QA model hash mismatch")
    reader = ReferenceReader(model["path"])
    started = time.perf_counter()
    extracted = reader.extract_batch([case["query"] for case in cases],
                                     [case["reference_text"] for case in cases])
    predictions = [dict(answer=row.answer, score=row.score, start=row.start, end=row.end)
                   for row in extracted]
    prediction_path = args.data / f"trusted_reference_v8_{args.reader}_{args.split}_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for case, prediction in zip(cases, predictions):
            handle.write(json.dumps({"query_id": case["query_id"],
                                     "reference_id": case["reference_id"],
                                     **prediction}, ensure_ascii=False) + "\n")
    if args.split == "calibration":
        threshold, metrics = choose_threshold(cases, predictions)
        path = calibration_path
    else:
        cal = json.loads(calibration_path.read_text(encoding="utf-8"))
        if cal["source_manifest_sha256"] != hashlib.sha256(
            (args.data / "manifest.json").read_bytes()
        ).hexdigest() or cal["model"] != model["name"]:
            raise ValueError("Calibration used a different source manifest")
        threshold = cal["min_reader_score"]
        metrics = outcomes(cases, predictions, threshold)
        path = output_path
    presence = {family: sum(case["reference_id"] in {
        doc["id"] for doc in case["retrievals"][family]["candidates"]
    } for case in cases) for family in ("clean", "naive", "corruptrag_ak")}
    report = {"split": args.split, "source": "benchmark qrel reference proxy",
              "source_manifest_sha256": hashlib.sha256(
                  (args.data / "manifest.json").read_bytes()
              ).hexdigest(), "model": model["name"],
              "model_weight_sha256": model["sha256"],
              "min_reader_score": threshold, "metrics": metrics,
              "reference_present_in_logged_top10": presence,
              "elapsed_seconds": round(time.perf_counter() - started, 2),
              "limitation": "Answer strings are used only for scoring; qrel reference is not a live source attestation."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
