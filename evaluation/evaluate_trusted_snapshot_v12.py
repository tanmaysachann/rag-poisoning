"""Qrel-free retrieval and extractive answer evaluation on PoisonArena NQ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch

from data.build_multifamily_v10 import v10_train_question
from evaluation.evaluate_poisonarena_v9 import verified_json
from generation.trusted_reference import ReferenceReader, answer_overlap
from retrieval.trusted_snapshot_v12 import TrustedSnapshotSearch


QA_PATH = Path("artifacts/vendor/roberta-base-squad2-nq")
QA_SHA256 = "b2f61c7b1710fcc6826fc0cc12dca8c59c0fbdadd82c05bb6ebff2028583fcf6"
POLICIES = ("top1", "max_qa5", "consensus10")


def policy_prediction(passages: list[dict], policy: str) -> dict:
    if not passages:
        return {"answer": "", "score": 0.0, "doc_ids": []}
    if policy == "top1":
        row = passages[0]
        return {"answer": row["answer"], "score": row["qa_score"],
                "doc_ids": [row["doc_id"]]}
    if policy == "max_qa5":
        row = max(passages[:5], key=lambda item: item["qa_score"])
        return {"answer": row["answer"], "score": row["qa_score"],
                "doc_ids": [row["doc_id"]]}
    if policy == "consensus10":
        groups = []
        for row in sorted(passages[:10], key=lambda item: -item["qa_score"]):
            matched = next((group for group in groups
                            if answer_overlap(group[0]["answer"], row["answer"])), None)
            if matched is None:
                groups.append([row])
            else:
                matched.append(row)
        eligible = [group for group in groups if len({row["doc_id"] for row in group}) >= 2]
        if eligible:
            group = max(eligible, key=lambda rows: sum(row["qa_score"] for row in rows))
            best = max(group, key=lambda row: row["qa_score"])
            return {"answer": best["answer"], "score": best["qa_score"],
                    "doc_ids": [row["doc_id"] for row in group]}
        return {"answer": "", "score": 0.0, "doc_ids": []}
    raise ValueError(f"Unknown policy: {policy}")


def outcomes(rows: list[dict], predictions: dict[str, dict], threshold: float,
             answers: dict[str, str]) -> dict:
    counts = Counter()
    for row in rows:
        prediction = predictions[row["qid"]]
        answer = prediction["answer"] if prediction["score"] >= threshold else ""
        if not answer:
            counts["abstained"] += 1
        elif answer_overlap(answer, answers[row["qid"]]):
            counts["correct_match"] += 1
        else:
            counts["wrong_emitted"] += 1
    return {"questions": len(rows), **{key: counts[key] for key in
            ("correct_match", "wrong_emitted", "abstained")}}


def calibrate(rows: list[dict], predictions: dict[str, dict],
              answers: dict[str, str]) -> tuple[float, dict]:
    budget = math.floor(len(rows) * 0.1)
    cutoffs = {0.0, 1.0, *(predictions[row["qid"]]["score"] for row in rows)}
    cutoffs.add(math.nextafter(max(cutoffs), math.inf))
    options = []
    for cutoff in cutoffs:
        summary = outcomes(rows, predictions, cutoff, answers)
        if summary["wrong_emitted"] <= budget:
            options.append((summary["correct_match"], -summary["wrong_emitted"],
                            -cutoff, cutoff, summary))
    if not options:
        raise ValueError("No calibration cutoff met the wrong-answer budget")
    _, _, _, cutoff, summary = max(options)
    return cutoff, summary


def load_questions(directory: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw_queries = (directory / "nq_queries.json").read_bytes()
    if hashlib.sha256(raw_queries).hexdigest() != manifest["files"]["nq_queries"]["sha256"]:
        raise ValueError("Question file changed")
    # The published arena data is used only to identify its 100 qids here;
    # answers and targets are deliberately not read until predictions are saved.
    arena_bytes = (directory / "nq_arena.json").read_bytes()
    if hashlib.sha256(arena_bytes).hexdigest() != manifest["files"]["nq_arena"]["sha256"]:
        raise ValueError("Arena file changed")
    qids = {match.decode("utf-8") for match in
            re.findall(rb'^    "(test[0-9]+)": \{$', arena_bytes, re.MULTILINE)}
    questions = [row for row in map(json.loads, raw_queries.splitlines())
                 if row["_id"] in qids]
    if len(questions) != 100 or {row["_id"] for row in questions} != qids:
        raise ValueError("Expected exactly 100 independent PoisonArena questions")
    return [{"qid": row["_id"], "question": row["text"],
             "split": "calibration" if v10_train_question(row["text"]) else "development"}
            for row in questions], manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/trusted_snapshot_v12"))
    parser.add_argument("--poisonarena", type=Path, default=Path("data/poisonarena_v9"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trusted_snapshot_v12.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an already scored evaluation")
    if hashlib.sha256((QA_PATH / "pytorch_model.bin").read_bytes()).hexdigest() != QA_SHA256:
        raise ValueError("Pinned QA weights changed")
    questions, source_manifest = load_questions(args.poisonarena)
    snapshot = TrustedSnapshotSearch(args.snapshot)
    try:
        cases = [{**row, "passages": snapshot.search(row["question"], limit=10)}
                 for row in questions]
    finally:
        snapshot.close()
    print("Retrieved frozen snapshot passages for 100 questions", flush=True)
    reader = ReferenceReader(QA_PATH)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    flat = [(case, passage) for case in cases for passage in case["passages"]]
    extracted = reader.extract_batch([case["question"] for case, _ in flat],
                                     [passage["text"] for _, passage in flat],
                                     batch_size=24)
    for (_, passage), result in zip(flat, extracted):
        passage["answer"] = result.answer.strip()
        passage["qa_score"] = result.score
        passage.pop("text")
    prediction_path = args.snapshot / "predictions.jsonl"
    if prediction_path.exists():
        raise FileExistsError("Refusing to overwrite prior answer predictions")
    with prediction_path.open("w", encoding="utf-8", newline="\n") as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("Saved answer predictions before opening qrels or answers", flush=True)
    arena = verified_json(args.poisonarena / "nq_arena.json",
                          source_manifest["files"]["nq_arena"]["sha256"])
    answers = {qid: row["correct answer"] for qid, row in arena.items()}
    calibration = [case for case in cases if case["split"] == "calibration"]
    development = [case for case in cases if case["split"] == "development"]
    if len(calibration) != 71 or len(development) != 29:
        raise ValueError("Unexpected development split")
    options = []
    for policy in POLICIES:
        predictions = {case["qid"]: policy_prediction(case["passages"], policy)
                       for case in cases}
        cutoff, counts = calibrate(calibration, predictions, answers)
        options.append((counts["correct_match"], -counts["wrong_emitted"],
                        -POLICIES.index(policy), policy, cutoff, counts))
    _, _, _, selected, cutoff, calibration_counts = max(options)
    predictions = {case["qid"]: policy_prediction(case["passages"], selected)
                   for case in cases}
    development_counts = outcomes(development, predictions, cutoff, answers)
    qrels_data = (args.poisonarena / "nq_qrels.json").read_bytes()
    if hashlib.sha256(qrels_data).hexdigest() != source_manifest["files"]["nq_qrels"]["sha256"]:
        raise ValueError("PoisonArena qrels changed")
    relevant = {case["qid"]: set() for case in cases}
    with io.StringIO(qrels_data.decode("utf-8")) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["query-id"] in relevant and int(row["score"]) > 0:
                relevant[row["query-id"]].add(row["corpus-id"])
    retrieval = {}
    for name, subset in (("calibration", calibration), ("development", development)):
        retrieval[name] = {f"qrel_at_{k}": sum(bool(relevant[case["qid"]] &
                {passage["doc_id"] for passage in case["passages"][:k]})
                for case in subset) for k in (1, 5, 10)}
    result_cases = []
    for case in cases:
        prediction = predictions[case["qid"]]
        emitted = prediction["answer"] if prediction["score"] >= cutoff else ""
        result_cases.append({"qid": case["qid"], "split": case["split"],
                             "question": case["question"], "answer": emitted,
                             "correct_answer": answers[case["qid"]],
                             "correct_strict_match": answer_overlap(emitted, answers[case["qid"]]),
                             "selected_doc_ids": prediction["doc_ids"],
                             "selected_score": prediction["score"],
                             "retrieved_doc_ids": [row["doc_id"] for row in case["passages"]]})
    cases_path = args.output.with_suffix(".cases.jsonl")
    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in result_cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"protocol": "research_protocols/trusted_snapshot_v12.md",
              "snapshot_manifest_sha256": hashlib.sha256(
                  (args.snapshot / "manifest.json").read_bytes()).hexdigest(),
              "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              "qa_weights_sha256": QA_SHA256,
              "poisonarena_manifest_sha256": hashlib.sha256(
                  (args.poisonarena / "manifest.json").read_bytes()).hexdigest(),
              "questions": len(cases), "calibration_questions": len(calibration),
              "development_questions": len(development),
              "calibration_policy_comparison": [
                  {"policy": option[3], "cutoff": option[4], **option[5]}
                  for option in options],
              "selected_policy": selected, "frozen_cutoff": cutoff,
              "calibration": calibration_counts, "development": development_counts,
              "retrieval": retrieval,
              "gate": {"development_correct_at_least": 9,
                       "development_wrong_at_most": 2,
                       "development_qrel_at_10_at_least": 9},
              "limitation": "Unreviewed BEIR snapshot, already inspected PoisonArena development questions; lexical exact-match scoring."}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
