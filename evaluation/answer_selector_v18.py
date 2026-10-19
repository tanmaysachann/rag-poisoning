"""Train and score a passage-answer selector without answer keys at inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from data.build_full_corpus_v17 import ROOT, normalized
from data.build_trusted_snapshot_v12 import digest
from evaluation.evaluate_full_corpus_v17 import answer_match, outcomes, split_rows
from evaluation.evaluate_trusted_snapshot_v12 import policy_prediction


V17_SPLIT = ROOT / "research_protocols" / "full_corpus_v17_matched_split.json"
V18_SPLIT = ROOT / "research_protocols" / "answer_selector_v18_split.json"
ANSWERS = ROOT / "data" / "full_corpus_v17" / "NQ-open.dev.jsonl"
V17_PREDICTIONS = ROOT / "data" / "full_corpus_v17" / "predictions.jsonl"
V18_PREDICTIONS = ROOT / "data" / "full_corpus_v18" / "predictions.jsonl"
ARTIFACT = ROOT / "artifacts" / "answer_selector_v18" / "selector.joblib"
TRAIN_REPORT = ROOT / "results" / "answer_selector_v18_training.json"
CALIBRATION_GATE_REPORT = ROOT / "results" / "answer_selector_v18_calibration_gate.json"
EVAL_REPORT = ROOT / "results" / "answer_selector_v18.json"
RANKINGS = ("lexical", "dense", "fusion")
FEATURES = (
    "qa_score", "lexical_reciprocal_rank", "dense_reciprocal_rank",
    "fusion_reciprocal_rank", "answer_log_words", "agreement_count",
    "agreement_score_sum", "agreement_best_score", "qa_gap_from_question_best",
)


def aliases_by_question(path: Path, expected_sha: str) -> dict[str, list[str]]:
    if digest(path, "sha256") != expected_sha:
        raise ValueError("Official answer file changed")
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[normalized(row["question"])].append(row["answer"])
    if any(len(value) != 1 for value in grouped.values()):
        raise ValueError("Ambiguous NQ-Open question text")
    return {key: value[0] for key, value in grouped.items()}


def candidates(case: dict) -> list[dict]:
    """Create one candidate per verified document; no IDs enter the features."""
    by_id = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for ranking in RANKINGS:
        for rank, row in enumerate(case["rankings"][ranking], 1):
            doc_id = row["doc_id"]
            ranks[doc_id][ranking] = rank
            if row["span_verified"] and row["answer"].strip():
                by_id[doc_id] = row
    if not by_id:
        return []
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in by_id.values():
        groups[normalized(row["answer"])].append(row)
    best_qa = max(float(row["qa_score"]) for row in by_id.values())
    result = []
    for doc_id, row in by_id.items():
        group = groups[normalized(row["answer"])]
        answer_words = len(normalized(row["answer"]).split())
        vector = [
            float(row["qa_score"]),
            *(1.0 / ranks[doc_id][method] if method in ranks[doc_id] else 0.0
              for method in RANKINGS),
            math.log1p(answer_words),
            float(len(group)),
            float(sum(float(item["qa_score"]) for item in group)),
            max(float(item["qa_score"]) for item in group),
            float(row["qa_score"]) - best_qa,
        ]
        result.append({"doc_id": doc_id, "answer": row["answer"],
                       "qa_score": float(row["qa_score"]), "features": vector})
    return result


def choose(case: dict, model) -> dict:
    options = candidates(case)
    if not options:
        return {"answer": "", "score": 0.0, "doc_ids": []}
    probabilities = model.predict_proba(np.asarray(
        [row["features"] for row in options], dtype=np.float32))[:, 1]
    index = max(range(len(options)), key=lambda i: (
        float(probabilities[i]), options[i]["qa_score"], -i))
    selected = options[index]
    return {"answer": selected["answer"], "score": float(probabilities[index]),
            "doc_ids": [selected["doc_id"]]}


def calibrate(rows: list[dict], predictions: dict[str, dict],
              answers: dict[str, list[str]]) -> tuple[float, dict]:
    cutoffs = {0.0, 1.0, *(predictions[row["qid"]]["score"] for row in rows)}
    cutoffs.add(math.nextafter(max(cutoffs), math.inf))
    options = []
    budget = math.floor(0.10 * len(rows))
    for cutoff in cutoffs:
        counts = outcomes(rows, predictions, answers, cutoff)
        if counts["nonmatching_emitted"] <= budget:
            options.append((counts["strict_match"], -counts["nonmatching_emitted"],
                            -cutoff, cutoff, counts))
    _, _, _, cutoff, counts = max(options)
    return cutoff, counts


def read_predictions(path: Path, expected_qids: set[str]) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["qid"] in result:
            raise ValueError("Duplicate prediction qid")
        result[row["qid"]] = row
    if set(result) != expected_qids:
        raise ValueError("Prediction qids differ from frozen split")
    return result


def train() -> dict:
    if ARTIFACT.exists() or TRAIN_REPORT.exists():
        raise FileExistsError("Refusing to overwrite v18 trained selector")
    v17, rows = split_rows(V17_SPLIT)
    v18, _ = split_rows(V18_SPLIT)
    if v18["prior_v17_split_sha256"] != digest(V17_SPLIT, "sha256"):
        raise ValueError("Fresh split does not exclude the frozen v17 split")
    aliases = aliases_by_question(ANSWERS, v17["nq_open_answers_sha256"])
    prediction_rows = read_predictions(V17_PREDICTIONS,
                                       {row["qid"] for row in rows})
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        ("v18-train:" + row["qid"]).encode("utf-8")).hexdigest())
    training, calibration = ordered[:500], ordered[500:]
    features = []
    labels = []
    for row in training:
        gold = aliases[normalized(row["question"])]
        for option in candidates(prediction_rows[row["qid"]]):
            features.append(option["features"])
            labels.append(int(answer_match(option["answer"], gold,
                                            substring=False)))
    if not labels or len(set(labels)) != 2:
        raise ValueError("Training set lacks positive or negative candidate labels")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=3.0, random_state=18)
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels))
    chosen = {row["qid"]: choose(prediction_rows[row["qid"]], model)
              for row in calibration}
    calibration_answers = {row["qid"]: aliases[normalized(row["question"])]
                           for row in calibration}
    cutoff, counts = calibrate(calibration, chosen, calibration_answers)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "cutoff": cutoff,
                 "training_qids": [row["qid"] for row in training],
                 "calibration_qids": [row["qid"] for row in calibration]}, ARTIFACT)
    result = {
        "protocol": "research_protocols/answer_selector_v18.md",
        "source_v17_split_sha256": digest(V17_SPLIT, "sha256"),
        "source_v17_predictions_sha256": digest(V17_PREDICTIONS, "sha256"),
        "fresh_v18_split_sha256": digest(V18_SPLIT, "sha256"),
        "model_sha256": digest(ARTIFACT, "sha256"),
        "training_questions": len(training), "calibration_questions": len(calibration),
        "training_candidates": len(labels), "positive_candidates": sum(labels),
        "features": FEATURES, "cutoff": cutoff, "calibration": counts,
        "gate_on_fresh_evaluation": {"strict_match_at_least": 150,
                                     "nonmatching_emitted_at_most": 50,
                                     "unverified_emissions": 0},
    }
    TRAIN_REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate(archive: Path) -> dict:
    if EVAL_REPORT.exists():
        raise FileExistsError("Refusing to overwrite fresh v18 evaluation")
    split, rows = split_rows(V18_SPLIT)
    if len(rows) != 500 or split["calibration"]:
        raise ValueError("Expected exactly 500 untouched v18 questions")
    saved = joblib.load(ARTIFACT)
    if tuple(saved["features"]) != FEATURES:
        raise ValueError("Saved model features differ from frozen code")
    training_report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    if digest(ARTIFACT, "sha256") != training_report["model_sha256"]:
        raise ValueError("Trained selector artifact changed")
    if training_report["fresh_v18_split_sha256"] != digest(V18_SPLIT, "sha256"):
        raise ValueError("Fresh v18 split changed after selector training")
    if digest(archive, "sha256") != split["beir_archive_sha256"]:
        raise ValueError("BEIR source archive changed")
    with ZipFile(archive) as source:
        qrels_bytes = source.read("nq/qrels/test.tsv")
    if hashlib.sha256(qrels_bytes).hexdigest() != split["beir_qrels_sha256"]:
        raise ValueError("BEIR relevance judgments changed")
    relevant = defaultdict(set)
    for row in csv.DictReader(io.StringIO(qrels_bytes.decode("utf-8")),
                              delimiter="\t"):
        if int(row["score"]) > 0:
            relevant[row["query-id"]].add(row["corpus-id"])
    prediction_rows = read_predictions(V18_PREDICTIONS,
                                       {row["qid"] for row in rows})
    aliases = aliases_by_question(ANSWERS, split["nq_open_answers_sha256"])
    answers = {row["qid"]: aliases[normalized(row["question"])] for row in rows}
    model = saved["model"]
    selected = {row["qid"]: choose(prediction_rows[row["qid"]], model)
                for row in rows}
    baseline = {row["qid"]: policy_prediction(
        prediction_rows[row["qid"]]["rankings"]["dense"], "consensus10")
        for row in rows}
    selected_counts = outcomes(rows, selected, answers, saved["cutoff"])
    baseline_cutoff = json.loads((ROOT / "results" / "full_corpus_v17.json").read_text(
        encoding="utf-8"))["calibrated_score_cutoff"]
    baseline_counts = outcomes(rows, baseline, answers, baseline_cutoff)
    retrieval = {ranking: {f"qrel_at_{k}": sum(bool(
        relevant[row["qid"]] & {item["doc_id"] for item in
                               prediction_rows[row["qid"]]["rankings"][ranking][:k]})
        for row in rows) for k in (1, 5, 10)} for ranking in RANKINGS}
    unverified = 0
    for row in rows:
        chosen = selected[row["qid"]]
        if chosen["answer"] and chosen["score"] >= saved["cutoff"]:
            if not any(item["doc_id"] in chosen["doc_ids"] and
                       item["answer"] == chosen["answer"] and item["span_verified"]
                       for ranking in RANKINGS
                       for item in prediction_rows[row["qid"]]["rankings"][ranking]):
                unverified += 1
    cases_path = EVAL_REPORT.with_suffix(".cases.jsonl")
    if cases_path.exists():
        raise FileExistsError("Refusing to overwrite fresh v18 case rows")
    with cases_path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            qid = row["qid"]
            new = selected[qid]
            old = baseline[qid]
            new_answer = new["answer"] if new["score"] >= saved["cutoff"] else ""
            old_answer = old["answer"] if old["score"] >= baseline_cutoff else ""
            out.write(json.dumps({"qid": qid, "question": row["question"],
                "answer_aliases": answers[qid], "new_answer": new_answer,
                "new_source_ids": new["doc_ids"] if new_answer else [],
                "new_strict_match": answer_match(new_answer, answers[qid],
                                                  substring=False),
                "baseline_answer": old_answer,
                "baseline_strict_match": answer_match(old_answer, answers[qid],
                                                       substring=False)},
                ensure_ascii=False) + "\n")
    gate_passed = (selected_counts["strict_match"] >= 150
                   and selected_counts["nonmatching_emitted"] <= 50
                   and selected_counts["strict_match"] > baseline_counts["strict_match"]
                   and unverified == 0)
    result = {
        "protocol": "research_protocols/answer_selector_v18.md",
        "split_sha256": digest(V18_SPLIT, "sha256"),
        "predictions_sha256": digest(V18_PREDICTIONS, "sha256"),
        "model_sha256": digest(ARTIFACT, "sha256"),
        "questions": 500, "corpus_passages": 2681468,
        "baseline_v17_frozen_policy": baseline_counts,
        "learned_v18_selector": selected_counts,
        "retrieval": retrieval,
        "unverified_emissions": unverified,
        "gate_passed": gate_passed,
        "case_rows_sha256": digest(cases_path, "sha256"),
        "prediction_timing": json.loads(V18_PREDICTIONS.with_suffix(
            ".meta.json").read_text(encoding="utf-8")),
        "limitations": ["Public benchmark clean QA, not an attack or deployment test",
                        "NQ exact aliases may miss valid equivalent answers",
                        "V17 spent questions trained this policy; v18 questions did not"],
    }
    EVAL_REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def calibration_gate() -> dict:
    if CALIBRATION_GATE_REPORT.exists():
        raise FileExistsError("Refusing to overwrite v18 calibration gate")
    v17, rows = split_rows(V17_SPLIT)
    by_qid = {row["qid"]: row for row in rows}
    saved = joblib.load(ARTIFACT)
    calibration = [by_qid[qid] for qid in saved["calibration_qids"]]
    prediction_rows = read_predictions(V17_PREDICTIONS, set(by_qid))
    aliases = aliases_by_question(ANSWERS, v17["nq_open_answers_sha256"])
    answers = {row["qid"]: aliases[normalized(row["question"])]
               for row in calibration}
    learned = {row["qid"]: choose(prediction_rows[row["qid"]], saved["model"])
               for row in calibration}
    baseline = {row["qid"]: policy_prediction(
        prediction_rows[row["qid"]]["rankings"]["dense"], "consensus10")
        for row in calibration}
    baseline_cutoff = json.loads((ROOT / "results" / "full_corpus_v17.json").read_text(
        encoding="utf-8"))["calibrated_score_cutoff"]
    learned_counts = outcomes(calibration, learned, answers, saved["cutoff"])
    baseline_counts = outcomes(calibration, baseline, answers, baseline_cutoff)
    passed = (learned_counts["strict_match"] > baseline_counts["strict_match"]
              and learned_counts["nonmatching_emitted"] <= 20)
    result = {"protocol": "research_protocols/answer_selector_v18.md",
              "model_sha256": digest(ARTIFACT, "sha256"),
              "calibration_questions": len(calibration),
              "baseline_v17_frozen": baseline_counts,
              "learned_v18": learned_counts,
              "proceed_to_fresh_evaluation": passed,
              "fresh_split_sha256": digest(V18_SPLIT, "sha256"),
              "fresh_split_predictions_created": V18_PREDICTIONS.exists()}
    CALIBRATION_GATE_REPORT.write_text(json.dumps(result, indent=2) + "\n",
                                       encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "calibration-gate", "evaluate"))
    parser.add_argument("--beir-archive", type=Path)
    args = parser.parse_args()
    if args.phase == "evaluate" and args.beir_archive is None:
        parser.error("evaluate requires --beir-archive")
    result = (train() if args.phase == "train" else
              calibration_gate() if args.phase == "calibration-gate" else
              evaluate(args.beir_archive))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
