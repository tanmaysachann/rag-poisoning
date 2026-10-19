"""Post-evaluation diagnosis of the spent v19 fresh split; no tuning here."""

from __future__ import annotations

import json
import statistics

from data.build_full_corpus_v17 import ROOT
from data.build_trusted_snapshot_v12 import digest
from evaluation.evaluate_full_corpus_v17 import answer_match


REPORT = ROOT / "results" / "answer_selector_v19.json"
CASES = ROOT / "results" / "answer_selector_v19.cases.jsonl"
SELECTED = ROOT / "data" / "full_corpus_v19" / "selected_predictions.jsonl"
OUTPUT = ROOT / "results" / "answer_selector_v19_diagnostics.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite post-evaluation diagnosis")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if (digest(CASES, "sha256") != report["case_rows_sha256"] or
            digest(SELECTED, "sha256") != report["selected_predictions_sha256"]):
        raise ValueError("Frozen evaluation inputs changed")
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines()]
    predictions = {row["qid"]: row for row in
                   (json.loads(line) for line in SELECTED.read_text(
                       encoding="utf-8").splitlines())}
    if len(cases) != 500 or set(predictions) != {row["qid"] for row in cases}:
        raise ValueError("Expected 500 aligned case rows")
    correct_scores, wrong_scores = [], []
    for case in cases:
        selected = predictions[case["qid"]]
        correct = answer_match(selected["answer"], case["answer_aliases"],
                               substring=False)
        (correct_scores if correct else wrong_scores).append(selected["score"])
    result = {
        "analysis_after_evaluation": True,
        "evaluation_sha256": digest(REPORT, "sha256"),
        "case_rows_sha256": digest(CASES, "sha256"),
        "selected_predictions_sha256": digest(SELECTED, "sha256"),
        "questions": len(cases),
        "reader_exact_candidate_questions": sum(row["reader_exact_candidate"]
                                                for row in cases),
        "unthresholded_model_top_candidate_exact": len(correct_scores),
        "calibrated_correct_emissions": sum(row["trained_strict_match"]
                                            for row in cases),
        "calibrated_wrong_emissions": sum(bool(row["trained_answer"] and
                                               not row["trained_strict_match"])
                                          for row in cases),
        "wrong_emissions_despite_exact_candidate": sum(bool(
            row["reader_exact_candidate"] and row["trained_answer"] and
            not row["trained_strict_match"]) for row in cases),
        "abstentions_despite_exact_candidate": sum(bool(
            row["reader_exact_candidate"] and not row["trained_answer"])
            for row in cases),
        "both_policies_correct": sum(row["trained_strict_match"] and
                                     row["baseline_strict_match"] for row in cases),
        "trained_only_correct": sum(row["trained_strict_match"] and
                                    not row["baseline_strict_match"] for row in cases),
        "baseline_only_correct": sum(row["baseline_strict_match"] and
                                     not row["trained_strict_match"] for row in cases),
        "top_candidate_score_median_if_exact": statistics.median(correct_scores),
        "top_candidate_score_median_if_nonmatching": statistics.median(wrong_scores),
        "limitation": "Oracle candidate presence and post-evaluation score distributions are diagnostics on a spent split, not an achievable accuracy claim or a tuning set.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
