"""Describe saved validation stealth failures without rerunning a defender."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import ROOT_DIR


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def independent_exact_span_counts(cases: list[dict], clean_cases: list[dict], corpus: list[dict]) -> dict:
    """Measure a generous full-corpus lexical ceiling for corroboration.

    Exact text in another document is only a proxy for independent support;
    it does not establish that the other document is trustworthy or independent.
    """
    documents = {row["doc_id"]: " ".join(row["text"].casefold().split()) for row in corpus}
    def elsewhere(answer: str, source_id: int | None) -> bool:
        span = " ".join(answer.casefold().split())
        return bool(span and not span.startswith("insufficient relevant evidence")
                    and any(span in text for doc_id, text in documents.items() if doc_id != source_id))

    return {
        "clean_selected_answer_in_other_corpus_document": sum(
            elsewhere(row["answer"], row["source_doc_id"]) for row in clean_cases
        ),
        "clean_selected_answer_nonabstentions": sum(row["source_doc_id"] is not None for row in clean_cases),
        "successful_stealth_answer_in_other_corpus_document": sum(
            elsewhere(row["defended_answer"], row["defended_source"])
            for row in cases if row["defended_attack_success"]
        ),
        "successful_stealth_answers": sum(row["defended_attack_success"] for row in cases),
        "note": "Exact whole-answer text in the full 75-passage validation corpus, excluding the selected source; this is not semantic or provenance verification.",
    }


def analyze(cases: list[dict], clean_cases: list[dict]) -> dict:
    clean = {row["qid"]: row for row in clean_cases}
    if len(clean) != len(clean_cases) or {row["qid"] for row in cases} != set(clean):
        raise ValueError("Stealth and clean cases must cover the same unique query IDs")
    if len({row["qid"] for row in cases}) != len(cases):
        raise ValueError("Duplicate stealth query ID")

    failures = [row for row in cases if row["defended_attack_success"]]
    prior_success = [row for row in cases if row["undefended_attack_success"]]
    counts = {
        "cases": len(cases),
        "attack_in_top5": sum(row["attack_top5_rank"] is not None for row in cases),
        "attack_quarantined": sum(row["attack_quarantined"] for row in cases),
        "undefended_attack_success": len(prior_success),
        "defended_attack_success": len(failures),
        "defended_success_from_attack_doc": sum(row["defended_source"] == row["attack_doc_id"] for row in failures),
        "defended_success_with_verified_integrity": sum(row["attack_integrity_status"] == "verified" for row in failures),
        "defended_success_with_changed_loo_answer": sum(row["loo_changed_answer"] is True for row in failures),
        "defended_success_with_clean_alias_match": sum(bool(clean[row["qid"]]["alias_match"]) for row in failures),
        "defended_success_answer_unchanged_from_undefended": sum(row["defended_answer"] == row["undefended_answer"] for row in failures),
    }
    rank_counts = dict(sorted(Counter(str(row["attack_top5_rank"]) for row in failures).items()))
    examples = [{
        "qid": row["qid"],
        "question": clean[row["qid"]]["question"],
        "attack_rank": row["attack_top5_rank"],
        "clean_answer": clean[row["qid"]]["answer"],
        "defended_answer": row["defended_answer"],
        "loo_changed_answer": row["loo_changed_answer"],
    } for row in failures[:8]]
    return {
        "split": "validation", "profile": "minilm_ranker", "attack_family": "stealth",
        "counts": counts, "failure_attack_rank_counts": rank_counts,
        "examples": examples,
        "interpretation": (
            "The attack replaced content before a new trusted snapshot. Verified integrity "
            "therefore proves snapshot parity, not factual truth. A changed leave-one-out "
            "answer shows dependency on a document, not whether that document is malicious."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT_DIR / "results/defense_validation_stealth_accepted_ingest_minilm_ranker.cases.jsonl")
    parser.add_argument("--clean", type=Path, default=ROOT_DIR / "results/clean_answers_validation_minilm_ranker_strict.cases.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT_DIR / "data/benchmark/validation/corpus.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "results/stealth_failure_analysis_validation.json")
    args = parser.parse_args()
    cases, clean_cases = read_jsonl(args.cases), read_jsonl(args.clean)
    report = analyze(cases, clean_cases)
    report["corroboration_coverage"] = independent_exact_span_counts(
        cases, clean_cases, read_jsonl(args.corpus)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "corroboration_coverage": report["corroboration_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
