"""Cited-extractive baseline on the frozen 20-question clean development set.

Takes the frozen MiniLM sentence ranker's selected sentence as the answer and
attaches its source document as the citation, then validates every citation
with the same exact-span check the local model must pass. No model inference
runs; the comparison column reuses the saved Qwen outputs on identical qids.

This measures what citation formatting alone is worth. It does not make the
extractor a generator and says nothing about attacked inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark
from data.validate_multisupport_benchmark import read_jsonl
from generation.local_llm import _is_source_span


def _compute(root: Path = ROOT_DIR) -> tuple[dict, list[dict]]:
    validate_benchmark(root / "data/benchmark")
    benchmark = root / "data/benchmark"
    corpus = {row["doc_id"]: row for row in read_jsonl(benchmark / "validation/corpus.jsonl")}
    queries = {row["qid"]: row for row in read_jsonl(benchmark / "validation/queries.jsonl")}
    qwen_rows = read_jsonl(root / "results/local_generation_grounded_validation_20.cases.jsonl")
    ranker_rows = {row["qid"]: row for row in read_jsonl(
        root / "results/clean_answers_validation_minilm_ranker_strict.cases.jsonl")}
    if len(qwen_rows) != 20 or len({row["qid"] for row in qwen_rows}) != 20:
        raise ValueError("Expected the frozen 20-case clean Qwen development sample")
    cases = []
    for saved in qwen_rows:
        query = queries[saved["qid"]]
        ranker = ranker_rows[saved["qid"]]
        answer, source = ranker["answer"], ranker["source_doc_id"]
        if source is None:
            cited, valid = None, False
        else:
            cited = f"{answer} [DOC {source}]"
            valid = _is_source_span(answer, corpus[source]["text"])
            if not valid:
                raise ValueError(f"Ranker sentence is not a source span: {saved['qid']}")
        cases.append({
            "qid": saved["qid"],
            "answer": cited,
            "source_doc_id": source,
            "citation_valid": valid,
            "cited_support": source in query["support_doc_ids"] if source is not None else False,
            # Frozen stored boolean: recomputing with the current alias matcher
            # flips 2/20 cases (msmarco-331/335, long-prose aliases), so the
            # comparison stays on the same basis as the saved Qwen/repair runs.
            "alias_match": ranker["alias_match"] if source is not None else False,
            "qwen_cited_support": saved["source_is_support"],
            "qwen_alias_match": saved["alias_match"],
        })
    result = {
        "split": "validation", "cases": len(cases),
        "input": "results/local_generation_grounded_validation_20.cases.jsonl",
        "cited_extractive_valid_citations": sum(row["citation_valid"] for row in cases),
        "cited_extractive_cited_support": sum(row["cited_support"] for row in cases),
        "cited_extractive_alias_match": sum(row["alias_match"] for row in cases),
        "cited_extractive_abstentions": sum(row["source_doc_id"] is None for row in cases),
        "same_questions_qwen_cited_support": sum(row["qwen_cited_support"] for row in cases),
        "same_questions_qwen_alias_match": sum(row["qwen_alias_match"] for row in cases),
        "note": "Same 20 clean development questions as the Qwen diagnostic. Citation validity is exact-span containment, not factual entailment or attack robustness.",
    }
    return result, cases


def evaluate(root: Path = ROOT_DIR) -> dict:
    return _compute(root)[0]


def verify(root: Path = ROOT_DIR) -> dict:
    summary = json.loads((root / "results/cited_extractive_validation_20.json").read_text(encoding="utf-8"))
    cases = read_jsonl(root / "results/cited_extractive_validation_20.cases.jsonl")
    recomputed, recomputed_cases = _compute(root)
    if any(summary[key] != value for key, value in recomputed.items()):
        raise ValueError("Saved cited-extractive summary disagrees with recomputation")
    if len(cases) != 20 or [row["qid"] for row in cases] != [row["qid"] for row in recomputed_cases]:
        raise ValueError("Saved cited-extractive cases cover the wrong questions")
    if any(row != saved for row, saved in zip(recomputed_cases, cases)):
        raise ValueError("Saved cited-extractive cases disagree with recomputation")
    if {row["qid"] for row in cases} != {
            row["qid"] for row in read_jsonl(root / "results/local_generation_grounded_validation_20.cases.jsonl")}:
        raise ValueError("Saved cited-extractive cases cover the wrong questions")
    return {"status": "verified", "cases": len(cases), "test_attacks_evaluated": False}


def main() -> None:
    result, cases = _compute()
    output = ROOT_DIR / "results/cited_extractive_validation_20.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
