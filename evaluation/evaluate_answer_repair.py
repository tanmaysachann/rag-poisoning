"""Replay saved Qwen clean-validation output through experimental span repair."""

from __future__ import annotations

import json
from pathlib import Path

from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark
from generation.answer_repair import repair_model_answer
from generation.local_llm import ABSTENTION, _is_source_span
from generation.sentence_ranker import _contains_alias


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    benchmark = ROOT_DIR / "data/benchmark"
    validate_benchmark(benchmark)
    source = ROOT_DIR / "results/local_generation_grounded_validation_20.cases.jsonl"
    saved = _jsonl(source)
    ranker_cases = {row["qid"]: row for row in _jsonl(
        ROOT_DIR / "results/clean_answers_validation_minilm_ranker_strict.cases.jsonl"
    )}
    corpus = {row["doc_id"]: row for row in _jsonl(benchmark / "validation/corpus.jsonl")}
    queries = {row["qid"]: row for row in _jsonl(benchmark / "validation/queries.jsonl")}
    if len(saved) != 20 or len({row["qid"] for row in saved}) != 20:
        raise ValueError("Expected the frozen 20-case clean Qwen development sample")
    cases = []
    for row in saved:
        query = queries[row["qid"]]
        if row["question"] != query["question"]:
            raise ValueError("Saved raw model output does not match the validation question")
        documents = [corpus[int(doc_id)] for doc_id in row["retrieved_doc_ids"]]
        answer, source_id, status = repair_model_answer(row["question"], row["raw_answer"], documents)
        span = answer.rsplit(" [DOC ", 1)[0] if source_id is not None else None
        if source_id is not None and not _is_source_span(span, corpus[source_id]["text"]):
            raise RuntimeError("Repair produced a span outside its cited source")
        cases.append({
            "qid": row["qid"], "raw_answer": row["raw_answer"],
            "answer": answer, "source_doc_id": source_id, "status": status,
            "cited_support": source_id in query["support_doc_ids"],
            "alias_match": _contains_alias(answer, query["answer_aliases"]) if source_id is not None else False,
            "baseline_cited_support": row["source_is_support"],
            "baseline_alias_match": row["alias_match"],
            "ranker_alias_match": ranker_cases[row["qid"]]["alias_match"],
        })
    result = {
        "split": "validation", "cases": len(cases),
        "input": str(source.relative_to(ROOT_DIR)).replace("\\", "/"),
        "baseline_cited_support": sum(row["baseline_cited_support"] for row in cases),
        "baseline_alias_match": sum(row["baseline_alias_match"] for row in cases),
        "same_questions_minilm_ranker_alias_match": sum(row["ranker_alias_match"] for row in cases),
        "repaired_cited_support": sum(row["cited_support"] for row in cases),
        "repaired_alias_match": sum(row["alias_match"] for row in cases),
        "repaired_abstentions": sum(row["answer"] == ABSTENTION for row in cases),
        "aligned_phrase_citations": sum(row["status"] == "aligned_phrase" for row in cases),
        "note": "Saved 20 clean development outputs only. Exact source-token alignment is not factual entailment or attack robustness.",
    }
    output = ROOT_DIR / "results/local_generation_answer_repair_validation_20.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
