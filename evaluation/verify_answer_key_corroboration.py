"""Verify saved answer-key train calibration and validation probe."""

from __future__ import annotations

import json
from pathlib import Path

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate


def verify(root: Path = ROOT_DIR) -> dict:
    benchmark = root / "data/benchmark_v2"
    validate(benchmark, root / "data/benchmark")
    train_queries = {row["qid"]: row for row in read_jsonl(benchmark / "train/queries.jsonl")}
    validation_queries = {row["qid"]: row for row in read_jsonl(benchmark / "validation/queries.jsonl")}
    train_docs = {row["doc_id"]: row for row in read_jsonl(benchmark / "train/corpus.jsonl")}
    validation_docs = {row["doc_id"]: row for row in read_jsonl(benchmark / "validation/corpus.jsonl")}
    results = root / "results"
    calibration = json.loads((results / "answer_key_corroboration_v2_train_calibration.json").read_text(encoding="utf-8"))
    train_cases = read_jsonl(results / "answer_key_corroboration_v2_train_calibration.cases.jsonl")
    if len(train_cases) != len(train_queries) or {row["qid"] for row in train_cases} != set(train_queries):
        raise ValueError("Train answer-key cases do not cover the train split")
    if (calibration["split"] != "train" or calibration["cases"] != len(train_cases)
            or calibration["missing_peer_document"] != sum(row["peer_doc_id"] is None for row in train_cases)
            or calibration["train_clean_alias_match"] != sum(row["alias_match"] for row in train_cases)):
        raise ValueError("Train answer-key summary disagrees with its cases")
    for row in train_cases:
        peer, source = row["peer_doc_id"], row["source_doc_id"]
        if peer is not None and (source is None
                or train_docs[peer]["source_query_id"] != train_docs[source]["source_query_id"]):
            raise ValueError(f"Train peer differs from selected passage group: {row['qid']}")
    summary = json.loads((results / "answer_key_corroboration_v2_validation.json").read_text(encoding="utf-8"))
    cases = read_jsonl(results / "answer_key_corroboration_v2_validation.cases.jsonl")
    if len(cases) != len(validation_queries) or {row["qid"] for row in cases} != set(validation_queries):
        raise ValueError("Validation answer-key cases do not cover the validation split")
    if summary["train_selected_mode"] != calibration["selected_mode"]:
        raise ValueError("Validation mode differs from train calibration")
    for row in cases:
        for prefix in ("clean", "attacked"):
            peer, source = row[prefix]["peer_doc_id"], row[prefix]["source_doc_id"]
            if peer is not None and (source is None
                    or validation_docs[peer]["source_query_id"] != validation_docs[source]["source_query_id"]):
                raise ValueError(f"Validation {prefix} peer differs from group: {row['qid']}")
    counts = {
        "clean_answer_alias_before_gate": sum(row["clean_alias_before_gate"] for row in cases),
        "clean_answer_alias_after_gate": sum(row["clean_alias_after_gate"] for row in cases),
        "clean_answer_kept": sum(row["clean"]["gate_keeps_answer"] for row in cases),
        "attack_retrieved": sum(row["attack_retrieved"] for row in cases),
        "attack_success_before_gate": sum(row["attack_success_before_gate"] for row in cases),
        "attack_success_after_gate": sum(row["attack_success_after_gate"] for row in cases),
        "attacked_answer_kept": sum(row["attacked"]["gate_keeps_answer"] for row in cases),
    }
    if any(summary[key] != value for key, value in counts.items()):
        raise ValueError("Validation answer-key summary disagrees with its cases")
    return {"status": "verified", "train_cases": len(train_cases),
            "validation_cases": len(cases), "test_attacks_evaluated": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
