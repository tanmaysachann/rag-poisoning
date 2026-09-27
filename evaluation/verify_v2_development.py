"""Verify saved v2 train/validation pair traces and their peer-group contract."""

from __future__ import annotations

import json
from pathlib import Path

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate


def verify(root: Path = ROOT_DIR) -> dict:
    benchmark = root / "data/benchmark_v2"
    validate(benchmark, root / "data/benchmark")
    queries = {
        split: {row["qid"]: row for row in read_jsonl(benchmark / split / "queries.jsonl")}
        for split in ("train", "validation")
    }
    documents = {
        split: {row["doc_id"]: row for row in read_jsonl(benchmark / split / "corpus.jsonl")}
        for split in ("train", "validation")
    }
    results = root / "results"
    calibration = json.loads((results / "paired_gate_v2_train_calibration.json").read_text(encoding="utf-8"))
    train_cases = read_jsonl(results / "paired_gate_v2_train_calibration.cases.jsonl")
    if (len(train_cases) != len(queries["train"]) or {row["qid"] for row in train_cases} != set(queries["train"])
            or calibration["peer_group_contract"] != "non-null source_query_id equality for retrieved IDs via corpus metadata"
            or calibration["missing_usable_pair_score"] != sum(row["answer_cosine"] is None for row in train_cases)
            or calibration["train_clean_rejected"] != sum(
                row["answer_cosine"] is None or row["answer_cosine"] < calibration["calibrated_threshold"]
                for row in train_cases
            )):
        raise ValueError("V2 train pair calibration disagrees with saved cases")
    for row in train_cases:
        peer = row["peer_doc_id"]
        if peer is not None and (row["source_doc_id"] is None
                or documents["train"][peer]["source_query_id"]
                != documents["train"][row["source_doc_id"]]["source_query_id"]):
            raise ValueError("V2 train peer differs from selected passage group")

    profiles = {}
    for name in ("doc_only", "full_context"):
        path = results / f"paired_gate_v2_validation_{name}.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        cases = read_jsonl(path.with_suffix(".cases.jsonl"))
        if (len(cases) != len(queries["validation"])
                or {row["qid"] for row in cases} != set(queries["validation"])
                or summary["peer_group_contract"] != calibration["peer_group_contract"]):
            raise ValueError(f"V2 {name} pair summary has missing questions or metadata")
        if name == "full_context" and summary["train_calibrated_cosine_threshold"] != calibration["calibrated_threshold"]:
            raise ValueError("V2 full-context threshold differs from training calibration")
        for row in cases:
            supports = queries["validation"][row["qid"]]["support_doc_ids"]
            for prefix in ("clean", "attacked"):
                peer = row[prefix]["peer_doc_id"]
                source = row[prefix]["source_doc_id"]
                if peer is not None and (source is None or
                        documents["validation"][peer]["source_query_id"]
                        != documents["validation"][source]["source_query_id"]):
                    raise ValueError(f"V2 {name} {prefix} peer differs from selected passage group: {row['qid']}")
                if row[f"{prefix}_peer_is_labeled_support"] != (peer in supports):
                    raise ValueError(f"V2 {name} {prefix} peer flag disagrees with ID")
        counts = {
            "clean_answer_alias_before_gate": sum(row["clean_alias_before_gate"] for row in cases),
            "clean_answer_alias_after_gate": sum(row["clean_alias_after_gate"] for row in cases),
            "attack_success_before_gate": sum(row["attack_success_before_gate"] for row in cases),
            "attack_success_after_gate": sum(row["attack_success_after_gate"] for row in cases),
            "clean_labeled_support_peer": sum(row["clean_peer_is_labeled_support"] for row in cases),
            "attacked_labeled_support_peer": sum(row["attacked_peer_is_labeled_support"] for row in cases),
        }
        if any(summary[key] != value for key, value in counts.items()):
            raise ValueError(f"V2 {name} summary disagrees with its cases")
        profiles[name] = counts
    return {"status": "verified", "train_cases": len(train_cases), "validation": profiles,
            "test_attacks_evaluated": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
