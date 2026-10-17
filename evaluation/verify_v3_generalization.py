"""Verify frozen v3 held-out generalization artifacts without rerunning inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from attack.heldout_families import FAMILY_TEMPLATES
from data.validate_benchmark import validate_benchmark
from evaluation.evaluate_v3_generalization import (
    BENCHMARK,
    PROTOCOL_PATH,
    ROOT,
    _hash_inputs,
    _read_jsonl,
    _sha256,
)


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Summary disagrees with case records: {label}")


def verify(root: Path = ROOT) -> dict:
    protocol_path = root / "research_protocols" / "v3_heldout_generalization.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("frozen_inputs") != _hash_inputs():
        raise ValueError("Frozen v3 protocol inputs have changed")
    for relative_path, digest in protocol.get("development_evidence", {}).items():
        if _sha256(root / relative_path) != digest:
            raise ValueError(f"Frozen validation development evidence changed: {relative_path}")
    if tuple(protocol.get("families", [])) != tuple(FAMILY_TEMPLATES):
        raise ValueError("Frozen attack-family list does not match the generator")
    validate_benchmark(BENCHMARK)

    manifest_path = root / "results" / "v3_generalization_test.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("Test manifest does not reference the frozen protocol")
    if manifest.get("frozen_inputs") != _hash_inputs():
        raise ValueError("Test manifest inputs differ from the frozen protocol")
    paths = {
        "summary": root / manifest["summary"],
        "cases": root / manifest["cases"],
        "clean_cases": root / manifest["clean_cases"],
    }
    for key, path in paths.items():
        if not path.is_file() or _sha256(path) != manifest["sha256"][key]:
            raise ValueError(f"Missing or changed frozen test {key}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("split") != "test" or summary.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("Summary split/protocol metadata is invalid")
    expected_qids = {str(row["qid"]) for row in _read_jsonl(BENCHMARK / "test" / "queries.jsonl")}
    clean_rows = _read_jsonl(paths["clean_cases"])
    if len(clean_rows) != len(expected_qids) or {str(row["qid"]) for row in clean_rows} != expected_qids:
        raise ValueError("Clean control cases do not exactly cover v3 test")
    if len({str(row["qid"]) for row in clean_rows}) != len(clean_rows):
        raise ValueError("Duplicate clean-control query IDs")
    cases = _read_jsonl(paths["cases"])
    if len(cases) != len(expected_qids) * len(FAMILY_TEMPLATES):
        raise ValueError("Held-out family case count is incorrect")
    if len({(str(row["qid"]), row["family"]) for row in cases}) != len(cases):
        raise ValueError("Duplicate query/family cases")
    if {row["family"] for row in cases} != set(FAMILY_TEMPLATES):
        raise ValueError("Case records do not cover the frozen family list")
    by_family = {family: [row for row in cases if row["family"] == family]
                 for family in FAMILY_TEMPLATES}
    for family, rows in by_family.items():
        if {str(row["qid"]) for row in rows} != expected_qids:
            raise ValueError(f"{family}: incomplete or unexpected test query IDs")
        for row in rows:
            before = row["undefended_source_doc_id"] == row["attack_doc_id"] and row["wrong_answer"].casefold() in row["undefended_answer"].casefold()
            after = row["detector_source_doc_id"] == row["attack_doc_id"] and row["wrong_answer"].casefold() in row["detector_answer"].casefold()
            if before != row["undefended_attack_success"] or after != row["defended_attack_success"]:
                raise ValueError(f"{family}/{row['qid']}: attack-success label is invalid")
            if row["integrity_only_attack_success"] != before:
                raise ValueError(f"{family}/{row['qid']}: accepted-ingest integrity-only control changed")
            if row["attack_integrity_status"] != ("verified" if row["attack_retrieved"] else "not_retrieved"):
                raise ValueError(f"{family}/{row['qid']}: integrity status is inconsistent")
            if bool(row["attack_quarantined"]) != any(
                item["doc_id"] == row["attack_doc_id"] and item["quarantined"]
                for item in row["detector_decisions"]
            ):
                raise ValueError(f"{family}/{row['qid']}: quarantine label is invalid")
            if row["clean_alias_recovered"] != any(
                alias.casefold() in row["detector_answer"].casefold()
                for alias in row["answer_aliases"]
            ):
                raise ValueError(f"{family}/{row['qid']}: answer-alias label is invalid")
        recorded = summary["paired_attack_families"][family]
        _close(recorded["attack_success_before_rate"], sum(row["undefended_attack_success"] for row in rows) / len(rows), family + " before rate")
        _close(recorded["attack_success_after_detector_rate"], sum(row["defended_attack_success"] for row in rows) / len(rows), family + " after rate")
        _close(recorded["attack_retrieval_rate"], sum(row["attack_retrieved"] for row in rows) / len(rows), family + " retrieval")
        _close(recorded["clean_alias_recovered_under_attack"], sum(row["clean_alias_recovered"] for row in rows) / len(rows), family + " clean recovery")
    before = [row["baseline_alias_match"] for row in clean_rows]
    after = [row["detector_alias_match"] for row in clean_rows]
    clean_summary = summary["clean_control"]
    _close(clean_summary["baseline_alias_match_rate"], sum(before) / len(before), "clean baseline")
    _close(clean_summary["detector_alias_match_rate"], sum(after) / len(after), "clean defended")
    return {
        "status": "verified",
        "split": "test",
        "queries": len(expected_qids),
        "heldout_families": len(FAMILY_TEMPLATES),
        "paired_cases": len(cases),
        "protocol_sha256": _sha256(protocol_path),
        "outputs_hash_checked": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
