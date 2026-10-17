"""Verify frozen data, split separation, and local model/evidence hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.build_benchmark import _tokens
from data.build_benchmark_v3 import load_exclusions
from detect.pilot_data import read_split

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "data" / "detector_pilot_v4"
V5 = ROOT / "data" / "detector_confirmation_v5"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_dataset(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["file_sha256"].items():
        if _sha256(path / name) != digest:
            raise ValueError(f"Data file changed: {path / name}")
    splits = {name: read_split(path, name) for name in manifest["counts"]}
    for name, rows in splits.items():
        if len(rows) != manifest["counts"][name]:
            raise ValueError(f"Count mismatch: {path / name}")
    ms_ids = [row["source_dataset"] + ":" + str(row["source_query_id"])
              for name, rows in splits.items() if name != "domain_test" for row in rows]
    if len(set(ms_ids)) != len(ms_ids):
        raise ValueError(f"MS MARCO source ID repeats within {path}")
    return splits


def main() -> None:
    v4 = _verify_dataset(V4)
    v5 = _verify_dataset(V5)
    v4_ms = {row["source_query_id"] for name, rows in v4.items()
             if name != "domain_test" for row in rows}
    v5_ms = {row["source_query_id"] for name, rows in v5.items()
             if name != "domain_test" for row in rows}
    if v4_ms & v5_ms:
        raise ValueError("v4/v5 MS MARCO source IDs overlap")
    old_ids, old_passages, old_questions = load_exclusions([
        ROOT / "data" / "benchmark", ROOT / "data" / "benchmark_v2",
        ROOT / "data" / "benchmark_v3",
    ])
    if (v4_ms | v5_ms) & old_ids:
        raise ValueError("New MS MARCO source IDs overlap prior benchmarks")
    for collection in (v4, v5):
        for name, rows in collection.items():
            if name == "domain_test":
                continue
            if any(_tokens(row["text"]) in old_passages or
                   _tokens(row["question"]) in old_questions for row in rows):
                raise ValueError("New MS MARCO text overlaps prior benchmarks")
    v4_squad = {_sha_text(row["text"]) for row in v4["domain_test"]}
    v5_squad = {_sha_text(row["text"]) for row in v5["domain_test"]}
    if v4_squad & v5_squad:
        raise ValueError("v4/v5 SQuAD contexts overlap")
    training_path = ROOT / "results" / "detector_pilot_v4_training.json"
    confirmation_path = ROOT / "results" / "detector_confirmation_v5_test.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if training["data_manifest_sha256"] != _sha256(V4 / "manifest.json"):
        raise ValueError("Training report data hash mismatch")
    if confirmation["data_manifest_sha256"] != _sha256(V5 / "manifest.json"):
        raise ValueError("Confirmation report data hash mismatch")
    manifest = json.loads((ROOT / "research_protocols" / "combined_gate_v5.json").read_text(encoding="utf-8"))
    for name, item in manifest["inputs"].items():
        if _sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Gate input hash mismatch: {name}")
    if not confirmation["combined_adoption_gate"]["pass"]:
        raise ValueError("Combined gate failed confirmation")
    print(json.dumps({
        "v4_counts": {name: len(rows) for name, rows in v4.items()},
        "v5_counts": {name: len(rows) for name, rows in v5.items()},
        "ms_marco_source_overlap": 0, "prior_benchmark_overlap": 0,
        "squad_context_overlap": 0,
        "frozen_gate_inputs_verified": len(manifest["inputs"]),
        "combined_adoption_gate": "pass",
    }, indent=2))


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
