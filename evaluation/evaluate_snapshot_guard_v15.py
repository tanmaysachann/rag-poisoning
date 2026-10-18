"""Replay v3 validation attacks against the added pinned-snapshot answer guard."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from backend.research_lab import ROOT, VALIDATION, _detector, _inputs, run_case
from pipeline.research_inference import infer_research
from retrieval.hybrid_retriever import HybridRetriever
from security.integrity import load_runtime_manifest, write_manifest


def main() -> None:
    output = ROOT / "results" / "snapshot_guard_v15.json"
    case_path = ROOT / "data" / "snapshot_guard_v15" / "cases.jsonl"
    if output.exists() or case_path.exists():
        raise FileExistsError("Refusing to overwrite the v15 development replay")
    queries, corpus, _, _ = _inputs()
    expected = json.loads((ROOT / "data" / "benchmark_v3" / "manifest.json").read_text(
        encoding="utf-8"))["file_sha256"]["validation/corpus.jsonl"]
    corpus_path = VALIDATION / "corpus.jsonl"
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Pinned clean corpus changed")
    started = time.perf_counter()
    clean = Counter()
    with tempfile.TemporaryDirectory(prefix="sentinel-v15-clean-") as temp:
        temp_path = Path(temp)
        manifest_path = temp_path / "snapshot.json"
        key = os.getenv("RAG_MANIFEST_KEY")
        write_manifest(manifest_path, corpus.values(), expected,
                       signing_key=key.encode("utf-8") if key else None)
        manifest = load_runtime_manifest(manifest_path)
        if manifest is None:
            raise RuntimeError("Clean snapshot manifest unavailable")
        retriever = HybridRetriever(corpus_path, artifact_dir=temp_path / "index",
                                    integrity_path=manifest_path)
        detector = _detector()
        for qid, row in sorted(queries.items()):
            result = infer_research(row["question"], retriever, manifest, detector,
                                    pinned_snapshot=manifest)
            baseline = result["defended"]
            guarded = result["snapshot_guarded"]
            clean["questions"] += 1
            clean["v3_alias_matches"] += any(
                alias.casefold() in baseline["answer"].casefold() for alias in row["answer_aliases"])
            clean["guard_alias_matches"] += any(
                alias.casefold() in guarded["answer"].casefold() for alias in row["answer_aliases"])
            clean["v3_abstentions"] += baseline["source_doc_id"] is None
            clean["guard_abstentions"] += guarded["source_doc_id"] is None
            clean["guard_unverified_emissions"] += (
                guarded["source_doc_id"] is not None
                and result["snapshot_guard"]["decision"] != "allow")
    aggregate = defaultdict(Counter)
    case_path.parent.mkdir(parents=True, exist_ok=True)
    with case_path.open("w", encoding="utf-8", newline="\n") as sink:
        for strategy in ("greedy", "stealth"):
            for surface in ("accepted_ingest", "post_index_tamper"):
                name = f"{strategy}:{surface}"
                for index, qid in enumerate(sorted(queries), 1):
                    result = run_case(qid, strategy, surface, snapshot_recovery=False)
                    trace = result["snapshot_guard"]
                    row = {
                        "qid": qid, "strategy": strategy, "surface": surface,
                        "attack_retrieved": result["attack_retrieved"],
                        "v3_attack_source_success": result["defended_attack_success"],
                        "guard_attack_source_success": result["snapshot_guarded_attack_success"],
                        "guard_target_answer_match": result["snapshot_guarded_target_match"],
                        "v3_alias_match": result["clean_alias_recovered"],
                        "guard_alias_match": result["snapshot_guarded_clean_alias_recovered"],
                        "guard_abstained": result["snapshot_guarded"]["source_doc_id"] is None,
                        "guard_decision": trace["decision"], "guard_reason": trace["reason"],
                        "guard_source_status": trace["source_status"],
                    }
                    sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                    totals = aggregate[name]
                    totals["cases"] += 1
                    for field in ("attack_retrieved", "v3_attack_source_success",
                                  "guard_attack_source_success", "guard_target_answer_match",
                                  "v3_alias_match", "guard_alias_match", "guard_abstained"):
                        totals[field] += int(row[field])
                    totals["guard_verified_source_emissions"] += trace["decision"] == "allow"
                    totals["guard_unverified_source_emissions"] += (
                        trace["decision"] != "allow" and not row["guard_abstained"])
                    totals["guard_added_blocks_of_v3_attack_sources"] += (
                        row["v3_attack_source_success"] and row["guard_abstained"])
                    if index % 25 == 0:
                        print(f"v15: {name} {index}/{len(queries)}", flush=True)
    report = {
        "protocol": "research_protocols/snapshot_guard_v15.md",
        "split": "previously_inspected_v3_validation", "snapshot_corpus_sha256": expected,
        "clean": dict(clean),
        "attack_replays": {name: dict(values) for name, values in aggregate.items()},
        "case_rows_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "limitations": [
            "This snapshot is a pre-attack benchmark corpus, not an independently reviewed factual authority.",
            "The attacker cannot change the pinned snapshot or server code in this replay.",
            "Validation aggregates and questions were inspected before v15; no fresh test claim.",
            "An abstention blocks a sourced attack answer but does not recover the correct answer.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
