"""Verify the V23 integrity gate preserves frozen clean RAG outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from detect.snapshot_integrity_detector_v22 import SnapshotIntegrityDetector
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import HybridRetriever


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="store_true",
                        help="Also enforce pinned document ID/hash identity")
    args = parser.parse_args()
    corpus_path = Path("data/benchmark/validation/corpus.jsonl")
    queries_path = Path("data/benchmark/validation/queries.jsonl")
    queries = [json.loads(line) for line in queries_path.open(encoding="utf-8") if line.strip()]
    detector = SnapshotIntegrityDetector(
        Path("data/benchmark_snapshot_v22"),
        expected_corpus_sha256=sha(corpus_path), containment=True,
        identity_corpus=corpus_path if args.identity else None)
    records = []
    try:
        with tempfile.TemporaryDirectory(prefix="sentinel-clean-v23-") as temporary:
            retriever = HybridRetriever(corpus_path, artifact_dir=Path(temporary) / "index",
                                        force_rebuild=True)
            for query in queries:
                candidates = retriever.retrieve(query["question"], top_k=5)
                baseline = _select_answer(query["question"], candidates, retriever)
                results = [detector.score_document(query["question"], doc["text"],
                                                   doc["doc_id"])
                           for doc in candidates]
                kept = [doc for doc, result in zip(candidates, results)
                        if result["decision"] != "quarantine"]
                defended = _select_answer(query["question"], kept, retriever)
                records.append({
                    "qid": query["qid"],
                    "retrieved": len(candidates),
                    "clean_quarantines": len(candidates) - len(kept),
                    "same_answer": baseline == defended,
                })
    finally:
        detector.close()
    output = Path("results/clean_snapshot_v24_identity.json" if args.identity
                  else "results/clean_snapshot_v23.json")
    cases = output.with_suffix(".cases.jsonl")
    with cases.open("w", encoding="utf-8", newline="\n") as sink:
        for row in records:
            sink.write(json.dumps(row) + "\n")
    report = {
        "status": "bounded clean MS MARCO validation replay with identity" if args.identity
                  else "bounded clean MS MARCO validation replay",
        "questions": len(records),
        "retrieved_clean_candidates": sum(row["retrieved"] for row in records),
        "clean_quarantines": sum(row["clean_quarantines"] for row in records),
        "same_answers": sum(row["same_answer"] for row in records),
        "corpus_sha256": sha(corpus_path),
        "snapshot_manifest_sha256": detector.manifest_sha256,
        "case_rows_sha256": sha(cases),
        "limits": [
            "This clean replay uses unmodified snapshot passages; approved updates were not tested.",
            "The local extractive answerer and 75 questions do not establish general utility.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "limits"}, indent=2))


if __name__ == "__main__":
    main()
