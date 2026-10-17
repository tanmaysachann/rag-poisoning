"""Run the frozen v5 combined content gate on a local JSONL corpus."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("RAG_USE_MINILM", "0")
os.environ.setdefault("RAG_USE_LLM", "0")

from detect.combined_gate import CombinedGate
from pipeline.research_inference import infer_research
from retrieval.hybrid_retriever import HybridRetriever
from security.integrity import load_runtime_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="JSONL documents with doc_id and text")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    gate = CombinedGate.from_manifest(ROOT / "research_protocols" / "combined_gate_v5.json")
    with tempfile.TemporaryDirectory(prefix="combined-gate-") as work:
        retriever = HybridRetriever(
            args.corpus, artifact_dir=Path(work) / "index", write_integrity_manifest=True
        )
        manifest = load_runtime_manifest(retriever.integrity_path)
        if manifest is None:
            raise RuntimeError("Could not seal the local corpus")
        result = infer_research(args.query, retriever, manifest, gate, top_k=args.top_k)
    print(json.dumps({
        "detector_profile": "v3_or_trained_minilm_v5",
        "attack_surface": "accepted_ingest_content_screening",
        "query": args.query, "undefended": result["undefended"],
        "defended": result["defended"],
        "documents": [{key: row[key] for key in (
            "rank", "doc_id", "risk", "threshold", "risk_score_kind",
            "detector_triggered_by", "decision", "reasons", "features"
        )} for row in result["documents"]],
        "timings": result["timings"],
    }, indent=2))


if __name__ == "__main__":
    main()
