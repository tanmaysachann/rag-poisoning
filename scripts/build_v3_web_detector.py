"""Create and verify the portable Linux serving export of the frozen v3 detector."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detect.research_detector import ResearchDetector
from retrieval.hybrid_retriever import TextEmbedder

SOURCE = ROOT / "artifacts" / "research_detector_v3_hashing.joblib"
OUTPUT = ROOT / "artifacts" / "research_detector_v3_hashing_web.joblib"
MANIFEST = ROOT / "research_protocols" / "v3_web_detector.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frozen = joblib.load(SOURCE)
    if not isinstance(frozen, ResearchDetector) or frozen.embedding_model != "sklearn-hashing-384":
        raise ValueError("Expected the frozen v3 hashing detector")

    # Replace the training retriever, which contains local absolute paths, with
    # the same deterministic hashing encoder used by the research inference path.
    portable_source = joblib.load(SOURCE)
    portable_source.embedder = TextEmbedder(preferred_backend="hashing")
    joblib.dump(portable_source, OUTPUT, compress=3)
    portable = joblib.load(OUTPUT)
    if not isinstance(portable, ResearchDetector) or portable.embedding_model != "sklearn-hashing-384":
        raise ValueError("Portable export failed its type/backend check")
    if portable.threshold != frozen.threshold:
        raise ValueError("Portable export changed the frozen threshold")

    cases = []
    validation = ROOT / "data" / "benchmark_v3" / "validation" / "queries.jsonl"
    for strategy in ("greedy", "stealth"):
        attack_path = ROOT / "results" / f"v3_attack_validation_{strategy}.jsonl"
        attacks = {row["qid"]: row for row in (
            json.loads(line) for line in attack_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )}
        for line in validation.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            query = json.loads(line)
            attack = attacks[query["qid"]]
            original_score = frozen.score(query["question"], attack["text"])["risk_score"]
            exported_score = portable.score(query["question"], attack["text"])["risk_score"]
            original_decision = original_score >= frozen.threshold
            exported_decision = exported_score >= portable.threshold
            if not np.isclose(original_score, exported_score, rtol=0.0, atol=1e-6):
                raise ValueError(f"Portable export score drift at {strategy}/{query['qid']}")
            if original_decision != exported_decision:
                raise ValueError(f"Portable export decision drift at {strategy}/{query['qid']}")
            cases.append((strategy, query["qid"]))

    manifest = {
        "schema_version": 1,
        "source_artifact": "artifacts/research_detector_v3_hashing.joblib",
        "source_sha256": _sha256(SOURCE),
        "serving_artifact": "artifacts/research_detector_v3_hashing_web.joblib",
        "serving_sha256": _sha256(OUTPUT),
        "embedding_backend": "sklearn-hashing-384",
        "threshold": portable.threshold,
        "validation_score_parity_cases": len(cases),
        "validation_score_parity_atol": 1e-6,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
