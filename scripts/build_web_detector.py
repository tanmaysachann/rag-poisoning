"""Export the frozen hashing detector without its Windows-only retriever paths.

The training artifact embeds a full HybridRetriever whose cached paths are
WindowsPath objects. Scoring needs only its hashing encoder. This command
creates a separate portable serving artifact and leaves the sealed v1 model
unchanged.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detect.research_detector import ResearchDetector
from retrieval.hybrid_retriever import TextEmbedder


SOURCE = ROOT / "artifacts" / "research_detector_hashing.joblib"
OUTPUT = ROOT / "artifacts" / "research_detector_hashing_web.joblib"


def main() -> None:
    detector = joblib.load(SOURCE)
    if not isinstance(detector, ResearchDetector) or detector.embedding_model != "sklearn-hashing-384":
        raise ValueError("Expected the frozen hashing research detector")
    detector.embedder = TextEmbedder(preferred_backend="hashing")
    joblib.dump(detector, OUTPUT, compress=3)
    print(f"source_sha256={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}")
    print(f"web_sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
    print(f"web_bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
