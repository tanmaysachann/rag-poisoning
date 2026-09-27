"""Preview and seal a separately reviewed source-origin map for a corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from security.provenance import seal_source_attestations


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="JSONL with unique doc_id and text")
    parser.add_argument("origins", type=Path, help="Separately reviewed JSON mapping doc IDs to origin IDs")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seal-reviewed-corpus-sha256")
    parser.add_argument("--seal-reviewed-origins-sha256")
    parser.add_argument("--replace-existing-manifest", action="store_true")
    args = parser.parse_args()
    documents = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    origins = json.loads(args.origins.read_text(encoding="utf-8"))
    if not isinstance(origins, dict):
        raise ValueError("Origin map must be a JSON object")
    corpus_sha = _sha256(args.corpus)
    origin_sha = _sha256(args.origins)
    ids = [str(doc["doc_id"]) for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Corpus contains duplicate document IDs")
    if any(not isinstance(value, str) or not value.strip() for value in origins.values()):
        raise ValueError("Each reviewed origin ID must be a nonempty string")
    preview = {
        "corpus_sha256": corpus_sha, "origins_sha256": origin_sha,
        "documents": len(documents), "distinct_origins": len(set(origins.values())),
        "missing_doc_ids": sorted(set(ids) - set(origins)),
        "extra_doc_ids": sorted(set(origins) - set(ids)),
        "manifest_exists": args.manifest.exists(),
    }
    requested = (args.seal_reviewed_corpus_sha256, args.seal_reviewed_origins_sha256)
    if any(requested) and not all(requested):
        raise ValueError("Both reviewed SHA-256 values are required to seal")
    if all(requested):
        if requested != (corpus_sha, origin_sha):
            raise ValueError("Corpus or origin map changed since operator review")
        key = os.getenv("RAG_MANIFEST_KEY", "").encode("utf-8")
        seal_source_attestations(
            args.manifest, documents, origins, corpus_sha, signing_key=key,
            replace_existing=args.replace_existing_manifest,
        )
        preview["sealed"] = True
    else:
        preview["sealed"] = False
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
