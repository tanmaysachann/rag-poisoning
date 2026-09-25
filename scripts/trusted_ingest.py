"""Preview a corpus and explicitly seal its reviewed digest as trusted."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from security.ingest import inspect_ingest, seal_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seal-reviewed-sha256", help="Seal only if corpus bytes still match this reviewed digest")
    parser.add_argument("--replace-existing-manifest", action="store_true")
    args = parser.parse_args()
    raw_key = os.getenv("RAG_MANIFEST_KEY")
    signing_key = raw_key.encode("utf-8") if raw_key else None
    if args.seal_reviewed_sha256:
        result = seal_ingest(
            args.corpus, args.manifest,
            expected_corpus_sha256=args.seal_reviewed_sha256,
            signing_key=signing_key,
            replace_existing_manifest=args.replace_existing_manifest,
        )
    else:
        result = inspect_ingest(args.corpus, args.manifest, signing_key=signing_key)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
