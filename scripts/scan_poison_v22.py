"""Run the trained local document scanner on a JSONL file or one candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detect.product_v22 import PoisonDocumentScanner


def main(*, default_integrity_mode: str = "near_copy") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL with query, reference, candidate")
    parser.add_argument("--query")
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path,
                        help="Optional separate pre-attack snapshot directory")
    parser.add_argument("--snapshot-manifest-sha256",
                        help="Required exact manifest hash with --snapshot-dir")
    parser.add_argument("--integrity-mode", choices=("near_copy", "containment"),
                        default=default_integrity_mode)
    parser.add_argument("--expected-sha256",
                        help="Pinned digest of one known document for exact identity checking")
    args = parser.parse_args()
    if args.input is None and not (args.query and args.reference_file and args.candidate_file):
        parser.error("Provide --input, or --query with both text files")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.no_snapshot and args.snapshot_dir:
        parser.error("Choose --no-snapshot or --snapshot-dir")
    if args.input is not None and args.expected_sha256:
        parser.error("Put expected_sha256 in each JSONL row instead")
    scanner = PoisonDocumentScanner(
        use_snapshot=not args.no_snapshot, snapshot_directory=args.snapshot_dir,
        snapshot_manifest_sha256=args.snapshot_manifest_sha256,
        integrity_mode=args.integrity_mode)
    try:
        if args.input is not None:
            with args.input.open(encoding="utf-8") as source:
                for index, line in enumerate(source):
                    if args.limit is not None and index >= args.limit:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    result = scanner.assess(
                        row["query"], row["reference"], row["candidate"],
                        expected_sha256=row.get("expected_sha256"))
                    metadata = {key: row[key] for key in ("query_id", "candidate_id", "family", "label")
                                if key in row}
                    print(json.dumps({**metadata, **result}, ensure_ascii=False), flush=True)
        else:
            result = scanner.assess(
                args.query, args.reference_file.read_text(encoding="utf-8"),
                args.candidate_file.read_text(encoding="utf-8"),
                expected_sha256=args.expected_sha256)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        scanner.close()


if __name__ == "__main__":
    main()
