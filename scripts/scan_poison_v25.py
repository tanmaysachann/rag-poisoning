"""Run the frozen V25 reference-adaptive scanner on local passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend import v25_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        help="JSONL with query, reference, and candidate")
    parser.add_argument("--query")
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--expected-sha256",
                        help="Only use an independently pinned known-document digest")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.input is None and not all((args.query, args.reference_file,
                                       args.candidate_file)):
        parser.error("Provide --input or --query with both passage files")
    if args.input is not None and args.expected_sha256:
        parser.error("Put expected_sha256 in each JSONL row instead")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.input is not None:
        with args.input.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                if args.limit is not None and index >= args.limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                result = v25_service.scan(
                    row["query"], row["reference"], row["candidate"],
                    row.get("expected_sha256"))
                metadata = {key: row[key] for key in
                            ("query_id", "candidate_id", "family", "label")
                            if key in row}
                print(json.dumps({**metadata, **result}, ensure_ascii=False),
                      flush=True)
    else:
        result = v25_service.scan(
            args.query,
            args.reference_file.read_text(encoding="utf-8"),
            args.candidate_file.read_text(encoding="utf-8"),
            args.expected_sha256)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
