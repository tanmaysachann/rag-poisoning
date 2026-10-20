"""Pin untouched MS MARCO train or validation passages for PPO testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = Path("data/benchmark") / args.split / "corpus.jsonl"
    output = args.output or Path(
        "data/benchmark_snapshot_v22" if args.split == "validation"
        else "data/benchmark_snapshot_v22_train")
    output.mkdir(parents=True, exist_ok=True)
    database = output / "passages_fts5.sqlite"
    manifest = output / "manifest.json"
    if database.exists() or manifest.exists():
        raise FileExistsError("Refusing to overwrite a pinned benchmark snapshot")
    connection = sqlite3.connect(database)
    count = 0
    try:
        connection.execute("CREATE VIRTUAL TABLE passages USING fts5("
                           "doc_id UNINDEXED, title, body, sha256 UNINDEXED, "
                           "tokenize='porter unicode61')")
        for line in source.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            body = row["text"]
            connection.execute(
                "INSERT INTO passages(doc_id,title,body,sha256) VALUES(?,?,?,?)",
                (str(row["doc_id"]), "", body,
                 hashlib.sha256(body.encode("utf-8")).hexdigest()))
            count += 1
        connection.commit()
    finally:
        connection.close()
    manifest.write_text(json.dumps({
        "source": f"frozen MS MARCO benchmark {args.split} corpus; not reviewed truth",
        "source_corpus_sha256": sha(source),
        "passages": count,
        "database_filename": database.name,
        "database_sha256": sha(database),
        "schema": "FTS5 doc_id,title,body,sha256; porter unicode61",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passages": count, "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
