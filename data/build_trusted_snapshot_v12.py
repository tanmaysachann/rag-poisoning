"""Index every passage in the pinned BEIR NQ corpus for qrel-free retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from zipfile import ZipFile


BEIR_MD5 = "d4d3d2e48787a744b6f6e691ff534307"
MEMBER = "nq/corpus.jsonl"


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def build(archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    database = output / "beir_nq_fts5.sqlite"
    temporary = output / "beir_nq_fts5.building.sqlite"
    manifest_path = output / "manifest.json"
    if database.exists() or temporary.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a frozen or partial snapshot")
    if digest(archive, "md5") != BEIR_MD5:
        raise ValueError("BEIR NQ archive differs from pinned source")
    started = time.perf_counter()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("CREATE VIRTUAL TABLE passages USING fts5("
                           "doc_id UNINDEXED, title, body, sha256 UNINDEXED, "
                           "tokenize='porter unicode61')")
        count = 0
        batch = []
        with ZipFile(archive) as source, source.open(MEMBER) as corpus:
            for line in corpus:
                row = json.loads(line)
                doc_id = str(row["_id"])
                title = str(row.get("title") or "")
                body = str(row["text"])
                full_text = "\n".join(part for part in (title, body) if part)
                batch.append((doc_id, title, body,
                              hashlib.sha256(full_text.encode("utf-8")).hexdigest()))
                if len(batch) >= 5000:
                    connection.executemany(
                        "INSERT INTO passages(doc_id,title,body,sha256) VALUES(?,?,?,?)",
                        batch)
                    connection.commit()
                    count += len(batch)
                    batch.clear()
                    if count % 100000 == 0:
                        print(f"indexed {count:,} passages in "
                              f"{time.perf_counter()-started:.0f}s", flush=True)
            if batch:
                connection.executemany(
                    "INSERT INTO passages(doc_id,title,body,sha256) VALUES(?,?,?,?)", batch)
                connection.commit()
                count += len(batch)
        connection.execute("INSERT INTO passages(passages) VALUES('optimize')")
        connection.commit()
    finally:
        connection.close()
    manifest = {"protocol": "research_protocols/trusted_snapshot_v12.md",
                "source": "official BEIR NQ corpus, unreviewed research snapshot",
                "archive_md5": BEIR_MD5, "archive_sha256": digest(archive, "sha256"),
                "member": MEMBER, "passages": count,
                "database_sha256": digest(temporary, "sha256"),
                "database_bytes": temporary.stat().st_size,
                "index_seconds": round(time.perf_counter()-started, 2),
                "schema": "FTS5 doc_id,title,body,sha256; porter unicode61"}
    os.replace(temporary, database)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/trusted_snapshot_v12"))
    args = parser.parse_args()
    print(json.dumps(build(args.beir_archive, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
