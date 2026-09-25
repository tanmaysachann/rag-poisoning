"""Stage document edits without rewriting the trusted corpus or its manifest."""

from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_document_attack(
    trusted_corpus: Path,
    staged_corpus: Path,
    *,
    doc_id: int,
    text: str,
    replace_existing: bool = False,
) -> None:
    """Create a separate corpus with one inserted or replaced document.

    The caller passes the *original* integrity path to the retriever. Calling
    ``write_integrity_manifest`` on this staged corpus would wrongly trust the
    attack, so this function never creates or changes a manifest.
    """
    if trusted_corpus.resolve() == staged_corpus.resolve():
        raise ValueError("The staged corpus must differ from the trusted corpus")
    if not text.strip():
        raise ValueError("Attack document text must not be empty")
    rows = _read_jsonl(trusted_corpus)
    matches = [index for index, row in enumerate(rows) if int(row["doc_id"]) == doc_id]
    if replace_existing:
        if len(matches) != 1:
            raise ValueError(f"Cannot replace missing or duplicate document ID {doc_id}")
        rows[matches[0]] = {**rows[matches[0]], "text": text}
    else:
        if matches:
            raise ValueError(f"Inserted document ID already exists: {doc_id}")
        rows.append({"doc_id": doc_id, "text": text})
    staged_corpus.parent.mkdir(parents=True, exist_ok=True)
    with staged_corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
