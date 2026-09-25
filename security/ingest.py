"""Operator-reviewed trusted ingest; attacker staging never calls this module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from security.integrity import document_hash, load_manifest, write_manifest


def _read_corpus(path: Path) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = [str(row["doc_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Trusted ingest requires a nonempty corpus with unique document IDs")
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        raise ValueError("Every ingested document needs nonempty text")
    return raw, rows


def inspect_ingest(corpus_path: Path, manifest_path: Path, *, signing_key: bytes | None = None) -> dict:
    raw, docs = _read_corpus(corpus_path)
    hashes = {str(doc["doc_id"]): document_hash(doc["text"]) for doc in docs}
    prior = load_manifest(manifest_path, signing_key=signing_key) if manifest_path.is_file() else None
    old = prior.hashes if prior else {}
    return {
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "documents": len(docs),
        "new_doc_ids": sorted(hashes.keys() - old.keys()),
        "changed_doc_ids": sorted(key for key in hashes.keys() & old.keys() if hashes[key] != old[key]),
        "missing_doc_ids": sorted(old.keys() - hashes.keys()),
        "manifest_exists": prior is not None,
        "prior_signature_verified": prior.signature_verified if prior else None,
    }


def seal_ingest(
    corpus_path: Path, manifest_path: Path, *, expected_corpus_sha256: str,
    signing_key: bytes | None = None, replace_existing_manifest: bool = False,
) -> dict:
    if corpus_path.resolve() == manifest_path.resolve():
        raise ValueError("The manifest must be separate from the corpus")
    if manifest_path.exists() and not replace_existing_manifest:
        raise FileExistsError("Existing manifest requires replace_existing_manifest=True")
    raw, docs = _read_corpus(corpus_path)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_corpus_sha256:
        raise ValueError("Corpus changed since review; digest does not match")
    write_manifest(manifest_path, docs, actual, signing_key=signing_key)
    return {"corpus_sha256": actual, "documents": len(docs),
            "manifest": str(manifest_path), "signed": signing_key is not None}
