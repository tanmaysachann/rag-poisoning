"""Versioned SHA-256 manifests with optional HMAC authentication.

An unsigned manifest is suitable only when its location is protected by the
operator. Set RAG_MANIFEST_KEY and RAG_REQUIRE_SIGNED_MANIFEST=1 when the
manifest itself could be modified by an attacker.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _signature(payload: dict, key: bytes) -> str:
    return hmac.new(key, _canonical_bytes(payload), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class IntegrityManifest:
    hashes: dict[str, str]
    corpus_sha256: str | None
    signature_verified: bool
    legacy: bool

    def check(self, doc_id: int | str, text: str) -> dict[str, str | None]:
        actual = document_hash(text)
        expected = self.hashes.get(str(doc_id))
        if expected is None:
            status = "unknown_document"
        elif not hmac.compare_digest(actual, expected):
            status = "tampered"
        else:
            status = "verified"
        return {"status": status, "expected_hash": expected, "actual_hash": actual}


def write_manifest(
    path: Path,
    documents: Iterable[dict],
    corpus_sha256: str,
    *,
    signing_key: bytes | None = None,
) -> None:
    """Seal an explicitly trusted ingestion batch; never call on untrusted data."""
    hashes: dict[str, str] = {}
    for doc in documents:
        doc_id = str(doc["doc_id"])
        if doc_id in hashes:
            raise ValueError(f"Duplicate document ID in trusted ingest: {doc_id}")
        hashes[doc_id] = document_hash(str(doc["text"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": corpus_sha256,
        "document_sha256": hashes,
    }
    result = {**payload, "signature": _signature(payload, signing_key) if signing_key else None}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_manifest(
    path: Path,
    *,
    signing_key: bytes | None = None,
    require_signature: bool = False,
) -> IntegrityManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Integrity manifest must be a JSON object")
    if "schema_version" not in raw:
        if require_signature:
            raise ValueError("Legacy unsigned manifest is not allowed")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in raw.items()):
            raise ValueError("Invalid legacy integrity manifest")
        return IntegrityManifest(raw, None, False, True)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported integrity manifest schema: {raw.get('schema_version')}")
    hashes = raw.get("document_sha256")
    corpus_sha256 = raw.get("corpus_sha256")
    if not isinstance(hashes, dict) or not isinstance(corpus_sha256, str):
        raise ValueError("Invalid integrity manifest fields")
    if not all(isinstance(key, str) and isinstance(value, str) and len(value) == 64 for key, value in hashes.items()):
        raise ValueError("Invalid document SHA-256 entries")
    signature = raw.get("signature")
    if require_signature and not signature:
        raise ValueError("Signed integrity manifest required")
    verified = False
    if signature:
        if not signing_key:
            raise ValueError("Manifest is signed but RAG_MANIFEST_KEY is unavailable")
        payload = {key: raw[key] for key in ("schema_version", "corpus_sha256", "document_sha256")}
        verified = hmac.compare_digest(str(signature), _signature(payload, signing_key))
        if not verified:
            raise ValueError("Integrity manifest signature mismatch")
    return IntegrityManifest(hashes, corpus_sha256, verified, False)


def load_runtime_manifest(path: Path) -> IntegrityManifest | None:
    if not path.is_file():
        return None
    raw_key = os.getenv("RAG_MANIFEST_KEY")
    require_signed = os.getenv("RAG_REQUIRE_SIGNED_MANIFEST", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    return load_manifest(
        path,
        signing_key=raw_key.encode("utf-8") if raw_key else None,
        require_signature=require_signed,
    )
