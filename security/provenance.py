"""Operator-attested document origins and exact-span corroboration.

Origin identities come from a separately reviewed map, never from retrieved
document text. A signature binds each origin to its reviewed document hash.
This is an optional high-assurance policy, not a factual truth oracle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from security.integrity import document_hash


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold(), re.UNICODE)


def _contains_span(text: str, span: str) -> bool:
    haystack, needle = _tokens(text), _tokens(span)
    return bool(needle) and any(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


@dataclass(frozen=True)
class SourceAttestations:
    corpus_sha256: str
    documents: dict[str, dict[str, str]]

    def origin_for(self, doc: dict) -> str | None:
        row = self.documents.get(str(doc["doc_id"]))
        return row["origin_id"] if row and hmac.compare_digest(
            row["sha256"], document_hash(str(doc["text"]))
        ) else None


def seal_source_attestations(
    path: Path, documents: list[dict], origins: dict[str, str],
    corpus_sha256: str, *, signing_key: bytes, replace_existing: bool = False,
) -> None:
    """Seal only after an operator reviewed the corpus and separate origin map."""
    if not signing_key:
        raise ValueError("A signing key is required for source attestations")
    if not re.fullmatch(r"[0-9a-f]{64}", corpus_sha256):
        raise ValueError("Invalid reviewed corpus SHA-256")
    if path.exists() and not replace_existing:
        raise FileExistsError(f"Source attestation already exists: {path}")
    ids = [str(doc["doc_id"]) for doc in documents]
    if len(ids) != len(set(ids)) or set(ids) != set(origins):
        raise ValueError("Origin map must contain exactly one entry for every unique document ID")
    if any(not isinstance(origin, str) or not 1 <= len(origin.strip()) <= 120
           for origin in origins.values()):
        raise ValueError("Every origin ID must be a nonempty string of at most 120 characters")
    payload = {
        "schema_version": 1, "corpus_sha256": corpus_sha256,
        "documents": {
            doc_id: {"sha256": document_hash(str(doc["text"])), "origin_id": origins[doc_id].strip()}
            for doc_id, doc in zip(ids, documents)
        },
    }
    signature = hmac.new(signing_key, _canonical(payload), hashlib.sha256).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps({**payload, "signature": signature}, ensure_ascii=False,
                                    indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_source_attestations(path: Path, *, signing_key: bytes) -> SourceAttestations:
    if not signing_key:
        raise ValueError("A signing key is required to verify source attestations")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported source attestation schema")
    payload = {key: raw.get(key) for key in ("schema_version", "corpus_sha256", "documents")}
    signature = raw.get("signature")
    expected = hmac.new(signing_key, _canonical(payload), hashlib.sha256).hexdigest()
    if not isinstance(signature, str) or not hmac.compare_digest(signature, expected):
        raise ValueError("Source attestation signature mismatch")
    documents = payload["documents"]
    if (not re.fullmatch(r"[0-9a-f]{64}", str(payload["corpus_sha256"]))
            or not isinstance(documents, dict)
            or any(not isinstance(key, str) or not isinstance(row, dict)
                   or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
                   or not isinstance(row.get("origin_id"), str) or not row["origin_id"].strip()
                   for key, row in documents.items())):
        raise ValueError("Malformed source attestation entries")
    return SourceAttestations(payload["corpus_sha256"], documents)


def corroborate_exact_span(
    span: str, source_doc_id: int, accepted_documents: list[dict],
    attestations: SourceAttestations, *, minimum_origins: int = 2,
) -> dict:
    """Require the cited exact span in distinct, signed, accepted origins."""
    if minimum_origins < 2:
        raise ValueError("minimum_origins must be at least two")
    source = next((doc for doc in accepted_documents if int(doc["doc_id"]) == source_doc_id), None)
    if source is None or not _contains_span(source["text"], span):
        return {"supported": False, "reason": "source_span_unverified", "support_doc_ids": [], "origin_ids": []}
    source_origin = attestations.origin_for(source)
    if source_origin is None:
        return {"supported": False, "reason": "source_origin_unverified", "support_doc_ids": [], "origin_ids": []}
    by_origin = {source_origin: int(source_doc_id)}
    for doc in accepted_documents:
        origin = attestations.origin_for(doc)
        if origin and origin not in by_origin and _contains_span(doc["text"], span):
            by_origin[origin] = int(doc["doc_id"])
    supported = len(by_origin) >= minimum_origins
    return {
        "supported": supported,
        "reason": "distinct_origins_corroborate_exact_span" if supported else "insufficient_distinct_origin_support",
        "support_doc_ids": list(by_origin.values()), "origin_ids": list(by_origin),
        "required_origins": minimum_origins,
    }
