"""PPO-compatible adapter for the frozen V22 changed-copy integrity rule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from detect.snapshot_near_duplicate import SnapshotNearDuplicate
from detect.snapshot_containment_v23 import SnapshotContainment


class SnapshotIntegrityDetector:
    """Use the same 0.90 V22 integrity threshold during document-edit PPO."""

    def __init__(self, directory: Path, *, expected_corpus_sha256: str,
                 threshold: float = 0.9, containment: bool = False,
                 identity_corpus: Path | None = None) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("Invalid integrity threshold")
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_corpus_sha256") != expected_corpus_sha256:
            raise ValueError("Snapshot does not belong to this episode corpus")
        self.threshold = threshold
        self.containment = containment
        self.identity_hashes = None
        if identity_corpus is not None:
            identity_corpus = Path(identity_corpus)
            if hashlib.sha256(identity_corpus.read_bytes()).hexdigest() != expected_corpus_sha256:
                raise ValueError("Identity corpus differs from pinned snapshot")
            with identity_corpus.open(encoding="utf-8") as source:
                self.identity_hashes = {
                    str(row["doc_id"]): hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                    for line in source if line.strip()
                    for row in [json.loads(line)]
                }
        self.scanner = (SnapshotContainment(self.directory) if containment
                        else SnapshotNearDuplicate(self.directory))

    def close(self) -> None:
        self.scanner.close()

    def score(self, query: str, doc_text: str) -> dict:
        if not query.strip() or not doc_text.strip():
            raise ValueError("Question and document are required")
        assessment = self.scanner.assess(doc_text)
        risk = assessment["near_copy_score"]
        return {
            "risk_score": risk,
            "threshold": self.threshold,
            "decision": "quarantine" if risk >= self.threshold else "accept",
            "integrity": assessment,
        }

    def score_document(self, query: str, doc_text: str, doc_id: str | int) -> dict:
        if self.identity_hashes is not None and str(doc_id) in self.identity_hashes:
            actual = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
            expected = self.identity_hashes[str(doc_id)]
            changed = actual != expected
            return {
                "risk_score": 1.0 if changed else 0.0,
                "threshold": self.threshold,
                "decision": "quarantine" if changed else "accept",
                "integrity": {"status": "known_document_hash_mismatch" if changed
                              else "known_document_hash_match", "doc_id": str(doc_id)},
            }
        return self.score(query, doc_text)
