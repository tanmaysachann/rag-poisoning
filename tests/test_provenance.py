"""Reviewed origin labels must be signed, hash-bound, and genuinely distinct."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.research_inference import infer_research
from security.integrity import IntegrityManifest, document_hash
from security.provenance import (
    corroborate_exact_span, load_source_attestations, seal_source_attestations,
)


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            {"doc_id": 1, "text": "The tower is in Paris, France.", "bm25_rank": 1, "dense_rank": 1, "score": .03},
            {"doc_id": 2, "text": "Paris, France is the tower's location.", "bm25_rank": 2, "dense_rank": 2, "score": .02},
            {"doc_id": 3, "text": "The tower is in London.", "bm25_rank": 3, "dense_rank": 3, "score": .01},
        ]

    def _seal(self, path: Path, origins: dict[str, str]):
        digest = hashlib.sha256(json.dumps(self.docs).encode()).hexdigest()
        seal_source_attestations(path, self.docs, origins, digest, signing_key=b"operator-key")
        return load_source_attestations(path, signing_key=b"operator-key")

    def _infer_strict(self, attestations=None):
        retriever = SimpleNamespace(
            documents=self.docs,
            embedder=SimpleNamespace(backend="hashing", model_name="sklearn-hashing-384"),
            retrieve=lambda _query, top_k: self.docs[:top_k],
        )
        manifest = IntegrityManifest(
            {str(doc["doc_id"]): document_hash(doc["text"]) for doc in self.docs},
            None, False, False,
        )
        detector = SimpleNamespace(score=lambda _query, _text: {
            "risk_score": 0.0, "threshold": 0.5, "decision": "accept", "features": {},
        })

        def answer(_query, documents, _retriever):
            if any(doc["doc_id"] == 1 for doc in documents):
                return {"answer": "Paris, France", "source_doc_id": 1,
                        "evidence_span": "Paris, France", "citations": [{"doc_id": 1, "span": "Paris, France"}],
                        "backend": "hashing_extractive", "generation_status": "extractive_selected"}
            return {"answer": "Insufficient evidence", "source_doc_id": None,
                    "evidence_span": None, "citations": [], "backend": "hashing_extractive",
                    "generation_status": "extractive_abstained"}

        with patch("pipeline.research_inference._answer", side_effect=answer):
            return infer_research("Where is the tower?", retriever, manifest, detector,
                                  source_attestations=attestations,
                                  require_independent_origins=True)

    def test_distinct_origin_exact_span_and_tamper_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "origins.json"
            attestations = self._seal(manifest, {"1": "publisher-a", "2": "publisher-b", "3": "publisher-c"})
            accepted = corroborate_exact_span("Paris, France", 1, self.docs, attestations)
            self.assertTrue(accepted["supported"])
            self.assertEqual(accepted["support_doc_ids"], [1, 2])
            self.assertFalse(corroborate_exact_span("London", 1, self.docs, attestations)["supported"])
            altered = [{**doc, "text": "The tower is in London."} if doc["doc_id"] == 2 else doc
                       for doc in self.docs]
            self.assertFalse(corroborate_exact_span("Paris, France", 1, altered, attestations)["supported"])
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["documents"]["2"]["origin_id"] = "forged-origin"
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                load_source_attestations(manifest, signing_key=b"operator-key")

    def test_two_documents_from_one_origin_do_not_count_as_corroboration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attestations = self._seal(
                Path(directory) / "origins.json",
                {"1": "same-publisher", "2": "same-publisher", "3": "other"},
            )
            result = corroborate_exact_span("Paris, France", 1, self.docs, attestations)
            self.assertFalse(result["supported"])
            self.assertEqual(result["reason"], "insufficient_distinct_origin_support")

    def test_strict_inference_abstains_without_source_attestations(self) -> None:
        result = self._infer_strict()
        self.assertEqual(result["defended"]["source_doc_id"], None)
        self.assertEqual(result["defended"]["citations"], [])
        self.assertEqual(result["provenance"]["reason"], "source_attestation_missing")

    def test_strict_inference_accepts_only_two_signed_origins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "origins.json"
            independent = self._seal(path, {"1": "publisher-a", "2": "publisher-b", "3": "publisher-c"})
            accepted = self._infer_strict(independent)
            self.assertEqual(accepted["defended"]["source_doc_id"], 1)
            self.assertEqual(accepted["provenance"]["support_doc_ids"], [1, 2])
            self.assertTrue(accepted["audit"][3]["supported"])
            removed_second_origin = next(row for row in accepted["counterfactuals"]
                                         if row["removed_doc_id"] == 2)
            self.assertTrue(removed_second_origin["answer_changed"])
            self.assertIsNone(removed_second_origin["source_after_removal"])

            path.unlink()
            duplicate_origin = self._seal(path, {"1": "publisher-a", "2": "publisher-a", "3": "publisher-c"})
            abstained = self._infer_strict(duplicate_origin)
            self.assertEqual(abstained["defended"]["source_doc_id"], None)
            self.assertEqual(abstained["provenance"]["reason"], "insufficient_distinct_origin_support")

    def test_operator_cli_requires_both_reviewed_digests(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            corpus, origins, manifest = (folder / name for name in ("corpus.jsonl", "origins.json", "signed.json"))
            corpus.write_text("".join(json.dumps({"doc_id": doc["doc_id"], "text": doc["text"]}) + "\n"
                                      for doc in self.docs), encoding="utf-8")
            origins.write_text(json.dumps({"1": "publisher-a", "2": "publisher-b", "3": "publisher-c"}),
                               encoding="utf-8")
            command = [sys.executable, str(root / "scripts/trusted_sources.py"),
                       str(corpus), str(origins), str(manifest)]
            preview = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
            hashes = json.loads(preview.stdout)
            self.assertFalse(hashes["sealed"])
            self.assertFalse(manifest.exists())
            env = {**os.environ, "RAG_MANIFEST_KEY": "operator-key"}
            sealed = subprocess.run(command + [
                "--seal-reviewed-corpus-sha256", hashes["corpus_sha256"],
                "--seal-reviewed-origins-sha256", hashes["origins_sha256"],
            ], cwd=root, env=env, capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(sealed.stdout)["sealed"])
            self.assertEqual(len(load_source_attestations(manifest, signing_key=b"operator-key").documents), 3)
            origins.write_text(json.dumps({"1": "changed", "2": "publisher-b", "3": "publisher-c"}),
                               encoding="utf-8")
            changed = subprocess.run(command + [
                "--seal-reviewed-corpus-sha256", hashes["corpus_sha256"],
                "--seal-reviewed-origins-sha256", hashes["origins_sha256"],
                "--replace-existing-manifest",
            ], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("changed since operator review", changed.stderr)


if __name__ == "__main__":
    unittest.main()
