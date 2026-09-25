import json
import tempfile
import unittest
from pathlib import Path

from security.ingest import inspect_ingest, seal_ingest
from security.integrity import load_manifest


class TrustedIngestTests(unittest.TestCase):
    def test_review_digest_is_required_and_signed_manifest_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "corpus.jsonl"
            manifest = Path(temporary) / "protected" / "manifest.json"
            corpus.write_text(json.dumps({"doc_id": 1, "text": "Paris is in France."}) + "\n", encoding="utf-8")
            preview = inspect_ingest(corpus, manifest)
            self.assertEqual(preview["new_doc_ids"], ["1"])
            with self.assertRaises(ValueError):
                seal_ingest(corpus, manifest, expected_corpus_sha256="0" * 64)
            sealed = seal_ingest(corpus, manifest, expected_corpus_sha256=preview["corpus_sha256"], signing_key=b"test-key")
            self.assertTrue(sealed["signed"])
            self.assertTrue(load_manifest(manifest, signing_key=b"test-key", require_signature=True).signature_verified)
            corpus.write_text(json.dumps({"doc_id": 1, "text": "Lyon is in France."}) + "\n", encoding="utf-8")
            changed = inspect_ingest(corpus, manifest, signing_key=b"test-key")
            self.assertEqual(changed["changed_doc_ids"], ["1"])
            with self.assertRaises(FileExistsError):
                seal_ingest(corpus, manifest, expected_corpus_sha256=changed["corpus_sha256"])


if __name__ == "__main__":
    unittest.main()
