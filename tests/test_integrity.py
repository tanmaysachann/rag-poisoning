"""Integrity must fail closed for altered, unknown, and forged documents."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from security.integrity import load_manifest, write_manifest


class IntegrityManifestTests(unittest.TestCase):
    def test_signed_manifest_rejects_changes_and_unknown_documents(self) -> None:
        documents = [{"doc_id": 1, "text": "Trusted evidence about Paris."}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_manifest(
                path, documents, hashlib.sha256(b"clean corpus").hexdigest(),
                signing_key=b"test-only-secret",
            )
            manifest = load_manifest(path, signing_key=b"test-only-secret", require_signature=True)
            self.assertTrue(manifest.signature_verified)
            self.assertEqual(manifest.check(1, documents[0]["text"])["status"], "verified")
            self.assertEqual(manifest.check(1, "Altered evidence")["status"], "tampered")
            self.assertEqual(manifest.check(2, "New evidence")["status"], "unknown_document")
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                load_manifest(path, signing_key=b"wrong-secret", require_signature=True)


if __name__ == "__main__":
    unittest.main()
