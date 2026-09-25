"""The attack harness must not promote edited corpus text into trusted evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attack.harness import stage_document_attack
from security.integrity import load_manifest, write_manifest


class AttackHarnessTests(unittest.TestCase):
    def test_staged_insert_and_replacement_keep_original_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trusted = root / "trusted.jsonl"
            trusted.write_text(json.dumps({"doc_id": 1, "text": "Paris is in France."}) + "\n", encoding="utf-8")
            manifest_path = root / "trusted_manifest.json"
            write_manifest(manifest_path, [{"doc_id": 1, "text": "Paris is in France."}], "fixed-digest")
            manifest_bytes = manifest_path.read_bytes()

            inserted = root / "inserted.jsonl"
            stage_document_attack(trusted, inserted, doc_id=2, text="An untrusted inserted report.")
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.check(2, "An untrusted inserted report.")["status"], "unknown_document")

            replaced = root / "replaced.jsonl"
            stage_document_attack(trusted, replaced, doc_id=1, text="An altered report.", replace_existing=True)
            self.assertEqual(manifest.check(1, "An altered report.")["status"], "tampered")
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
            self.assertEqual(trusted.read_text(encoding="utf-8"), json.dumps({"doc_id": 1, "text": "Paris is in France."}) + "\n")


if __name__ == "__main__":
    unittest.main()
