"""Integrity checks should flag changed copies and spare exact pinned text."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from detect.snapshot_near_duplicate import SnapshotNearDuplicate
from detect.snapshot_containment_v23 import SnapshotContainment
from detect.snapshot_integrity_detector_v22 import SnapshotIntegrityDetector


class SnapshotNearDuplicateTests(unittest.TestCase):
    def test_exact_and_changed_copy(self) -> None:
        body = ("Silverwood observatory recorded a bright comet near the northern "
                "horizon. Researchers at Silverwood observatory catalogued the "
                "comet with careful measurements and archived the public report.")
        altered = body.replace("public report", "pubic report")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            database = directory / "passages_fts5.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE VIRTUAL TABLE passages USING fts5("
                                   "doc_id UNINDEXED,title,body,sha256 UNINDEXED)")
                connection.execute(
                    "INSERT INTO passages(doc_id,title,body,sha256) VALUES(?,?,?,?)",
                    ("source-1", "", body,
                     hashlib.sha256(body.encode("utf-8")).hexdigest()))
                connection.commit()
            finally:
                connection.close()
            (directory / "manifest.json").write_text(json.dumps({
                "database_filename": database.name,
                "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            scanner = SnapshotNearDuplicate(directory)
            try:
                exact = scanner.assess(body)
                changed = scanner.assess(altered)
                self.assertTrue(exact["exact_snapshot_match"])
                self.assertEqual(exact["near_copy_score"], 0.0)
                self.assertFalse(changed["exact_snapshot_match"])
                self.assertEqual(changed["matched_doc_id"], "source-1")
                self.assertGreater(changed["near_copy_score"], 0.9)
            finally:
                scanner.close()
            containment = SnapshotContainment(directory)
            try:
                exact_contained = containment.assess(body)
                self.assertEqual(exact_contained["matched_doc_id"], "source-1")
                self.assertEqual(exact_contained["match_method"], "exact_snapshot_match")
                prefixed = ("A misleading answer was inserted before the original passage. "
                            "The inserted answer is repeated to dominate retrieval. " * 3 + body)
                found = containment.assess(prefixed)
                self.assertEqual(found["matched_doc_id"], "source-1")
                self.assertEqual(found["near_copy_score"], 1.0)
                self.assertEqual(found["match_method"], "contained_source")
            finally:
                containment.close()

    def test_short_text_is_unassessable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            database = directory / "passages_fts5.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE VIRTUAL TABLE passages USING fts5("
                                   "doc_id UNINDEXED,title,body,sha256 UNINDEXED)")
                connection.commit()
            finally:
                connection.close()
            (directory / "manifest.json").write_text(json.dumps({
                "database_filename": database.name,
                "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            scanner = SnapshotNearDuplicate(directory)
            try:
                self.assertEqual(scanner.assess("hi")["status"], "unassessable_short_text")
            finally:
                scanner.close()

    def test_known_id_hash_blocks_changed_document(self) -> None:
        body = ("Silverwood observatory recorded a bright comet near the northern "
                "horizon and archived the complete original report for readers.")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            corpus = directory / "corpus.jsonl"
            corpus.write_text(json.dumps({"doc_id": 17, "text": body}) + "\n",
                              encoding="utf-8")
            source_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
            database = directory / "passages_fts5.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE VIRTUAL TABLE passages USING fts5("
                                   "doc_id UNINDEXED,title,body,sha256 UNINDEXED)")
                connection.execute(
                    "INSERT INTO passages(doc_id,title,body,sha256) VALUES(?,?,?,?)",
                    ("17", "", body, hashlib.sha256(body.encode("utf-8")).hexdigest()))
                connection.commit()
            finally:
                connection.close()
            (directory / "manifest.json").write_text(json.dumps({
                "database_filename": database.name,
                "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                "source_corpus_sha256": source_sha,
            }), encoding="utf-8")
            detector = SnapshotIntegrityDetector(
                directory, expected_corpus_sha256=source_sha,
                containment=True, identity_corpus=corpus)
            try:
                self.assertEqual(detector.score_document("comet?", body, 17)["decision"],
                                 "accept")
                changed = detector.score_document("comet?", body + " forged answer", 17)
                self.assertEqual(changed["decision"], "quarantine")
                self.assertEqual(changed["integrity"]["status"],
                                 "known_document_hash_mismatch")
            finally:
                detector.close()


if __name__ == "__main__":
    unittest.main()
