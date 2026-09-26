import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from backend.api import assets, research_cases, research_summary, test_summary
from evaluation.verify_test_v1 import _matches_saved_hash


class ResearchApiTests(unittest.TestCase):
    def test_research_summary_contains_only_validation_summaries(self):
        payload = research_summary()
        self.assertEqual(payload["split"], "validation")
        self.assertTrue(payload["test_split_evaluated"])
        ppo = payload.get("summaries", {}).get("ppo")
        if ppo is not None:
            self.assertNotIn("checkpoint", ppo)
            self.assertIsInstance(ppo.get("cases"), int)

    def test_static_asset_route_rejects_path_escape(self):
        with self.assertRaises(HTTPException) as caught:
            assets("../config.py")
        self.assertEqual(caught.exception.status_code, 404)

    def test_case_route_rejects_unlisted_paths(self):
        with self.assertRaises(HTTPException) as caught:
            research_cases("../secrets", "hashing")
        self.assertEqual(caught.exception.status_code, 400)

    def test_test_summary_contains_only_sealed_aggregates(self):
        payload = test_summary()
        self.assertEqual(payload["split"], "test")
        self.assertEqual(payload["protocol"], "v1")
        if payload["available"]:
            self.assertEqual(payload["summaries"]["clean"]["cases"], 75)
            self.assertNotIn("cases", payload)

    def test_sealed_json_hash_accepts_only_checkout_line_ending_change(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            windows_bytes = b'{\r\n  "count": 75\r\n}\r\n'
            expected = hashlib.sha256(windows_bytes).hexdigest()
            path.write_bytes(windows_bytes.replace(b"\r\n", b"\n"))
            self.assertTrue(_matches_saved_hash(path, expected, portable_json=True))
            self.assertFalse(_matches_saved_hash(path, expected, portable_json=False))
            path.write_bytes(b'{\n  "count": 76\n}\n')
            self.assertFalse(_matches_saved_hash(path, expected, portable_json=True))


if __name__ == "__main__":
    unittest.main()
