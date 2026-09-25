import unittest

from fastapi import HTTPException

from backend.api import assets, research_cases, research_summary, test_summary


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


if __name__ == "__main__":
    unittest.main()
