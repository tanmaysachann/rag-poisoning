import unittest
import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from backend.api import ResearchRunRequest, assets, lab_cases, lab_ppo, lab_ppo_case, lab_run, research_cases, research_summary, test_summary
from backend.research_lab import _detector
from evaluation.verify_test_v1 import _matches_saved_hash
from retrieval.hybrid_retriever import TextEmbedder


class ResearchApiTests(unittest.TestCase):
    def test_research_summary_contains_only_validation_summaries(self):
        payload = research_summary()
        self.assertEqual(payload["split"], "validation")
        self.assertTrue(payload["test_split_evaluated"])
        ppo = payload.get("summaries", {}).get("ppo")
        if ppo is not None:
            self.assertNotIn("checkpoint", ppo)
            self.assertIsInstance(ppo.get("cases"), int)
        multiseed = payload.get("summaries", {}).get("ppo_multiseed")
        self.assertIsNotNone(multiseed)
        self.assertEqual(multiseed["total_paired_cases"], 225)
        self.assertEqual(multiseed["ppo_successes"], 31)
        self.assertEqual(multiseed["fixed_substitution_successes"], 34)
        self.assertEqual(multiseed["random_edit_successes"], 6)
        ablation = payload["summaries"]["ppo_detection_ablation"]
        self.assertEqual(ablation["original_successes"], 31)
        self.assertEqual(ablation["no_detection_reward_successes"], 11)
        self.assertEqual(payload["summaries"]["ppo_proxy_ablation"]["changed_case_outcomes"], 0)
        self.assertEqual(payload["summaries"]["ppo_cache_ablation"]["changed_case_outcomes"], 0)
        self.assertEqual(payload["summaries"]["ppo_head_ablation"]["changed_case_outcomes"], 0)
        self.assertEqual(payload["summaries"]["retrieval_contriever"]["cases"], 75)
        self.assertEqual(payload["summaries"]["unseen_templates"]["styles"]["qa_header"]["cases"], 75)

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

    def test_live_research_lab_runs_isolated_validation_attack(self):
        self.assertIsInstance(_detector().embedder, TextEmbedder)
        catalog = lab_cases()
        self.assertEqual(catalog["split"], "validation")
        self.assertEqual(len(catalog["cases"]), 75)
        request = ResearchRunRequest(qid="msmarco-275", strategy="stealth", surface="accepted_ingest")
        accepted = lab_run(request)
        altered = next(row for row in accepted["documents"] if row["is_attack"])
        self.assertEqual(altered["integrity"], "verified")
        self.assertEqual(altered["decision"], "accept")
        self.assertTrue(accepted["defended_attack_success"])
        self.assertEqual(len(altered["features"]), 8)
        self.assertEqual([row["stage"] for row in accepted["audit"]], [
            "retrieval", "integrity", "detection", "provenance", "answer",
        ])
        self.assertEqual(len(accepted["counterfactuals"]), sum(
            row["decision"] == "accept" for row in accepted["documents"]
        ))
        self.assertIn("answer_and_loo_ms", accepted["stage_times"])
        citation = accepted["defended"]["citations"][0]
        cited_doc = next(row for row in accepted["documents"] if row["doc_id"] == citation["doc_id"])
        self.assertEqual(cited_doc["decision"], "accept")
        self.assertIn(citation["span"], cited_doc["text"])

        request.surface = "post_index_tamper"
        tampered = lab_run(request)
        altered = next(row for row in tampered["documents"] if row["is_attack"])
        self.assertEqual(altered["integrity"], "tampered")
        self.assertEqual(altered["decision"], "quarantine")
        self.assertFalse(tampered["defended_attack_success"])
        self.assertNotEqual(tampered["defended"]["source_doc_id"], altered["doc_id"])

        request.surface = "accepted_ingest"
        request.attack_text = next(row["original_text"] for row in catalog["cases"] if row["qid"] == request.qid)
        edited = lab_run(request)
        self.assertFalse(edited["defended_attack_success"])

    def test_ppo_endpoint_exposes_training_and_one_validation_trace(self):
        history = lab_ppo()
        self.assertEqual(len(history["runs"]["undefended"]), 10)
        self.assertEqual(history["validation"]["undefended"]["cases"], 75)
        self.assertNotIn("checkpoint", history["validation"]["undefended"])
        case = lab_ppo_case("msmarco-275")
        self.assertIn("trace", case["runs"]["defender_aware"])
        with self.assertRaises(HTTPException):
            lab_ppo_case("not-a-validation-question")

    def test_research_page_has_every_live_lab_binding(self):
        root = Path(__file__).resolve().parents[1] / "frontend"
        html = (root / "index.html").read_text(encoding="utf-8")
        script = (root / "research.js").read_text(encoding="utf-8")
        html_ids = set(re.findall(r'id="([^"]+)"', html))
        script_ids = set(re.findall(r"el\('([^']+)'\)", script))
        self.assertFalse(script_ids - html_ids, script_ids - html_ids)


if __name__ == "__main__":
    unittest.main()
