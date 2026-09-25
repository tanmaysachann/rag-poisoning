"""Behavioral tests for the edit MDP and its terminal retrieval reward."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poison.actions import EditAction, OPERATIONS
from poison.env import DocumentEditEnv
from poison.reward import terminal_reward


class DocumentEditEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.corpus = Path(self.temp.name) / "corpus.jsonl"
        docs = [
            {"doc_id": 1, "text": "The capital of France is Paris. Paris is in Europe."},
            {"doc_id": 2, "text": "Mars is a planet in the solar system."},
        ]
        self.corpus.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
        self.query = {
            "qid": "fixture-1", "question": "What is the capital of France?",
            "answer_aliases": ["Paris"], "support_doc_ids": [1],
        }

    def _env(self):
        return DocumentEditEnv(self.corpus, self.query, "Lyon", max_steps=3)

    def test_mask_and_seeded_reset_are_reproducible(self):
        env = self._env()
        first, info = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (774,))
        self.assertFalse(info["action_mask"][OPERATIONS.index("STOP"), 0, 0])
        self.assertFalse(info["action_mask"][OPERATIONS.index("STOP"), 1, 0])

    def test_edit_and_terminal_reward(self):
        env = self._env()
        env.reset(seed=7)
        _, _, terminated, truncated, info = env.step(EditAction("INSERT", 0, 1))
        self.assertTrue(info["valid"])
        self.assertIn("Lyon", info["document"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, reward, terminated, truncated, info = env.step(EditAction("STOP"))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["terminal"]["retrieved"])
        self.assertGreater(reward, 0)
        with self.assertRaises(RuntimeError):
            env.step(EditAction("STOP"))

    def test_invalid_action_does_not_mutate_document(self):
        env = self._env()
        env.reset(seed=7)
        before = env.text
        _, reward, _, _, info = env.step(EditAction("PARAPHRASE", 1, 0))
        self.assertFalse(info["valid"])
        self.assertEqual(env.text, before)
        self.assertEqual(reward, -1.0)

    def test_identical_insert_is_masked_after_first_edit(self):
        env = self._env()
        env.reset(seed=7)
        env.step(EditAction("INSERT", 0, 1))
        self.assertFalse(env.action_mask()[OPERATIONS.index("INSERT"), 0, 1])

    def test_terminal_success_requires_retrieval(self):
        with self.assertRaises(ValueError):
            terminal_reward(retrieved=False, attack_success=True)

    def test_defender_aware_terminal_records_detection(self):
        class Detector:
            def score(self, question, text):
                risk = 1.0 if "Lyon" in text else 0.0
                return {"risk_score": risk, "decision": "quarantine" if risk else "accept"}

        env = DocumentEditEnv(self.corpus, self.query, "Lyon", max_steps=3, detector=Detector())
        env.reset(seed=7)
        env.step(EditAction("INSERT", 0, 1))
        _, _, _, _, info = env.step(EditAction("STOP"))
        self.assertTrue(info["terminal"]["attack_doc_detected"])
        self.assertFalse(info["terminal"]["defended_attack_success"])
        self.assertEqual(info["terminal"]["reward"]["attack_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
