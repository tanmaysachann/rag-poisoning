import unittest

from generation.answer_corroboration import corroboration_decision, is_abstention, token_f1


class AnswerCorroborationTests(unittest.TestCase):
    def test_token_f1_is_symmetric_and_bounded(self):
        first = "Paris has a large tower"
        second = "large tower in Paris"
        self.assertAlmostEqual(token_f1(first, second), token_f1(second, first))
        self.assertGreaterEqual(token_f1(first, second), 0.0)
        self.assertLessEqual(token_f1(first, second), 1.0)
        self.assertEqual(token_f1(first, first), 1.0)
        self.assertEqual(token_f1(first, "completely different words xyz"), 0.0)
        self.assertEqual(token_f1("", second), 0.0)

    def test_selected_abstention_is_kept(self):
        decision = corroboration_decision(
            "Insufficient relevant evidence was retrieved to answer this question.",
            None, "Paris", 2, 0.5,
        )
        self.assertTrue(decision["gate_keeps_answer"])
        self.assertEqual(decision["reason"], "selected_abstained_no_answer")

    def test_missing_peer_fail_closed_abstains(self):
        decision = corroboration_decision("Paris", 1, None, None, 0.5)
        self.assertFalse(decision["gate_keeps_answer"])
        self.assertEqual(decision["reason"], "missing_peer_or_peer_abstained")

    def test_missing_peer_fail_open_keeps(self):
        decision = corroboration_decision(
            "Paris", 1, None, None, 0.5, fail_closed_on_missing_peer=False
        )
        self.assertTrue(decision["gate_keeps_answer"])

    def test_disagreement_abstains_without_failover(self):
        decision = corroboration_decision("Paris", 1, "Berlin", 2, 0.5)
        self.assertFalse(decision["gate_keeps_answer"])
        self.assertEqual(decision["reason"], "peer_disagrees_abstain")

    def test_agreement_keeps(self):
        decision = corroboration_decision("Paris is the capital of France", 1, "capital of France is Paris", 2, 0.3)
        self.assertTrue(decision["gate_keeps_answer"])
        self.assertEqual(decision["reason"], "peer_agrees")

    def test_is_abstention(self):
        self.assertTrue(is_abstention("Insufficient relevant evidence was retrieved."))
        self.assertFalse(is_abstention("Paris"))


if __name__ == "__main__":
    unittest.main()
