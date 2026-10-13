import unittest

from generation.answer_key_corroboration import (
    containment_decision,
    content_tokens,
    is_abstention,
    key_supported,
)


class AnswerKeyCorroborationTests(unittest.TestCase):
    def test_key_supported_strict_vs_content_only(self):
        peer = "Paris is the capital of France, home to the Eiffel Tower."
        supported, missing = key_supported("Paris is the capital", peer, content_only=False)
        self.assertTrue(supported)
        self.assertEqual(missing, [])
        supported, missing = key_supported("Berlin is the capital", peer, content_only=True)
        self.assertFalse(supported)
        self.assertIn("berlin", missing)

    def test_stopwords_ignored_in_content_mode(self):
        peer = "Canberra serves as capital."
        supported, _ = key_supported("Canberra is the capital", peer, content_only=True)
        self.assertTrue(supported)
        supported, missing = key_supported("Canberra is the capital", peer, content_only=False)
        self.assertFalse(supported)
        self.assertTrue(missing)

    def test_selected_abstention_is_kept(self):
        decision = containment_decision(
            "Insufficient relevant evidence was retrieved to answer this question.",
            None, "Some peer text.", 2, content_only=True)
        self.assertTrue(decision["gate_keeps_answer"])

    def test_missing_peer_abstains(self):
        decision = containment_decision("Paris", 1, None, None, content_only=True)
        self.assertFalse(decision["gate_keeps_answer"])
        self.assertEqual(decision["reason"], "missing_peer")

    def test_is_abstention(self):
        self.assertTrue(is_abstention("Insufficient relevant evidence."))
        self.assertFalse(is_abstention("Paris"))
        self.assertEqual(content_tokens("What is Paris?"), ["paris"])


if __name__ == "__main__":
    unittest.main()
