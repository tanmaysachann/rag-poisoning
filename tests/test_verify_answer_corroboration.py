import unittest

from evaluation.verify_answer_corroboration import verify


class VerifyAnswerCorroborationTests(unittest.TestCase):
    def test_saved_probe_verifies_without_test_split(self):
        result = verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["train_cases"], 140)
        self.assertEqual(result["validation_cases"], 30)
        self.assertFalse(result["test_attacks_evaluated"])


if __name__ == "__main__":
    unittest.main()
