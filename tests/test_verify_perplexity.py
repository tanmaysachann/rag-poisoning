import unittest

from evaluation.verify_perplexity import verify


class VerifyPerplexityTests(unittest.TestCase):
    def test_saved_probe_verifies_without_test_split(self):
        result = verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["cases"], 75)
        self.assertFalse(result["test_attacks_evaluated"])


if __name__ == "__main__":
    unittest.main()
