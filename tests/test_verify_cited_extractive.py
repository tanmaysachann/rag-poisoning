import unittest

from evaluation.evaluate_cited_extractive import verify


class VerifyCitedExtractiveTests(unittest.TestCase):
    def test_saved_comparison_verifies_without_test_split(self):
        result = verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["cases"], 20)
        self.assertFalse(result["test_attacks_evaluated"])


if __name__ == "__main__":
    unittest.main()
