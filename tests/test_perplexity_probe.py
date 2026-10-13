import unittest

from evaluation.evaluate_perplexity import flag_high_nll, threshold_at_fpr


class PerplexityProbeTests(unittest.TestCase):
    def test_threshold_holds_target_fpr(self):
        clean = [float(value) for value in range(100)]
        threshold = threshold_at_fpr(clean, 0.05)
        flagged = sum(score > threshold for score in clean)
        self.assertLessEqual(flagged / len(clean), 0.05)
        with self.assertRaises(ValueError):
            threshold_at_fpr([], 0.05)
        with self.assertRaises(ValueError):
            threshold_at_fpr(clean, 0.0)

    def test_flag_direction(self):
        self.assertTrue(flag_high_nll(5.0, 4.0))
        self.assertFalse(flag_high_nll(3.0, 4.0))
        self.assertFalse(flag_high_nll(4.0, 4.0))


if __name__ == "__main__":
    unittest.main()
