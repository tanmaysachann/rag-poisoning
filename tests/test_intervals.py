import unittest

from evaluation.metrics import paired_bootstrap_difference, wilson_interval


class IntervalTests(unittest.TestCase):
    def test_wilson_handles_zero_and_all_successes(self):
        low = wilson_interval(0, 75)
        high = wilson_interval(75, 75)
        self.assertEqual(low[0], 0.0)
        self.assertLess(low[1], 0.05)
        self.assertGreater(high[0], 0.95)
        self.assertEqual(high[1], 1.0)

    def test_paired_bootstrap_preserves_query_pairing(self):
        self.assertEqual(paired_bootstrap_difference([True] * 10, [False] * 10), (1.0, 1.0))
        with self.assertRaises(ValueError):
            paired_bootstrap_difference([True], [])


if __name__ == "__main__":
    unittest.main()
