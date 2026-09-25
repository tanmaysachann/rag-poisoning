import unittest

from evaluation.calibrate_paired_gate_v2 import threshold_for_clean_rejection


class PairedGateCalibrationTests(unittest.TestCase):
    def test_missing_peers_consume_clean_rejection_budget(self):
        threshold, rejected = threshold_for_clean_rejection(
            [None, 0.1, 0.2, *([0.8] * 17)], 0.10
        )
        self.assertEqual(threshold, 0.2)
        self.assertEqual(rejected, 2)

    def test_missing_peer_target_returns_best_possible_floor(self):
        threshold, rejected = threshold_for_clean_rejection([None, 0.8, 0.9], 0.05)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(rejected, 1)


if __name__ == "__main__":
    unittest.main()
