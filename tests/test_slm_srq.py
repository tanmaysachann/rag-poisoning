import unittest

import numpy as np

from detect.slm_srq import calibrate_srq, score_srq


class FakeEmbedder:
    def encode(self, words):
        mapping = {
            "illness": [1.0, 0.0], "disease": [1.0, 0.0],
            "health": [0.8, 0.2], "city": [0.0, 1.0],
        }
        return np.asarray([mapping.get(word, [0.0, 0.0]) for word in words])


class SrqTests(unittest.TestCase):
    def test_semantic_variant_and_query_contribution(self):
        result = score_srq("city", "illness city", "disease disease", FakeEmbedder())
        self.assertEqual(result["document_vocabulary_size"], 2)
        self.assertAlmostEqual(result["query_contribution"], 0.5)
        self.assertAlmostEqual(result["response_contribution"], 1.0)
        self.assertAlmostEqual(result["raw_srq"], 1.5)
        self.assertEqual(result["display_srq"], 1.0)

    def test_validation_threshold_is_not_a_probability_cutoff(self):
        threshold = calibrate_srq([0.1, 0.2, 0.3, 0.4], target_fpr=0.0)
        self.assertGreater(threshold, 0.4)

    def test_empty_document_rejected(self):
        with self.assertRaises(ValueError):
            score_srq("city", "", "city", FakeEmbedder())


if __name__ == "__main__":
    unittest.main()
