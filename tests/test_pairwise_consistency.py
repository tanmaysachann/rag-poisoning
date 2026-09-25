import unittest

import numpy as np

from detect.pairwise_consistency import FEATURE_NAMES, pair_features


class _FakeEmbedder:
    def encode(self, texts):
        vectors = np.asarray([[len(text), sum(ord(char) for char in text) % 17] for text in texts], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class PairwiseConsistencyTests(unittest.TestCase):
    def test_features_are_symmetric_and_finite(self):
        embedder = _FakeEmbedder()
        left = "Paris has a large tower next to the Seine river."
        right = "The Seine river runs through Paris near the tower."
        first = pair_features("Where is the tower?", left, right, embedder)
        second = pair_features("Where is the tower?", right, left, embedder)
        self.assertEqual(len(first), len(FEATURE_NAMES))
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
