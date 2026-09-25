import unittest

import numpy as np

from detect.model_probes import ProbeReference, attention_concentration


class ModelProbeTests(unittest.TestCase):
    def test_uniform_and_peaked_attention(self):
        uniform = np.full((2, 4, 4), 0.25)
        peaked = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
        self.assertAlmostEqual(attention_concentration(uniform), 0.0)
        self.assertAlmostEqual(attention_concentration(peaked), 1.0)

    def test_reference_scores_real_vectors_not_embedding_proxy(self):
        vectors = np.asarray([[float(i), float(i % 3)] for i in range(8)])
        reference = ProbeReference.fit(vectors, np.linspace(0.1, 0.3, 8))
        clean = reference.score(vectors[2], 0.2)
        outlier = reference.score(np.array([100.0, 100.0]), 0.8)
        self.assertGreater(outlier["layer3_mahalanobis"], clean["layer3_mahalanobis"])
        self.assertGreater(outlier["attention_deviation"], clean["attention_deviation"])


if __name__ == "__main__":
    unittest.main()
