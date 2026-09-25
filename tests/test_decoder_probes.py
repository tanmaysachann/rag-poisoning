"""Causal final-token attention concentration normalization."""

from __future__ import annotations

import unittest

import numpy as np

from detect.decoder_probes import final_token_attention_concentration


class DecoderProbeTests(unittest.TestCase):
    def test_uniform_and_single_key_attention(self) -> None:
        uniform = np.full((2, 4), 0.25)
        peaked = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.assertAlmostEqual(final_token_attention_concentration(uniform), 0.0)
        self.assertAlmostEqual(final_token_attention_concentration(peaked), 1.0)

    def test_rejects_full_attention_matrix(self) -> None:
        with self.assertRaises(ValueError):
            final_token_attention_concentration(np.ones((2, 4, 4)))


if __name__ == "__main__":
    unittest.main()
