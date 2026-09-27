import unittest

import numpy as np

from detect.pairwise_consistency import FEATURE_NAMES, pair_features
from evaluation.evaluate_paired_gate_v2 import paired_decision


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

    def test_pair_gate_uses_same_query_group_only(self):
        class FirstSentence:
            def select(self, _query, documents, _retriever):
                first = documents[0]
                return first["text"], first["doc_id"], first["text"]

        class Encoder:
            def __init__(self, documents):
                self.documents = documents

            def encode(self, _value):
                return np.asarray([[1.0, 0.0]])

        grouped = [
            {"doc_id": 1, "text": "Paris", "source_query_id": 11},
            {"doc_id": 3, "text": "London", "source_query_id": 22},
            {"doc_id": 2, "text": "Paris, France", "source_query_id": 11},
        ]
        ranked = [{key: value for key, value in row.items() if key != "source_query_id"}
                  for row in grouped]
        self.assertEqual(paired_decision("Where?", ranked, Encoder(grouped), FirstSentence(), 0)["peer_doc_id"], 2)
        ungrouped = [{key: value for key, value in row.items() if key != "source_query_id"}
                     for row in grouped]
        self.assertIsNone(paired_decision("Where?", ranked, Encoder(ungrouped), FirstSentence(), 0)["peer_doc_id"])


if __name__ == "__main__":
    unittest.main()
