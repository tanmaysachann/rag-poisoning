"""The optional generator may only replace a verified accepted-source answer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from generation.grounded_answer import answer_from_accepted


class GroundedAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [{"doc_id": 7, "text": "The tower is in Paris, France."}]
        self.retriever = SimpleNamespace(embedder=SimpleNamespace(backend="hashing"))
        self.extractor = patch(
            "generation.grounded_answer._select_answer",
            return_value=("The tower is in Paris, France.", 7, "The tower is in Paris, France."),
        )

    def test_invalid_model_citation_preserves_extractively_grounded_answer(self) -> None:
        with self.extractor:
            for candidate in (
                ("The tower is in London. [DOC 7]", 7),
                ("The tower is in Paris. [DOC 8]", 8),
                ("The tower is in Paris. [DOC 7]", 8),
                ("Insufficient relevant evidence was retrieved to answer this question.", None),
            ):
                with self.subTest(candidate=candidate):
                    result = answer_from_accepted(
                        "Where is the tower?", self.docs, self.retriever,
                        model_enabled=True, model_answerer=lambda _q, _d: candidate,
                    )
                    self.assertEqual(result["source_doc_id"], 7)
                    self.assertEqual(result["citations"], [
                        {"doc_id": 7, "span": "The tower is in Paris, France."}
                    ])
                    self.assertEqual(result["generation_status"], "model_invalid_extractive_fallback")

    def test_valid_model_span_has_verified_citation(self) -> None:
        with self.extractor:
            result = answer_from_accepted(
                "Where is the tower?", self.docs, self.retriever,
                model_enabled=True,
                model_answerer=lambda _q, _d: ("Paris, France. [DOC 7]", 7),
            )
        self.assertEqual(result["backend"], "local_llm_verified_span")
        self.assertEqual(result["citations"], [{"doc_id": 7, "span": "Paris, France."}])

    def test_model_load_error_uses_extractor(self) -> None:
        def unavailable(_query, _documents):
            raise RuntimeError("weights unavailable")

        with self.extractor:
            result = answer_from_accepted(
                "Where is the tower?", self.docs, self.retriever,
                model_enabled=True, model_answerer=unavailable,
            )
        self.assertEqual(result["generation_status"], "model_unavailable_extractive_fallback")
        self.assertEqual(result["source_doc_id"], 7)


if __name__ == "__main__":
    unittest.main()
