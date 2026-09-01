"""Regression tests for query-dependent retrieval and defense-only filtering."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline.secure_rag import secure_rag_answer


def _synthetic_live_documents(query: str, limit: int = 3) -> list[dict]:
    """Return deterministic live-source stand-ins without making a network call."""

    facts = {
        "india": (710001, "India", "New Delhi is the capital of India and the seat of its central government."),
        "japan": (710002, "Japan", "Tokyo is the capital of Japan and its most populous city."),
    }
    key = next((name for name in facts if name in query.lower()), None)
    if key is None:
        return []
    doc_id, title, text = facts[key]
    return [{
        "doc_id": doc_id,
        "title": title,
        "source_type": "test live source",
        "source_url": f"https://example.test/{key}",
        "text": text,
        "live_source": True,
    }][:limit]


class DynamicPipelineTests(unittest.TestCase):
    @patch("pipeline.secure_rag.fetch_wikipedia_documents", side_effect=_synthetic_live_documents)
    def test_unseen_queries_produce_different_retrieved_answers(self, _fetch: object) -> None:
        india = secure_rag_answer("What is the capital of India?", live_retrieval=True)
        japan = secure_rag_answer("What is the capital of Japan?", live_retrieval=True)

        self.assertIn("New Delhi", india["answer"])
        self.assertIn("Tokyo", japan["answer"])
        self.assertNotEqual(india["answer"], japan["answer"])
        self.assertEqual(india["source_doc_id"], 710001)
        self.assertEqual(japan["source_doc_id"], 710002)

    @patch("pipeline.secure_rag.fetch_wikipedia_documents", side_effect=_synthetic_live_documents)
    def test_defense_does_not_switch_answer_algorithm(self, _fetch: object) -> None:
        query = "What is the capital of India?"
        defended = secure_rag_answer(query, defense_enabled=True, live_retrieval=True)
        unfiltered = secure_rag_answer(query, defense_enabled=False, live_retrieval=True)

        self.assertEqual(defended["answer"], unfiltered["answer"])
        self.assertEqual(defended["source_doc_id"], unfiltered["source_doc_id"])

    def test_poisoning_can_change_answer_without_forced_attack_logic(self) -> None:
        query = "Where is the Eiffel Tower located?"
        defended = secure_rag_answer(query, defense_enabled=True, live_retrieval=False)
        unfiltered = secure_rag_answer(query, defense_enabled=False, live_retrieval=False)

        self.assertIn("Paris", defended["answer"])
        self.assertIn("London", unfiltered["answer"])
        self.assertIn(1001, [doc["doc_id"] for doc in defended["filtered_docs"]])


if __name__ == "__main__":
    unittest.main()
