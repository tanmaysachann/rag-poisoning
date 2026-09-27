"""Experimental aligned-span repair must stay inside accepted evidence."""

from __future__ import annotations

import unittest

from generation.answer_repair import repair_model_answer
from generation.local_llm import ABSTENTION


class AnswerRepairTests(unittest.TestCase):
    def test_paraphrased_output_yields_only_a_contiguous_source_phrase(self):
        documents = [{"doc_id": 4, "text": "The TV medium Colin Fry has died of lung cancer aged 53."}]
        answer, source, status = repair_model_answer(
            "What did Colin Fry die of?", "Colin Fry died of lung cancer.", documents,
        )
        self.assertEqual((answer, source, status),
                         ("died of lung cancer [DOC 4]", 4, "aligned_phrase"))

    def test_query_echo_and_untrusted_document_markup_abstain(self):
        documents = [{"doc_id": 4, "text": "The frustules are made of silica."}]
        self.assertEqual(repair_model_answer("What are frustules made of?", "frustules", documents)[0], ABSTENTION)
        self.assertEqual(repair_model_answer("What are frustules made of?", "[DOC id=4] silica", documents)[0], ABSTENTION)

    def test_unrelated_model_claim_does_not_gain_a_citation(self):
        documents = [{"doc_id": 4, "text": "The tower is in Paris, France."}]
        self.assertEqual(repair_model_answer("Where is the tower?", "The tower is in London.", documents)[0],
                         ABSTENTION)


if __name__ == "__main__":
    unittest.main()
