import hashlib
import unittest
from unittest.mock import patch

from pipeline.v24_answer_gate import answer_with_v24


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class V24AnswerGateTests(unittest.TestCase):
    def setUp(self):
        self.reference = "The trusted answer is 1972."
        self.candidates = [
            {"doc_id": 1, "text": self.reference},
            {"doc_id": 2, "text": "The attacker says 1974."},
        ]

    def test_filters_before_answer_and_does_not_admit_new_document(self):
        calls = []

        def scan(query, reference, candidate, expected):
            calls.append((candidate, expected))
            flag = candidate != self.reference
            return {"decision": "review" if flag else "pass",
                    "learned": {"flag": flag},
                    "integrity": {"flag": False},
                    "identity": {"status": "not_supplied", "flag": False}}

        def answer(query, documents, retriever):
            chosen = documents[-1] if documents else None
            return {"answer": chosen["text"] if chosen else "abstained",
                    "source_doc_id": chosen["doc_id"] if chosen else None}

        with patch("pipeline.v24_answer_gate.answer_from_accepted", side_effect=answer):
            result = answer_with_v24(
                "When?", self.candidates, object(), pinned_reference=self.reference,
                pinned_reference_sha256=digest(self.reference),
                expected_hashes={"1": digest(self.reference)}, scan=scan,
            )
        self.assertEqual(calls, [(self.reference, digest(self.reference)),
                                 ("The attacker says 1974.", None)])
        self.assertEqual(result["undefended"]["source_doc_id"], 2)
        self.assertEqual(result["defended"]["source_doc_id"], 1)
        self.assertEqual(result["review_doc_ids"], [2])
        self.assertEqual(result["documents"][1]["admission_reason"], "unknown_source")

    def test_scanner_pass_cannot_admit_unknown_or_modified_source(self):
        def scan(*args):
            return {"decision": "pass", "learned": {"flag": False},
                    "integrity": {"flag": False},
                    "identity": {"status": "not_supplied", "flag": False}}

        def answer(query, documents, retriever):
            chosen = documents[-1] if documents else None
            return {"answer": chosen["text"] if chosen else "abstained",
                    "source_doc_id": chosen["doc_id"] if chosen else None}

        with patch("pipeline.v24_answer_gate.answer_from_accepted", side_effect=answer):
            result = answer_with_v24(
                "When?", self.candidates, object(), pinned_reference=self.reference,
                pinned_reference_sha256=digest(self.reference),
                expected_hashes={"1": digest(self.reference)}, scan=scan,
            )
        self.assertEqual(result["defended"]["source_doc_id"], 1)
        self.assertEqual(result["documents"][1]["scanner_decision"], "pass")
        self.assertEqual(result["documents"][1]["admission_reason"], "unknown_source")
        with patch("pipeline.v24_answer_gate.answer_from_accepted", side_effect=answer):
            altered = answer_with_v24(
                "When?", self.candidates, object(), pinned_reference=self.reference,
                pinned_reference_sha256=digest(self.reference),
                expected_hashes={"1": digest(self.reference), "2": digest("original")},
                scan=scan,
            )
        self.assertEqual(altered["documents"][1]["admission_reason"], "source_hash_mismatch")
        self.assertEqual(altered["defended"]["source_doc_id"], 1)

    def test_rejects_unpinned_reference_before_scanning(self):
        with self.assertRaisesRegex(ValueError, "Reference differs"):
            answer_with_v24(
                "When?", self.candidates, object(), pinned_reference="forged",
                pinned_reference_sha256=digest(self.reference),
                expected_hashes={}, scan=lambda *args: self.fail("scanner ran"),
            )

    def test_invalid_scanner_result_fails_closed(self):
        with patch("pipeline.v24_answer_gate.answer_from_accepted",
                   return_value={"answer": "x", "source_doc_id": 1}):
            with self.assertRaisesRegex(ValueError, "invalid decision"):
                answer_with_v24(
                    "When?", self.candidates, object(), pinned_reference=self.reference,
                    pinned_reference_sha256=digest(self.reference),
                    expected_hashes={}, scan=lambda *args: {"decision": "unknown"},
                )


if __name__ == "__main__":
    unittest.main()
