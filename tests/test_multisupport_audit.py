import unittest

from data.audit_multisupport import audit


class MultisupportAuditTests(unittest.TestCase):
    def test_distinguishes_two_answer_spans_from_two_selected_spans(self):
        row = {
            "query_id": 12, "query": "Where is the tower?", "answers": ["Paris"],
            "passages": {
                "passage_text": ["The tower is in Paris.", "Paris contains the tower."],
                "is_selected": [1, 0],
            },
        }
        result = audit([row], limit=1)["counts"]
        self.assertEqual(result["two_or_more_answer_passages"], 1)
        self.assertEqual(result["two_or_more_selected_answer_passages"], 0)


if __name__ == "__main__":
    unittest.main()
