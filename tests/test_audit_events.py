import unittest

from audit.events import build_event, validate_event, validate_trace


class AuditEventTests(unittest.TestCase):
    def test_build_and_validate_round_trip(self):
        event = build_event("retrieval", {"candidate_ids": [1, 2]})
        self.assertEqual(event["schema_version"], 1)
        validate_event(event)

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            build_event("training", {})

    def test_malformed_event_rejected(self):
        with self.assertRaises(ValueError):
            validate_event({"schema_version": 999, "stage": "retrieval", "payload": {}})
        with self.assertRaises(ValueError):
            validate_event({"schema_version": 1, "stage": "retrieval", "payload": []})

    def test_trace_requires_retrieval_and_answer(self):
        validate_trace([
            build_event("retrieval", {}),
            build_event("answer", {"abstained": False}),
        ])
        with self.assertRaises(ValueError):
            validate_trace([build_event("retrieval", {})])
        with self.assertRaises(ValueError):
            validate_trace([])


if __name__ == "__main__":
    unittest.main()
