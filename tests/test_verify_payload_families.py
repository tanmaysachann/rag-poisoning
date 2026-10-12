import unittest

from evaluation.verify_payload_families import verify


class VerifyPayloadFamiliesTests(unittest.TestCase):
    def test_saved_probe_verifies_without_test_split(self):
        result = verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(
            set(result["families"]),
            {"instruction_takeover", "url_insertion", "refusal_jamming"},
        )
        self.assertFalse(result["test_attacks_evaluated"])


if __name__ == "__main__":
    unittest.main()
