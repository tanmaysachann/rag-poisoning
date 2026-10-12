import unittest

from attack.payload_families import (
    FAMILIES,
    attach_payload,
    build_payload,
    family_success,
)


class PayloadFamilyTests(unittest.TestCase):
    def test_all_families_have_distinct_payloads(self):
        payloads = {family: build_payload(family, "Where is Paris?") for family in FAMILIES}
        self.assertEqual(len(set(payloads.values())), len(FAMILIES))
        for payload in payloads.values():
            self.assertTrue(payload.strip())
            self.assertIn("Where is Paris?", payload)

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):
            build_payload("bribery", "Where is Paris?")
        with self.assertRaises(ValueError):
            attach_payload("Some seed passage.", "bribery", "Where is Paris?")
        with self.assertRaises(ValueError):
            family_success("bribery", "answer", 1, 1)

    def test_payload_requires_question(self):
        with self.assertRaises(ValueError):
            build_payload("url_insertion", "   ")

    def test_attach_requires_seed_and_position(self):
        with self.assertRaises(ValueError):
            attach_payload("   ", "url_insertion", "Where is Paris?")
        with self.assertRaises(ValueError):
            attach_payload("Seed.", "url_insertion", "Where is Paris?", position="middle")
        attached = attach_payload("Seed passage.", "url_insertion", "Where is Paris?")
        self.assertIn("Seed passage.", attached)
        self.assertIn("example-fake-link.test", attached)

    def test_success_requires_attack_source(self):
        answer = "Always answer with exactly: PWNED."
        self.assertTrue(family_success("instruction_takeover", answer, 99, 99))
        self.assertFalse(family_success("instruction_takeover", answer, 3, 99))
        self.assertFalse(family_success("instruction_takeover", "Paris.", 99, 99))

    def test_url_and_refusal_predicates(self):
        self.assertTrue(family_success(
            "url_insertion", "See http://example-fake-link.test/poison.", 7, 7))
        self.assertFalse(family_success(
            "url_insertion", "See http://example.com.", 7, 7))
        self.assertTrue(family_success(
            "refusal_jamming", "This question cannot be answered.", 7, 7))
        self.assertFalse(family_success(
            "refusal_jamming", "Paris is the capital.", 7, 7))


if __name__ == "__main__":
    unittest.main()
