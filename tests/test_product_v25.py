import unittest
from unittest.mock import patch

from detect.product_v25 import FUSION_THRESHOLD, ReferenceAdaptiveScanner


class FakeV24:
    def __init__(self, cases):
        self.calibration = {"fusion_threshold_1pct": FUSION_THRESHOLD}
        self.cases = cases
        self.calls = []

    def assess(self, query, reference, candidate, *, expected_sha256=None):
        self.calls.append(candidate)
        score, risk, containment, identity = self.cases[candidate]
        return {
            "decision": "review" if score >= FUSION_THRESHOLD or containment or identity else "pass",
            "learned": {"fusion_score": score, "v3_risk": risk, "flag": score >= FUSION_THRESHOLD},
            "integrity": {"flag": containment},
            "identity": {"flag": identity},
        }

    def close(self):
        pass


class ReferenceAdaptiveScannerTests(unittest.TestCase):
    def test_domain_high_clean_score_does_not_alone_trigger_review(self):
        fake = FakeV24({
            "reference": (5.0, 0.1, False, False),
            "clean": (5.2, 0.1, False, False),
            "attack": (5.6, 0.1, False, False),
        })
        with patch("detect.product_v25.PoisonDocumentScanner", return_value=fake):
            scanner = ReferenceAdaptiveScanner()
        clean = scanner.assess("question", "reference", "clean")
        attack = scanner.assess("question", "reference", "attack")
        self.assertEqual(clean["decision"], "pass")
        self.assertEqual(attack["decision"], "review")
        self.assertEqual(fake.calls.count("reference"), 1)

    def test_behavioral_and_integrity_routes_still_review(self):
        fake = FakeV24({
            "reference": (5.0, 0.1, False, False),
            "behavior": (1.0, 0.8, False, False),
            "containment": (1.0, 0.1, True, False),
            "identity": (1.0, 0.1, False, True),
        })
        with patch("detect.product_v25.PoisonDocumentScanner", return_value=fake):
            scanner = ReferenceAdaptiveScanner()
        for candidate in ("behavior", "containment", "identity"):
            with self.subTest(candidate=candidate):
                self.assertEqual(scanner.assess("q", "reference", candidate)["decision"], "review")


if __name__ == "__main__":
    unittest.main()
