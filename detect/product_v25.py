"""Reference-adaptive V25 ensemble for source-label review.

The rule was chosen on inspected PoisonArena and SciFact development data.
It combines the trained V3 behavioral detector, V24 learned fusion, source
containment, and optional authenticated document identity. The supplied
reference still requires external authentication and is not a truth oracle.
"""

from __future__ import annotations

from pathlib import Path

from detect.product_v22 import PoisonDocumentScanner


V3_THRESHOLD = 0.75
FUSION_THRESHOLD = 4.272025739459554  # Frozen V24 clean-calibration threshold.
REFERENCE_MARGIN = 0.5


class ReferenceAdaptiveScanner:
    def __init__(self, *, root: Path = Path(".")) -> None:
        self.v24 = PoisonDocumentScanner(root=root, integrity_mode="containment")
        if self.v24.calibration["fusion_threshold_1pct"] != FUSION_THRESHOLD:
            raise ValueError("V24 clean-calibration threshold changed")
        self._reference_scores: dict[tuple[str, str], float] = {}

    def close(self) -> None:
        self.v24.close()

    def _reference_fusion(self, query: str, reference: str) -> float:
        key = (query, reference)
        if key not in self._reference_scores:
            self._reference_scores[key] = self.v24.assess(
                query, reference, reference)["learned"]["fusion_score"]
        return self._reference_scores[key]

    def assess(self, query: str, reference: str, candidate: str, *,
               expected_sha256: str | None = None) -> dict:
        baseline = self.v24.assess(query, reference, candidate,
                                   expected_sha256=expected_sha256)
        candidate_score = baseline["learned"]["fusion_score"]
        reference_score = self._reference_fusion(query, reference)
        v3_risk = baseline["learned"]["v3_risk"]
        v3_flag = v3_risk >= V3_THRESHOLD
        relative_flag = (candidate_score >= FUSION_THRESHOLD
                         and candidate_score - reference_score >= REFERENCE_MARGIN)
        containment_flag = baseline["integrity"].get("flag", False)
        identity_flag = baseline["identity"]["flag"]
        decision = "review" if any((v3_flag, relative_flag,
                                    containment_flag, identity_flag)) else "pass"
        return {
            **baseline, "decision": decision, "version": "v25-development",
            "v3": {"risk": v3_risk, "threshold": V3_THRESHOLD, "flag": v3_flag},
            "reference_adaptive": {
                "candidate_fusion_score": candidate_score,
                "reference_fusion_score": reference_score,
                "margin": candidate_score - reference_score,
                "required_margin": REFERENCE_MARGIN,
                "fusion_threshold": FUSION_THRESHOLD,
                "flag": relative_flag,
            },
            "scope": "source_label_review; supplied_reference_must_be_authenticated",
        }
