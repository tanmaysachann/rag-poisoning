"""V26 source-label scanner: frozen V25 plus trained role-boundary signal."""

from __future__ import annotations

from pathlib import Path

import joblib

from detect.product_v25 import ReferenceAdaptiveScanner
from detect.role_boundary_v26 import features


class RoleAdaptiveScanner:
    def __init__(self, *, root: Path = Path(".")) -> None:
        payload = joblib.load(root / "artifacts/role_boundary_v26.joblib")
        self.role_model = payload["model"]
        self.role_threshold = float(payload["threshold"])
        self.v25 = ReferenceAdaptiveScanner(root=root)

    def close(self) -> None:
        self.v25.close()

    def assess(self, query: str, reference: str, candidate: str, *,
               expected_sha256: str | None = None) -> dict:
        baseline = self.v25.assess(query, reference, candidate,
                                   expected_sha256=expected_sha256)
        signal = features(reference, candidate)
        score = float(self.role_model.predict_proba([signal])[0, 1])
        flag = bool(signal["system_marker"] and score >= self.role_threshold)
        return {
            **baseline, "version": "v26-development",
            "decision": "review" if flag or baseline["decision"] == "review" else "pass",
            "role_boundary": {"risk": score, "threshold": self.role_threshold,
                              "system_marker": bool(signal["system_marker"]), "flag": flag,
                              "training": "NFCorpus grouped development, no FiQA questions"},
        }
