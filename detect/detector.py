"""Runtime interface for the trained Review-1 fusion detector."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from config import ARTIFACTS_DIR
from detect.signals import behavioural_signal_components, isolation_forest_score, mahalanobis_score

class FusionDetector:
    """Score retrieved documents and expose the evidence behind each decision."""

    def __init__(self, embedder: Any, model_path: str | Path | None = None) -> None:
        self.embedder = embedder
        backend = getattr(getattr(embedder, "embedder", embedder), "backend", "hashing")
        path = Path(model_path) if model_path else ARTIFACTS_DIR / f"fusion_classifier_{backend}.joblib"
        if not path.exists():
            path = ARTIFACTS_DIR / "fusion_classifier.joblib"
        if not path.exists():
            raise FileNotFoundError("Fusion model is missing. Run: python -m detect.train_fusion_classifier")
        self.bundle = joblib.load(path)

    def score(self, query: str, doc_text: str, dense_rank_normalized: float,
              counterfactual_influence: float = 0.0) -> dict[str, Any]:
        embedding = self.embedder.encode(doc_text)[0]
        behaviour = behavioural_signal_components(query, doc_text, self.embedder)
        mahal = mahalanobis_score(embedding, self.bundle["clean_mean"], self.bundle["clean_cov_inv"])
        isolation = isolation_forest_score(embedding, self.bundle["isolation_forest"])
        raw = np.asarray([[mahal, isolation, behaviour["relevance_spike"],
                           behaviour["instruction"], behaviour["url"],
                           behaviour["authority"], dense_rank_normalized]], dtype=np.float64)
        model_probability = float(self.bundle["classifier"].predict_proba(
            self.bundle["scaler"].transform(raw))[0, 1])
        # High-confidence behavioural evidence is retained as a safety floor. It
        # does not depend on a known document ID or label at inference time.
        behavioural_floor = max(
            behaviour["instruction"] * 0.94,
            behaviour["url"] * 0.88,
            behaviour["authority"] * 0.84,
        )
        mahal_ratio = mahal / max(self.bundle["mahal_p95"], 1e-6)
        # S1 is risk above the clean 95th-percentile boundary, not the raw
        # distance divided by that boundary (which made normal documents look
        # almost 100% anomalous in the UI).
        s1_geometry = float(np.clip((mahal_ratio - 1.0) / 0.18, 0.0, 1.0))
        s4_stability = float(np.clip(counterfactual_influence, 0.0, 1.0))
        s1_floor = s1_geometry * 0.90
        s4_floor = s4_stability * 0.92
        probability = float(np.clip(max(
            model_probability, behavioural_floor, s1_floor, s4_floor), 0.0, 1.0))
        reasons: list[str] = []
        if behaviour["instruction"] >= 0.5: reasons.append("Embedded instruction language")
        if behaviour["url"] > 0: reasons.append("External URL inside retrieved context")
        if behaviour["authority"] >= 0.5: reasons.append("Manufactured authority or correction cue")
        if s1_geometry >= 0.5: reasons.append("S1 geometry probe: outside clean embedding boundary")
        if behaviour["relevance_spike"] >= 0.35: reasons.append("Single-sentence relevance spike")
        if s4_stability >= 0.35: reasons.append("S4 counterfactual probe: answer changes under ablation")
        if not reasons: reasons.append("No high-confidence attack indicators")
        return {
            "probability": probability,
            "model_probability": model_probability,
            "signals": {
                "mahalanobis": s1_geometry,
                "mahalanobis_raw": float(mahal),
                "isolation_forest": isolation,
                "relevance_spike": behaviour["relevance_spike"],
                "instruction_pattern": behaviour["instruction"],
                "url_pattern": behaviour["url"],
                "authority_cue": behaviour["authority"],
                "counterfactual_influence": s4_stability,
            },
            "reasons": reasons,
        }
