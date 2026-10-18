"""Reference-conditioned document scorer for an offline poisoning experiment.

Model labels are source labels: judged reference, clean other passage, and
published attack passage. The score is P(published attack), not a proof of
malice or an entailment probability. Runtime requires a verified reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from security.provenance import SourceAttestations


LABELS = ("reference", "clean_other", "poison")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def token_windows(tokenizer, query: str, reference: str, candidate: str) -> list[dict]:
    if not query.strip() or not reference.strip() or not candidate.strip():
        raise ValueError("Question, verified reference and candidate are required")
    query_ids = tokenizer.encode(query, add_special_tokens=False)[:48]
    reference_ids = tokenizer.encode(reference, add_special_tokens=False)[:190]
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    # Keep the question and reference fixed; examine the complete candidate.
    prefix = tokenizer.decode(query_ids, skip_special_tokens=True)
    first = prefix + " Trusted reference: " + tokenizer.decode(
        reference_ids, skip_special_tokens=True)
    windows = []
    for start in range(0, len(candidate_ids), 200):
        chunk = tokenizer.decode(candidate_ids[start:start + 250], skip_special_tokens=True)
        windows.append(tokenizer(first, chunk, truncation="only_second",
                                 max_length=512, padding=False))
        if start + 250 >= len(candidate_ids):
            break
    return windows


@dataclass
class ReferenceConditionedScorer:
    model_path: Path
    device: torch.device

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                        local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, local_files_only=True).to(self.device).eval()
        labels = {int(index): label for index, label in self.model.config.id2label.items()}
        if labels != dict(enumerate(LABELS)):
            raise ValueError("Unexpected reference-detector labels")

    @torch.inference_mode()
    def score(self, rows: Sequence[dict], *, batch_size: int = 32) -> np.ndarray:
        windows = []
        owners = []
        for index, row in enumerate(rows):
            parts = token_windows(self.tokenizer, row["query"], row["reference"],
                                  row["candidate"])
            windows.extend(parts)
            owners.extend([index] * len(parts))
        scores = np.zeros(len(rows), dtype=np.float64)
        for start in range(0, len(windows), batch_size):
            batch = self.tokenizer.pad(windows[start:start + batch_size],
                                       return_tensors="pt").to(self.device)
            probability = torch.softmax(self.model(**batch).logits.float(), dim=1)
            values = probability[:, LABEL_TO_ID["poison"]].cpu().numpy()
            for owner, value in zip(owners[start:start + batch_size], values):
                scores[owner] = max(scores[owner], float(value))
        return scores


def assess_attested_candidates(query: str, documents: list[dict],
                               attestations: SourceAttestations,
                               trusted_origins: set[str],
                               scorer: ReferenceConditionedScorer,
                               *, threshold: float,
                               reference_doc_id: str | int | None) -> list[dict]:
    """Score only when a matching, operator-attested trusted source is present.

    Caller supplies a separately selected reference ID; this function verifies
    its exact hash and trusted origin. It cannot decide source relevance. An
    unassessable document is never silently labelled clean. This research
    interface does not quarantine documents or change the live answer path.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Invalid threshold")
    if not trusted_origins or reference_doc_id is None:
        return [{"doc_id": doc["doc_id"], "status": "unassessable_no_selected_trusted_reference",
                 "risk_score": None, "flag": None} for doc in documents]
    matches = [doc for doc in documents if str(doc["doc_id"]) == str(reference_doc_id)]
    if len(matches) != 1:
        return [{"doc_id": doc["doc_id"], "status": "unassessable_reference_id_invalid",
                 "risk_score": None, "flag": None} for doc in documents]
    reference = matches[0] if attestations.origin_for(matches[0]) in trusted_origins else None
    if reference is None:
        return [{"doc_id": doc["doc_id"], "status": "unassessable_no_verified_reference",
                 "risk_score": None, "flag": None} for doc in documents]
    rows = [{"query": query, "reference": str(reference["text"]),
             "candidate": str(doc["text"])} for doc in documents]
    scores = scorer.score(rows)
    return [{"doc_id": doc["doc_id"], "reference_doc_id": reference["doc_id"],
             "status": "scored_against_verified_reference", "risk_score": float(score),
             "flag": bool(score >= threshold)}
            for doc, score in zip(documents, scores)]
