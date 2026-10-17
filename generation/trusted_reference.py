"""Answer only from a separately reviewed, hash-verified source passage.

An origin attestation proves that a passage came from the operator-reviewed
corpus snapshot. It does not prove that the passage is factually true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import pipeline

from security.provenance import SourceAttestations


ABSTENTION = "Insufficient trusted evidence to answer this question."


@dataclass(frozen=True)
class ExtractedAnswer:
    answer: str
    score: float
    start: int
    end: int


class ReferenceReader:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.qa = pipeline(
            "question-answering", model=str(model_path), tokenizer=str(model_path),
            device=0 if torch.cuda.is_available() else -1,
        )

    def extract_batch(self, questions: Sequence[str], contexts: Sequence[str],
                      *, batch_size: int = 16) -> list[ExtractedAnswer]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have equal lengths")
        if not questions:
            return []
        rows = [{"question": question, "context": context}
                for question, context in zip(questions, contexts)]
        predictions = self.qa(rows, batch_size=batch_size,
                              handle_impossible_answer=False,
                              max_seq_len=384, doc_stride=128)
        return [ExtractedAnswer(str(row["answer"]), float(row["score"]),
                                int(row["start"]), int(row["end"]))
                for row in predictions]

    def extract(self, question: str, context: str) -> ExtractedAnswer:
        return self.extract_batch([question], [context])[0]


def verified_reference(documents: list[dict], attestations: SourceAttestations,
                       trusted_origins: set[str]) -> dict | None:
    """Select the first attested document from an explicitly trusted origin."""
    if not trusted_origins:
        return None
    for doc in documents:
        origin = attestations.origin_for(doc)
        if origin in trusted_origins:
            return doc
    return None


def answer_from_verified_reference(
    query: str, documents: list[dict], attestations: SourceAttestations,
    trusted_origins: set[str], reader: ReferenceReader, *, min_score: float,
) -> dict:
    """Fail closed if the verified source or an adequate answer span is absent."""
    if not query.strip() or not 0 <= min_score <= 1:
        raise ValueError("Invalid query or score threshold")
    reference = verified_reference(documents, attestations, trusted_origins)
    if reference is None:
        return {"answer": ABSTENTION, "source_doc_id": None,
                "reference_status": "verified_reference_missing", "score": None,
                "evidence_span": None}
    prediction = reader.extract(query, str(reference["text"]))
    if not prediction.answer.strip() or prediction.score < min_score:
        return {"answer": ABSTENTION, "source_doc_id": None,
                "reference_status": "reader_abstained", "score": prediction.score,
                "evidence_span": None}
    # The reader is extractive, but verify exact support before returning it.
    span = str(reference["text"])[prediction.start:prediction.end]
    if span != prediction.answer:
        return {"answer": ABSTENTION, "source_doc_id": None,
                "reference_status": "invalid_reader_span", "score": prediction.score,
                "evidence_span": None}
    return {"answer": prediction.answer, "source_doc_id": reference["doc_id"],
            "reference_status": "verified_reference_answer", "score": prediction.score,
            "evidence_span": span}


def normalized_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def answer_overlap(first: str, second: str) -> bool:
    """Conservative lexical agreement; it does not establish equivalence."""
    left, right = normalized_answer(first), normalized_answer(second)
    return bool(left and right) and (
        f" {left} " in f" {right} " or f" {right} " in f" {left} "
    )
