"""Versioned factored actions and masks for the document-edit environment."""

from __future__ import annotations

from dataclasses import dataclass

OPERATIONS = ("INSERT", "PARAPHRASE", "SYNONYM", "DELETE", "STOP")
POSITIONS = ("start", "middle", "end")
PAYLOADS = ("plain_false_answer", "query_matched_answer", "authority_claim")


@dataclass(frozen=True)
class EditAction:
    operation: str
    position: int = 0
    payload: int = 0

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(f"Unknown edit operation: {self.operation}")
        if not 0 <= self.position < len(POSITIONS):
            raise ValueError("Position index is out of range")
        if not 0 <= self.payload < len(PAYLOADS):
            raise ValueError("Payload index is out of range")
