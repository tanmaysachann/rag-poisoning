"""Versioned audit events for the staged secure-RAG inference path.

Events are plain JSON-serializable dicts with a schema version, stage name,
and payload. The live inference path keeps them in memory inside its trace;
this module only builds and validates them so consoles and verifiers share
one contract. It never writes files or touches model weights.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1

_STAGES = (
    "retrieval",
    "integrity",
    "detection",
    "provenance",
    "corroboration",
    "answer",
)


def build_event(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    if stage not in _STAGES:
        raise ValueError(f"Unknown audit stage: {stage}")
    if not isinstance(payload, dict):
        raise ValueError("Audit payload must be a dict")
    return {"schema_version": SCHEMA_VERSION, "stage": stage, "payload": dict(payload)}


def validate_event(event: dict[str, Any]) -> None:
    if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported audit event schema")
    if event.get("stage") not in _STAGES or not isinstance(event.get("payload"), dict):
        raise ValueError("Malformed audit event")


def validate_trace(trace: list[dict[str, Any]]) -> None:
    if not isinstance(trace, list) or not trace:
        raise ValueError("Audit trace must be a nonempty list")
    seen = set()
    for event in trace:
        validate_event(event)
        seen.add(event["stage"])
    if "retrieval" not in seen or "answer" not in seen:
        raise ValueError("Audit trace must include retrieval and answer stages")
