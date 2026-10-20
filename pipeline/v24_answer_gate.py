"""Apply V24 and authenticated source admission before extractive answering.

The caller supplies a reference from a pre-attack, operator-pinned source.
This module verifies that reference against its independent digest and passes
known document hashes from the same source. A new inserted document has no
known hash; its scanner score is reported, but it cannot supply an answer
until the operator admits it to an authenticated corpus snapshot.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Callable, Mapping

from generation.grounded_answer import answer_from_accepted
from pipeline.snapshot_guard import guard_with_recovery


def answer_with_v24(
    query: str, candidates: list[dict], retriever, *,
    pinned_reference: str, pinned_reference_sha256: str,
    expected_hashes: Mapping[str, str],
    scan: Callable[[str, str, str, str | None], dict],
    snapshot=None, snapshot_retriever=None,
) -> dict:
    """Answer only from scanner-passing, previously admitted source bytes."""
    if not query.strip() or not pinned_reference.strip():
        raise ValueError("A question and pinned reference are required")
    actual_reference_hash = hashlib.sha256(pinned_reference.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(actual_reference_hash, pinned_reference_sha256.lower()):
        raise ValueError("Reference differs from the pinned pre-attack source")
    started = time.perf_counter()
    undefended = answer_from_accepted(query, candidates, retriever)
    kept = []
    decisions = []
    seen_ids = set()
    for rank, candidate in enumerate(candidates, 1):
        doc_id = str(candidate["doc_id"])
        if doc_id in seen_ids:
            raise ValueError("Duplicate retrieved document ID")
        seen_ids.add(doc_id)
        expected = expected_hashes.get(doc_id)
        result = scan(query, pinned_reference, candidate["text"], expected)
        if result.get("decision") not in {"pass", "review"}:
            raise ValueError("V24 scanner returned an invalid decision")
        actual = hashlib.sha256(candidate["text"].encode("utf-8")).hexdigest()
        admitted = expected is not None and hmac.compare_digest(actual, expected.lower())
        decision = "pass" if result["decision"] == "pass" and admitted else "review"
        if decision == "pass":
            kept.append(candidate)
        decisions.append({
            "rank": rank, "doc_id": candidate["doc_id"],
            "decision": decision,
            "scanner_decision": result["decision"],
            "source_admitted": admitted,
            "admission_reason": ("known_source_hash_match" if admitted else
                                 "unknown_source" if expected is None else "source_hash_mismatch"),
            "known_source_id": expected is not None,
            "learned_flag": result["learned"]["flag"],
            "containment_flag": result["integrity"].get("flag", False),
            "identity_status": result["identity"]["status"],
            "identity_flag": result["identity"]["flag"],
        })
    screened = answer_from_accepted(query, kept, retriever)
    if (snapshot is None) != (snapshot_retriever is None):
        raise ValueError("Snapshot and recovery retriever must be supplied together")
    if snapshot is not None:
        defended, recovery, _ = guard_with_recovery(
            query, screened, kept, snapshot, snapshot_retriever)
    else:
        defended, recovery = screened, {"enforced": False, "decision": "not_run"}
    if defended["source_doc_id"] is not None and str(defended["source_doc_id"]) not in {
            str(doc["doc_id"]) for doc in kept} and recovery["decision"] != "recover":
        raise RuntimeError("Answer source is neither screened nor independently recovered")
    return {
        "undefended": undefended, "screened": screened,
        "defended": defended, "snapshot_recovery": recovery,
        "documents": decisions,
        "accepted_doc_ids": [doc["doc_id"] for doc in kept],
        "review_doc_ids": [row["doc_id"] for row in decisions if row["decision"] == "review"],
        "reference_sha256": actual_reference_hash,
        "reference_source": "operator_pinned_pre_attack_snapshot",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "scope": "research_answer_path; authenticated_source_admission_not_truth_verification",
    }
