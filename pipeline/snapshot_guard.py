"""Reject answers whose exact source text is absent from a pinned clean snapshot."""

from __future__ import annotations

from typing import Any

from generation.grounded_answer import answer_from_accepted
from generation.local_llm import ABSTENTION
from security.integrity import IntegrityManifest


def guard_answer(
    answer: dict[str, Any], accepted_documents: list[dict], snapshot: IntegrityManifest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check source identity and cited span; never infer truth from provenance."""
    if not snapshot.corpus_sha256 or snapshot.legacy:
        raise ValueError("A versioned clean-snapshot manifest is required")
    source_id = answer.get("source_doc_id")
    span = answer.get("evidence_span")
    trace: dict[str, Any] = {
        "enforced": True, "snapshot_corpus_sha256": snapshot.corpus_sha256,
        "selected_source_id": source_id, "source_status": None,
        "source_sha256": None, "expected_sha256": None,
        "span_verified": False, "citation_verified": False,
        "decision": "abstain", "reason": None,
    }
    if source_id is None or not isinstance(span, str) or not span:
        trace["reason"] = "upstream_abstained"
        return answer, trace
    matches = [doc for doc in accepted_documents if int(doc["doc_id"]) == int(source_id)]
    if len(matches) != 1:
        trace["reason"] = ("selected_source_not_accepted" if not matches
                           else "duplicate_selected_source_id")
    else:
        source = matches[0]
        check = snapshot.check(source_id, str(source["text"]))
        trace["source_status"] = check["status"]
        trace["source_sha256"] = check["actual_hash"]
        trace["expected_sha256"] = check["expected_hash"]
        trace["span_verified"] = span in str(source["text"])
        trace["citation_verified"] = any(
            str(item.get("doc_id")) == str(source_id) and item.get("span") == span
            for item in answer.get("citations", [])
        )
        if check["status"] == "verified" and trace["span_verified"] and trace["citation_verified"]:
            trace["decision"] = "allow"
            trace["reason"] = "pinned_source_and_span_verified"
            return answer, trace
        trace["reason"] = (
            "source_not_in_pinned_snapshot" if check["status"] == "unknown_document"
            else "source_differs_from_pinned_snapshot" if check["status"] == "tampered"
            else "cited_span_missing_from_verified_source" if not trace["span_verified"]
            else "citation_missing_or_mismatched"
        )
    return {
        "answer": ABSTENTION, "source_doc_id": None, "evidence_span": None,
        "citations": [], "backend": answer.get("backend", "unknown"),
        "generation_status": "snapshot_guard_abstained",
    }, trace


def guard_with_recovery(
    query: str, answer: dict[str, Any], accepted_documents: list[dict],
    snapshot: IntegrityManifest, snapshot_retriever,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Use question-only pinned-corpus retrieval after the primary source fails."""
    if getattr(snapshot_retriever, "corpus_digest", None) != snapshot.corpus_sha256:
        raise ValueError("Recovery retriever is not built from the pinned snapshot")
    guarded, trace = guard_answer(answer, accepted_documents, snapshot)
    snapshot_candidates = snapshot_retriever.retrieve(query, top_k=5)
    verified_candidates = [doc for doc in snapshot_candidates
                           if snapshot.check(doc["doc_id"], str(doc["text"]))["status"] == "verified"]
    independent = answer_from_accepted(query, verified_candidates, snapshot_retriever)
    trace["independent_top_five_ids"] = [int(doc["doc_id"]) for doc in snapshot_candidates]
    trace["independent_verified_ids"] = [int(doc["doc_id"]) for doc in verified_candidates]
    if trace["decision"] == "allow":
        trace["recovery_attempted"] = False
        trace["recovery_source_id"] = independent["source_doc_id"]
        return guarded, trace, independent
    recovered, recovery_trace = guard_answer(independent, verified_candidates, snapshot)
    trace["recovery_attempted"] = True
    trace["recovery_source_id"] = recovery_trace["selected_source_id"]
    trace["recovery_source_status"] = recovery_trace["source_status"]
    trace["primary_reason"] = trace["reason"]
    if recovery_trace["decision"] == "allow":
        trace["decision"] = "recover"
        trace["reason"] = "independent_snapshot_answer_verified"
        return recovered, trace, independent
    trace["reason"] = "independent_snapshot_answer_unavailable"
    trace["recovery_reason"] = recovery_trace["reason"]
    return guarded, trace, independent
