"""One auditable inference path for the bounded research workbench.

This module only consumes a retriever, a previously sealed integrity manifest,
and a frozen detector. It never trains or seals evidence during inference.
"""

from __future__ import annotations

import time
from typing import Any

from detect.research_detector import ResearchDetector
from generation.grounded_answer import answer_from_accepted
from generation.local_llm import ABSTENTION
from security.integrity import IntegrityManifest
from security.provenance import SourceAttestations, corroborate_exact_span


def _answer(query: str, documents: list[dict], retriever) -> dict[str, Any]:
    return answer_from_accepted(query, documents, retriever)


def _apply_origin_policy(
    answer: dict[str, Any], documents: list[dict], attestations: SourceAttestations | None,
    *, required: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance: dict[str, Any] = {"enforced": required, "supported": None,
                                  "reason": "policy_disabled"}
    if not required:
        return answer, provenance
    if attestations is None:
        provenance = {"enforced": True, "supported": False,
                      "reason": "source_attestation_missing"}
    elif answer["source_doc_id"] is None or not answer["evidence_span"]:
        provenance = {"enforced": True, "supported": False,
                      "reason": "no_cited_answer_span"}
    else:
        provenance = {"enforced": True, **corroborate_exact_span(
            answer["evidence_span"], answer["source_doc_id"], documents, attestations,
        )}
    if provenance["supported"]:
        return answer, provenance
    return {
        "answer": ABSTENTION, "source_doc_id": None, "evidence_span": None,
        "citations": [], "backend": answer["backend"],
        "generation_status": "provenance_abstained",
    }, provenance


def infer_research(
    query: str, retriever, manifest: IntegrityManifest | None,
    detector: ResearchDetector, *, top_k: int = 5,
    source_attestations: SourceAttestations | None = None,
    require_independent_origins: bool = False,
) -> dict[str, Any]:
    """Retrieve, verify, score, filter, answer, and audit the same candidate set."""
    if not query.strip():
        raise ValueError("Query must not be empty")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    started = time.perf_counter()
    candidates = retriever.retrieve(query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - started) * 1000
    checked = []
    for doc in candidates:
        status = manifest.check(doc["doc_id"], doc["text"]) if manifest else {
            "status": "manifest_missing", "actual_hash": None, "expected_hash": None,
        }
        checked.append((doc, status))
    integrity_ms = (time.perf_counter() - started) * 1000 - retrieval_ms
    rows, kept = [], []
    for rank, (doc, integrity) in enumerate(checked, 1):
        score = detector.score(query, doc["text"])
        quarantine = integrity["status"] != "verified" or score["decision"] == "quarantine"
        if not quarantine:
            kept.append(doc)
        rows.append({
            "rank": rank, "doc_id": int(doc["doc_id"]), "text": doc["text"],
            "bm25_rank": doc["bm25_rank"], "dense_rank": doc["dense_rank"],
            "rrf_score": doc["score"], "integrity": integrity["status"],
            "sha256": integrity["actual_hash"], "risk": score["risk_score"],
            "threshold": score["threshold"], "features": score["features"],
            "decision": "quarantine" if quarantine else "accept",
            "reasons": ([f"integrity:{integrity['status']}"] if integrity["status"] != "verified" else [])
            + (["detector:threshold"] if score["decision"] == "quarantine" else []),
        })
    detection_ms = (time.perf_counter() - started) * 1000 - retrieval_ms - integrity_ms
    undefended = _answer(query, candidates, retriever)
    defended, provenance = _apply_origin_policy(
        _answer(query, kept, retriever), kept, source_attestations,
        required=require_independent_origins,
    )
    # Counterfactuals operate on the accepted batch and never change the gate.
    # A changed answer is evidence of influence, not proof of poisoning.
    counterfactuals = []
    for doc in kept:
        remaining = [item for item in kept if item["doc_id"] != doc["doc_id"]]
        without, _ = _apply_origin_policy(
            _answer(query, remaining, retriever), remaining, source_attestations,
            required=require_independent_origins,
        )
        counterfactuals.append({
            "removed_doc_id": int(doc["doc_id"]),
            "answer_changed": without["answer"] != defended["answer"],
            "answer_after_removal": without["answer"],
            "source_after_removal": without["source_doc_id"],
        })
    answer_ms = (time.perf_counter() - started) * 1000 - retrieval_ms - integrity_ms - detection_ms
    timings = {
        "retrieval_ms": round(retrieval_ms, 2),
        "integrity_ms": round(integrity_ms, 2),
        "detection_ms": round(detection_ms, 2),
        "answer_and_loo_ms": round(answer_ms, 2),
    }
    return {
        "query": query, "documents": rows, "undefended": undefended,
        "defended": defended, "counterfactuals": counterfactuals,
        "provenance": provenance,
        "timings": timings, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "retrieval_backend": {
            "dense": retriever.embedder.model_name, "sparse": "BM25", "fusion": "RRF(k=60)",
        },
        "audit": [
            {"stage": "retrieval", "candidate_ids": [int(doc["doc_id"]) for doc in candidates]},
            {"stage": "integrity", "statuses": {str(row["doc_id"]): row["integrity"] for row in rows}},
            {"stage": "detection", "accepted_ids": [int(doc["doc_id"]) for doc in kept],
             "quarantined_ids": [row["doc_id"] for row in rows if row["decision"] == "quarantine"]},
            {"stage": "provenance", "enforced": require_independent_origins,
             "supported": provenance["supported"], "reason": provenance["reason"]},
            {"stage": "answer", "source_doc_id": defended["source_doc_id"],
             "abstained": defended["source_doc_id"] is None},
        ],
    }
