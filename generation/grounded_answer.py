"""Source-span answers with a verified local-model option and safe fallback."""

from __future__ import annotations

from collections.abc import Callable

from generation.local_llm import ABSTENTION, CITATION_RE, CITATION_TAIL_RE, _is_source_span, generate_cited_answer
from pipeline.secure_rag import _select_answer


def answer_from_accepted(
    query: str, documents: list[dict], retriever, *,
    model_enabled: bool = False,
    model_answerer: Callable[[str, list[dict]], tuple[str, int | None]] = generate_cited_answer,
) -> dict:
    """Return a cited exact span or abstain; never expose an invalid model answer.

    The fallback uses the same accepted documents and extractive selector as the
    bounded demo. A citation proves the span exists in that document, not truth.
    """
    if not query.strip():
        raise ValueError("Query must not be empty")
    answer, source_id, evidence = _select_answer(query, documents, retriever)
    result = {
        "answer": answer, "source_doc_id": source_id, "evidence_span": evidence,
        "citations": ([{"doc_id": source_id, "span": evidence}]
                      if source_id is not None and evidence else []),
        "backend": "hashing_extractive" if retriever.embedder.backend == "hashing" else "minilm_extractive",
        "generation_status": "extractive_selected" if source_id is not None else "extractive_abstained",
    }
    if not model_enabled or not documents:
        return result
    try:
        candidate, candidate_source = model_answerer(query, documents)
    except (OSError, RuntimeError, ValueError, ImportError):
        result["generation_status"] = "model_unavailable_extractive_fallback"
        return result
    tail = CITATION_TAIL_RE.search(candidate)
    span = candidate[:tail.start()].strip() if tail else ""
    cited_ids = [int(value) for value in CITATION_RE.findall(tail.group())] if tail else []
    by_id = {int(doc["doc_id"]): doc["text"] for doc in documents}
    if (candidate == ABSTENTION or candidate_source is None or not cited_ids
            or candidate_source != cited_ids[0]
            or any(doc_id not in by_id or not _is_source_span(span, by_id[doc_id])
                   for doc_id in cited_ids)):
        result["generation_status"] = "model_invalid_extractive_fallback"
        return result
    return {
        "answer": candidate, "source_doc_id": candidate_source,
        "evidence_span": span,
        "citations": [{"doc_id": doc_id, "span": span} for doc_id in cited_ids],
        "backend": "local_llm_verified_span", "generation_status": "model_citation_verified",
    }
