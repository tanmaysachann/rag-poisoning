"""Experimental source-span repair for uncited local-model answers.

This is a development diagnostic. A matching source phrase proves lexical
grounding only; it does not prove that the phrase answers the question.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from generation.local_llm import ABSTENTION, CITATION_RE, _answer_span, _is_source_span


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
COMMON = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
          "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with"}


def _terms(value: str) -> set[str]:
    return {match.group().casefold() for match in TOKEN_RE.finditer(value)} - COMMON


def repair_model_answer(
    query: str, raw_output: str, documents: list[dict], *, minimum_coverage: float = 0.4,
) -> tuple[str, int | None, str]:
    """Use the longest source-aligned token run if it contains new content."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    raw = _answer_span(raw_output)
    if (not raw or raw.casefold().startswith("insufficient evidence")
            or CITATION_RE.search(raw) or "[DOC" in raw.upper() or "<" in raw or ">" in raw):
        return ABSTENTION, None, "rejected_output"
    raw_tokens = list(TOKEN_RE.finditer(raw))
    if not raw_tokens:
        return ABSTENTION, None, "no_tokens"
    raw_words = [item.group().casefold() for item in raw_tokens]
    query_terms = _terms(query)
    choices = []
    for rank, doc in enumerate(documents):
        source = str(doc["text"])
        source_tokens = list(TOKEN_RE.finditer(source))
        source_words = [item.group().casefold() for item in source_tokens]
        block = SequenceMatcher(None, raw_words, source_words, autojunk=False).find_longest_match(
            0, len(raw_words), 0, len(source_words)
        )
        if not block.size:
            continue
        span = source[source_tokens[block.b].start():source_tokens[block.b + block.size - 1].end()]
        new_terms = _terms(span) - query_terms
        if not new_terms or (block.size > 1 and len(new_terms) < 2):
            continue
        if block.size / len(raw_words) < minimum_coverage:
            continue
        if block.size < 3 and len(raw_words) > 1:
            continue
        if not _is_source_span(span, source):
            continue
        choices.append((block.size, len(new_terms), -rank, span, int(doc["doc_id"])))
    if not choices:
        return ABSTENTION, None, "no_qualified_source_span"
    _, _, _, span, doc_id = max(choices)
    exact = _is_source_span(raw, next(doc["text"] for doc in documents if int(doc["doc_id"]) == doc_id))
    return f"{span} [DOC {doc_id}]", doc_id, "exact" if exact else "aligned_phrase"
