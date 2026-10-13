"""Answer-key containment gate for two-passage development data.

Unlike sentence-similarity gates (cosine, token-F1), which drown the answer
in shared context words, this gate checks the discriminative part: every
content token of the selected answer must occur somewhere in the peer
passage's full text. A substituted wrong answer carries distinctive tokens
from another query's answer pool, so it fails containment and abstains. The
gate never fails over to the peer.

Two MS MARCO passages from one query row do not prove independent
publication origin or truth. This is a bounded robustness probe.
"""

from __future__ import annotations

import re

ABSTENTION_PREFIX = "Insufficient relevant evidence"

STOPWORDS = frozenset({
    "what", "which", "who", "where", "when", "how", "the", "is", "are",
    "was", "were", "a", "an", "of", "to", "in", "on", "at", "and", "or",
    "for", "as", "by", "with", "it", "its", "this", "that", "these",
    "those", "be", "been", "are", "from", "into", "than", "then",
})


def is_abstention(answer: str) -> bool:
    return answer.strip().startswith(ABSTENTION_PREFIX)


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold(), re.UNICODE)


def content_tokens(answer: str) -> list[str]:
    return [token for token in _tokens(answer) if token not in STOPWORDS]


def key_supported(answer: str, peer_text: str, *, content_only: bool) -> tuple[bool, list[str]]:
    """Check the answer key against the full peer passage text.

    Returns (supported, missing_tokens). Strict mode uses every answer token;
    content-only mode drops stopwords first.
    """
    key = content_tokens(answer) if content_only else _tokens(answer)
    if not key:
        return False, []
    peer_words = set(_tokens(peer_text))
    missing = [token for token in dict.fromkeys(key) if token not in peer_words]
    return (not missing), missing


def containment_decision(
    selected_answer: str,
    selected_source: int | None,
    peer_text: str | None,
    peer_doc_id: int | None,
    *,
    content_only: bool,
) -> dict:
    if selected_source is None or is_abstention(selected_answer):
        return {"gate_keeps_answer": True, "reason": "selected_abstained_no_answer",
                "missing_tokens": []}
    if peer_doc_id is None or peer_text is None:
        return {"gate_keeps_answer": False, "reason": "missing_peer",
                "missing_tokens": []}
    supported, missing = key_supported(selected_answer, peer_text, content_only=content_only)
    if supported:
        return {"gate_keeps_answer": True, "reason": "peer_passage_contains_answer_key",
                "missing_tokens": []}
    return {"gate_keeps_answer": False, "reason": "answer_key_absent_from_peer",
            "missing_tokens": missing}
