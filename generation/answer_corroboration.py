"""Answer-level corroboration for two-passage development data.

The gate compares independently selected answers from a selected document
and its query-group peer. On disagreement it abstains instead of failing
over to the peer, because failover can select a poisoned peer.

Two MS MARCO passages from one query row do not prove independent
publication origin or truth. This is a bounded robustness probe, not a
provenance solution.
"""

from __future__ import annotations

import re

ABSTENTION_PREFIX = "Insufficient relevant evidence"


def is_abstention(answer: str) -> bool:
    return answer.strip().startswith(ABSTENTION_PREFIX)


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold(), re.UNICODE)


def token_f1(first: str, second: str) -> float:
    """Symmetric unigram F1 between two answer strings."""
    first_tokens, second_tokens = _tokens(first), _tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in second_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in first_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(first_tokens)
    recall = overlap / len(second_tokens)
    return 2 * precision * recall / (precision + recall)


def corroboration_decision(
    selected_answer: str,
    selected_source: int | None,
    peer_answer: str | None,
    peer_doc_id: int | None,
    token_f1_threshold: float,
    *,
    fail_closed_on_missing_peer: bool = True,
) -> dict:
    """Decide whether to keep the selected answer or abstain.

    Missing peer or peer abstention has no usable agreement score. When
    ``fail_closed_on_missing_peer`` is True the gate abstains in that case;
    otherwise it keeps the selected answer. A selected abstention is always
    kept because no wrong answer is emitted.
    """
    if not 0.0 <= token_f1_threshold <= 1.0:
        raise ValueError("token_f1_threshold must be in [0, 1]")
    if selected_source is None or is_abstention(selected_answer):
        return {
            "gate_keeps_answer": True,
            "reason": "selected_abstained_no_answer",
            "token_f1": None,
        }
    if peer_doc_id is None or peer_answer is None or is_abstention(peer_answer):
        return {
            "gate_keeps_answer": not fail_closed_on_missing_peer,
            "reason": "missing_peer_or_peer_abstained",
            "token_f1": None,
        }
    score = token_f1(selected_answer, peer_answer)
    if score >= token_f1_threshold:
        return {"gate_keeps_answer": True, "reason": "peer_agrees", "token_f1": score}
    return {"gate_keeps_answer": False, "reason": "peer_disagrees_abstain", "token_f1": score}
