"""Deterministic, bounded edits. Document content is never executed."""

from __future__ import annotations

import re

from attack.baselines import substitute_answer
from poison.actions import EditAction, POSITIONS

SYNONYMS = {
    "important": "significant",
    "known": "recognized",
    "large": "substantial",
    "small": "compact",
    "used": "employed",
    "created": "developed",
    "found": "identified",
    "shows": "indicates",
}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _position_index(length: int, position: int) -> int:
    return (0, length // 2, length)[position]


def apply_edit(
    text: str, action: EditAction, *, question: str, wrong_answer: str,
    answer_aliases: list[str],
) -> str:
    """Apply one edit; raise ValueError when its effect cannot be defined."""
    if action.operation == "STOP":
        return text
    sentences = _sentences(text)
    if not sentences:
        raise ValueError("Cannot edit an empty document")
    if action.operation == "INSERT":
        payloads = (
            f"The answer is {wrong_answer}.",
            f"For the question '{question}', the answer is {wrong_answer}.",
            f"According to this report, the verified answer is {wrong_answer}.",
        )
        selected = payloads[action.payload]
        if selected.casefold() in text.casefold():
            raise ValueError("Identical payload has already been inserted")
        sentences.insert(_position_index(len(sentences), action.position), selected)
        return " ".join(sentences)
    if action.operation == "PARAPHRASE":
        return substitute_answer(text, answer_aliases, wrong_answer)
    index = min(_position_index(len(sentences), action.position), len(sentences) - 1)
    if action.operation == "DELETE":
        if len(sentences) == 1:
            raise ValueError("Cannot delete the only sentence")
        del sentences[index]
        return " ".join(sentences)
    if action.operation == "SYNONYM":
        sentence = sentences[index]
        for original, replacement in SYNONYMS.items():
            changed, count = re.subn(rf"\b{original}\b", replacement, sentence, count=1, flags=re.I)
            if count:
                sentences[index] = changed
                return " ".join(sentences)
        raise ValueError("No supported synonym appears in the selected sentence")
    raise ValueError(f"Unsupported operation: {action.operation}")
