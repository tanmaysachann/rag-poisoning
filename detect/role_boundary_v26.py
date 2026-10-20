"""Features for a trained role-boundary and instruction-redirection detector."""

from __future__ import annotations

import re


ROLE = {
    "bracket_system": re.compile(r"\[\s*system\s*\]", re.I),
    "chat_system": re.compile(r"<\|im_start\|>\s*system\b", re.I),
    "xml_system": re.compile(r"<message\s+role\s*=\s*['\"]system['\"]", re.I),
    "markdown_system": re.compile(r"(?m)^\s*#{2,4}\s*system\b", re.I),
    "bracket_assistant": re.compile(r"\[\s*assistant\s*\]", re.I),
    "chat_assistant": re.compile(r"<\|im_start\|>\s*assistant\b", re.I),
    "xml_assistant": re.compile(r"<message\s+role\s*=\s*['\"]assistant['\"]", re.I),
    "markdown_assistant": re.compile(r"(?m)^\s*#{2,4}\s*assistant\b", re.I),
}
DIRECTIVE = re.compile(r"\b(ignore|disregard|override|instead|replace|reply|respond|output|answer|follow|must|should|set)\b", re.I)
TARGET = re.compile(r"\b(final|response|question|user|source|passage|article|retrieved|answer)\b", re.I)
QUOTE = re.compile(r"\b(literal|example|syntax|serialization|notation|format|quoted|tutorial|documentation)\b", re.I)


def features(reference: str, candidate: str) -> dict[str, float]:
    # The tail is the attacker-controlled addition if the candidate preserves
    # the reference. For unrelated clean passages, scan the whole passage.
    tail = candidate[len(reference):] if candidate.startswith(reference) else candidate
    tail = tail[-1200:]
    found = {name: int(bool(pattern.search(tail))) for name, pattern in ROLE.items()}
    system = int(any(found[name] for name in found if name.endswith("system")))
    assistant = int(any(found[name] for name in found if name.endswith("assistant")))
    directives = len(DIRECTIVE.findall(tail))
    targets = len(TARGET.findall(tail))
    quotes = len(QUOTE.findall(tail))
    return {
        **found, "system_marker": system, "assistant_marker": assistant,
        "marker_and_directive": int(system and directives > 0),
        "marker_and_target": int(system and targets > 0),
        "role_transition": int(system and assistant),
        "directive_count": min(directives, 8), "target_count": min(targets, 8),
        "quote_count": min(quotes, 8),
        "marker_in_backticks": int(bool(re.search(r"`[^`]*(?:\[system\]|im_start|role=|### System)[^`]*`", tail, re.I))),
        "tail_fraction": min(len(tail) / max(len(candidate), 1), 1.0),
    }
