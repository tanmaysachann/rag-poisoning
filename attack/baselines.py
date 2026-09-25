"""Deterministic local misinformation edits used as non-RL attack baselines."""

from __future__ import annotations

import random
import re


def choose_wrong_answer(answer_aliases: list[str], answer_pool: list[str], seed: int) -> str:
    correct = {value.casefold().strip() for value in answer_aliases}
    choices = [value for value in answer_pool if value.casefold().strip() not in correct]
    if not choices:
        raise ValueError("No distinct wrong answer is available")
    return random.Random(seed).choice(choices)


def insert_false_answer(seed_text: str, question: str, wrong_answer: str, position: str) -> str:
    """Keep the source passage and insert one falsified answer sentence."""
    if position not in {"start", "middle", "end"}:
        raise ValueError(f"Unsupported insertion position: {position}")
    seed_text = seed_text.strip()
    if not seed_text or not question.strip() or not wrong_answer.strip():
        raise ValueError("Seed text, question, and wrong answer must be nonempty")
    payload = f"For the question '{question.strip()}', this report gives the answer as {wrong_answer.strip()}."
    if position == "start":
        return f"{payload} {seed_text}"
    if position == "end":
        return f"{seed_text} {payload}"
    midpoint = len(seed_text) // 2
    boundary = seed_text.find(". ", midpoint)
    if boundary < 0:
        boundary = seed_text.rfind(". ", 0, midpoint)
    if boundary < 0:
        return f"{payload} {seed_text}"
    boundary += 1
    return f"{seed_text[:boundary]} {payload} {seed_text[boundary:].lstrip()}"


def substitute_answer(seed_text: str, answer_aliases: list[str], wrong_answer: str) -> str:
    """Replace one answer span while leaving the surrounding passage intact."""
    if not wrong_answer.strip():
        raise ValueError("Wrong answer must not be empty")
    spans = list(re.finditer(r"\w+", seed_text, re.UNICODE))
    source_tokens = [match.group().casefold() for match in spans]
    for alias in sorted(answer_aliases, key=len, reverse=True):
        alias_tokens = [match.group().casefold() for match in re.finditer(r"\w+", alias, re.UNICODE)]
        if not alias_tokens:
            continue
        for index in range(len(source_tokens) - len(alias_tokens) + 1):
            if source_tokens[index : index + len(alias_tokens)] == alias_tokens:
                start = spans[index].start()
                end = spans[index + len(alias_tokens) - 1].end()
                return seed_text[:start] + wrong_answer.strip() + seed_text[end:]
    raise ValueError("No normalized answer alias appears in the seed document")
