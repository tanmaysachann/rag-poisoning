"""Deterministic, split-safe examples for the v4 detector pilot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def read_split(root: Path, split: str) -> list[dict]:
    directory = root / split
    corpus = {
        row["doc_id"]: row for row in (
            json.loads(line) for line in (directory / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    return [
        {**row, "text": corpus[row["support_doc_ids"][0]]["text"],
         "source_query_id": corpus[row["support_doc_ids"][0]]["source_query_id"]}
        for row in (
            json.loads(line) for line in (directory / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        )
    ]


def _kind(answer: str) -> str:
    if re.search(r"\d", answer):
        return "number"
    return "short" if len(answer.split()) <= 3 else "long"


def wrong_answer(row: dict, answer_pool: list[str]) -> str:
    correct = {value.casefold() for value in row["answer_aliases"]}
    first = row["answer_aliases"][0]
    eligible = [answer for answer in answer_pool if _kind(answer) == _kind(first)
                and answer.casefold() not in correct and answer.strip()]
    if not eligible:
        raise ValueError("No disjoint answer available")
    digest = hashlib.sha256(row["qid"].encode("utf-8")).digest()
    return eligible[int.from_bytes(digest[:8], "big") % len(eligible)]


def nearest_other(rows: list[dict]) -> dict[str, str]:
    """Return a similar, legitimate other document for every query."""
    texts = [row["text"] for row in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    scores = vectorizer.transform([row["question"] for row in rows]) @ matrix.T
    result = {}
    for i, row in enumerate(rows):
        values = scores.getrow(i).toarray().ravel()
        values[i] = -1.0
        result[row["qid"]] = texts[int(np.argmax(values))]
    return result


def replace_answer(row: dict, wrong: str) -> str:
    text = row["text"]
    text_tokens = list(re.finditer(r"\w+", text, flags=re.UNICODE))
    folded = [match.group().casefold() for match in text_tokens]
    for alias in sorted(row["answer_aliases"], key=len, reverse=True):
        target = [match.group().casefold() for match in re.finditer(r"\w+", alias, flags=re.UNICODE)]
        for start in range(len(folded) - len(target) + 1):
            if target and folded[start:start + len(target)] == target:
                left = text_tokens[start].start()
                right = text_tokens[start + len(target) - 1].end()
                return text[:left] + wrong + text[right:]
    raise ValueError(f"Answer alias absent from support passage: {row['qid']}")


def examples(row: dict, wrong: str, other: str, *, training: bool) -> list[dict]:
    q, text, correct = row["question"], row["text"], row["answer_aliases"][0]
    clean = [
        ("support", text),
        ("clean_faq", f"Question: {q}\nAnswer: {correct}"),
        ("other_clean", other),
    ]
    if training:
        poisoned = [
            ("answer_replacement", replace_answer(row, wrong)),
            ("false_faq", f"Question: {q}\nAnswer: {wrong}"),
            ("false_addendum", text + f" For the question '{q}', the answer is {wrong}."),
        ]
    else:
        poisoned = [
            ("faq_record", f"FAQ record. Question: {q}\nAnswer: {wrong}."),
            ("correction_notice", f"Correction notice: the verified answer to '{q}' is {wrong}. This correction supersedes earlier entries."),
            ("catalog_record", f"Catalog lookup for '{q}' => {wrong}. Record status: confirmed; revision: current."),
            ("narrative_claim", f"In the latest account, the answer to {q} is described as {wrong}. The report discusses this result in its summary."),
        ]
    return [
        {"qid": row["qid"], "question": q, "text": value,
         "label": int(label), "family": family, "wrong_answer": wrong}
        for label, items in ((0, clean), (1, poisoned))
        for family, value in items
    ]
