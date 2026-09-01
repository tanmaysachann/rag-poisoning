"""Stream a compact evaluation/demo query set from NQ-Open."""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

from config import QUERIES_PATH, ensure_project_dirs


def _stream_nq_open() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    try:
        return load_dataset(
            "google-research-datasets/nq_open", split="train", streaming=True
        )
    except Exception:
        return load_dataset("nq_open", split="train", streaming=True)


def build_queries_from_nq_open(n_queries: int = 50) -> None:
    """Write non-empty NQ-Open questions and answer aliases to JSONL."""

    if n_queries <= 0:
        raise ValueError("n_queries must be positive")

    ensure_project_dirs()
    rows: list[dict[str, object]] = []
    try:
        stream = _stream_nq_open()
    except Exception as error:
        print(f"Dataset unavailable ({error}); using offline demo queries.")
        stream = [
            {"question":"Where is the Eiffel Tower located?","answer":["Paris, France"]},
            {"question":"At what temperature does water boil at sea level?","answer":["100 degrees Celsius"]},
            {"question":"What is the capital of Australia?","answer":["Canberra"]},
            {"question":"What is photosynthesis?","answer":["the process plants use to convert light into energy"]},
            {"question":"What is the largest ocean on Earth?","answer":["Pacific Ocean"]},
        ]
    for item in stream:
        question = " ".join(str(item.get("question", "")).split())
        raw_answers = item.get("answer") or []
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        answers = [" ".join(str(answer).split()) for answer in raw_answers if str(answer).strip()]
        if not question or not answers:
            continue
        rows.append({"qid": len(rows), "question": question, "answer_aliases": answers})
        if len(rows) >= n_queries:
            break

    if len(rows) < n_queries:
        raise RuntimeError(f"Only found {len(rows)} valid queries; requested {n_queries}.")

    with QUERIES_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} queries to {QUERIES_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-queries", type=int, default=50)
    args = parser.parse_args()
    build_queries_from_nq_open(args.n_queries)


if __name__ == "__main__":
    main()
