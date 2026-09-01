"""Stream a small, reproducible passage corpus from MS MARCO v1.1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from config import BASE_CORPUS_PATH, ensure_project_dirs


def _stream_msmarco() -> Iterable[dict[str, Any]]:
    """Load the requested dataset while tolerating its old Hub alias."""

    from datasets import load_dataset

    try:
        return load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
    except Exception:
        return load_dataset("ms_marco", "v1.1", split="train", streaming=True)


def build_corpus_from_msmarco(n_passages: int = 500, seed: int = 42) -> None:
    """Build ``data/corpus.jsonl`` from streamed MS MARCO passage text.

    A pool three times the requested size is collected before seeded shuffling,
    which keeps the build quick while avoiding a simple first-N sample.
    """

    if n_passages <= 0:
        raise ValueError("n_passages must be positive")

    ensure_project_dirs()
    candidates: list[str] = []
    seen: set[str] = set()
    pool_size = 3 * n_passages

    try:
      stream = _stream_msmarco()
    except Exception as error:
      print(f"Dataset unavailable ({error}); using offline demo passages.")
      stream = [{"passages": {"passage_text": [
        "The Eiffel Tower is an iron lattice tower in Paris, France, built for the 1889 World's Fair and visited by millions of people.",
        "At sea level, water boils at approximately 100 degrees Celsius because its vapor pressure matches atmospheric pressure.",
        "Canberra is the capital city of Australia and serves as the country's political center.",
        "Photosynthesis is the process by which green plants use light, water, and carbon dioxide to produce sugars and oxygen.",
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering more area than any other ocean."
      ]}}]
    for row in stream:
        passages = row.get("passages") or {}
        texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
        for raw_text in texts:
            text = " ".join(str(raw_text).split())
            if not 80 <= len(text) <= 1200:
                continue
            key = text[:200]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(text)
            if len(candidates) >= pool_size:
                break
        if len(candidates) >= pool_size:
            break

    if len(candidates) < n_passages:
        raise RuntimeError(
            f"Only found {len(candidates)} eligible passages; requested {n_passages}."
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n_passages]
    with BASE_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for doc_id, text in enumerate(selected):
            handle.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} passages to {BASE_CORPUS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-passages", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_corpus_from_msmarco(args.n_passages, args.seed)


if __name__ == "__main__":
    main()
