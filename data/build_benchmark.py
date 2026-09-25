"""Build a frozen, answer-supported MS MARCO benchmark and NQ-Open challenge set.

The existing five-case Review-1 demo is intentionally left untouched. This
builder writes a separate research benchmark with split-specific corpora so
policy and detector training cannot inspect held-out source documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MS_MARCO_DATASET = "microsoft/ms_marco"
NQ_OPEN_DATASET = "google-research-datasets/nq_open"
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def normalize_text(value: object) -> str:
    return SPACE_RE.sub(" ", str(value)).strip()


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(normalize_text(value).casefold()))


def _contains_answer(passage: str, answers: list[str]) -> bool:
    haystack = _tokens(passage)
    for answer in answers:
        needle = _tokens(answer)
        if needle and any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1)):
            return True
    return False


def _answers(value: object) -> list[str]:
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    results = [normalize_text(item) for item in values]
    return [item for item in results if item and item.casefold() != "no answer present."]


def collect_msmarco_pairs(rows: Iterable[dict[str, Any]], target_pairs: int) -> tuple[list[dict], int]:
    """Keep one selected passage with an explicit answer span per unique query."""
    if target_pairs < 3:
        raise ValueError("target_pairs must be at least 3 to populate all splits")
    pairs: list[dict] = []
    seen_queries: set[tuple[str, ...]] = set()
    seen_passages: set[tuple[str, ...]] = set()
    scanned = 0
    for row in rows:
        scanned += 1
        query = normalize_text(row.get("query", ""))
        answers = _answers(row.get("answers"))
        query_key = _tokens(query)
        if not query_key or not answers or query_key in seen_queries:
            continue
        passages = row.get("passages") or {}
        if not isinstance(passages, dict):
            continue
        texts = passages.get("passage_text") or []
        selected = passages.get("is_selected") or []
        for passage_index, raw in enumerate(texts):
            if passage_index >= len(selected) or not selected[passage_index]:
                continue
            passage = normalize_text(raw)
            passage_key = _tokens(passage)
            if not 80 <= len(passage) <= 1200 or passage_key in seen_passages:
                continue
            if not _contains_answer(passage, answers):
                continue
            seen_queries.add(query_key)
            seen_passages.add(passage_key)
            pairs.append(
                {
                    "doc_id": len(pairs),
                    "text": passage,
                    "question": query,
                    "answer_aliases": answers,
                    "source_query_id": int(row["query_id"]) if row.get("query_id") is not None else None,
                    "source_passage_index": passage_index,
                }
            )
            break
        if len(pairs) >= target_pairs:
            break
    if len(pairs) < target_pairs:
        raise RuntimeError(
            f"Found {len(pairs)} answer-supported pairs after scanning {scanned} MS MARCO rows; "
            f"requested {target_pairs}. Increase source rows or lower --pairs."
        )
    return pairs, scanned


def split_pairs(pairs: list[dict], seed: int) -> dict[str, list[dict]]:
    """Split unique query/source pairs without sharing documents across sets."""
    if len(pairs) < 3:
        raise ValueError("At least three pairs are required")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * 0.70))
    n_validation = max(1, int(n * 0.15))
    if n_train + n_validation >= n:
        n_train = n - 2
        n_validation = 1
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }


def collect_nq_challenge(rows: Iterable[dict[str, Any]], limit: int, test_docs: list[dict]) -> list[dict]:
    """Mark lexical alias presence; NQ answerability still needs verification."""
    if limit < 0:
        raise ValueError("NQ challenge limit must be nonnegative")
    if limit == 0:
        return []
    challenge: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        question = normalize_text(row.get("question", ""))
        answers = _answers(row.get("answer"))
        key = _tokens(question)
        if not key or not answers or key in seen:
            continue
        seen.add(key)
        support_ids = [doc["doc_id"] for doc in test_docs if _contains_answer(doc["text"], answers)]
        challenge.append(
            {
                "qid": f"nq-{len(challenge)}",
                "question": question,
                "answer_aliases": answers,
                "support_doc_ids": support_ids,
                "answer_alias_present_in_test_corpus": bool(support_ids),
                "source_dataset": NQ_OPEN_DATASET,
            }
        )
        if len(challenge) >= limit:
            break
    return challenge


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_benchmark(
    msmarco_rows: Iterable[dict[str, Any]],
    nq_rows: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    target_pairs: int,
    nq_limit: int,
    seed: int,
    msmarco_revision: str,
    nq_revision: str,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Benchmark output already exists: {output_dir}")
    pairs, scanned = collect_msmarco_pairs(msmarco_rows, target_pairs)
    splits = split_pairs(pairs, seed)
    challenge = collect_nq_challenge(nq_rows, nq_limit, splits["test"])
    if len(challenge) < nq_limit:
        raise RuntimeError(f"Only found {len(challenge)} NQ-Open questions; requested {nq_limit}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        split_dir = output_dir / name
        split_dir.mkdir()
        _write_jsonl(
            split_dir / "corpus.jsonl",
            (
                {
                    "doc_id": item["doc_id"],
                    "text": item["text"],
                    "source_dataset": MS_MARCO_DATASET,
                    "source_query_id": item["source_query_id"],
                    "source_passage_index": item["source_passage_index"],
                }
                for item in items
            ),
        )
        _write_jsonl(
            split_dir / "queries.jsonl",
            (
                {
                    "qid": f"msmarco-{item['doc_id']}",
                    "question": item["question"],
                    "answer_aliases": item["answer_aliases"],
                    "support_doc_ids": [item["doc_id"]],
                    "source_dataset": MS_MARCO_DATASET,
                }
                for item in items
            ),
        )
    _write_jsonl(output_dir / "nq_challenge.jsonl", challenge)

    files = {
        str(path.relative_to(output_dir)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.rglob("*.jsonl"))
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "ms_marco": {"dataset": MS_MARCO_DATASET, "config": "v1.1", "split": "train", "revision": msmarco_revision},
            "nq_open": {"dataset": NQ_OPEN_DATASET, "split": "train", "revision": nq_revision},
        },
        "seed": seed,
        "selection": "selected MS MARCO passage with exact normalized answer-token span; one passage per unique query",
        "msmarco_rows_scanned": scanned,
        "counts": {**{name: len(items) for name, items in splits.items()},
                   "nq_challenge": len(challenge),
                   "nq_alias_present": sum(item["answer_alias_present_in_test_corpus"] for item in challenge)},
        "file_sha256": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _resolve_revision(dataset: str, revision: str | None) -> str:
    if revision:
        return revision
    from huggingface_hub import HfApi

    resolved = HfApi().dataset_info(dataset).sha
    if not resolved:
        raise RuntimeError(f"Could not resolve a revision for {dataset}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--nq-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "benchmark")
    parser.add_argument("--msmarco-revision")
    parser.add_argument("--nq-revision")
    args = parser.parse_args()

    from datasets import load_dataset

    msmarco_revision = _resolve_revision(MS_MARCO_DATASET, args.msmarco_revision)
    nq_revision = _resolve_revision(NQ_OPEN_DATASET, args.nq_revision)
    msmarco_rows = load_dataset(MS_MARCO_DATASET, "v1.1", split="train", streaming=True, revision=msmarco_revision)
    nq_rows = load_dataset(NQ_OPEN_DATASET, split="train", streaming=True, revision=nq_revision)
    manifest = build_benchmark(
        msmarco_rows,
        nq_rows,
        args.output,
        target_pairs=args.pairs,
        nq_limit=args.nq_queries,
        seed=args.seed,
        msmarco_revision=msmarco_revision,
        nq_revision=nq_revision,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Frozen benchmark: {args.output}")


if __name__ == "__main__":
    main()
