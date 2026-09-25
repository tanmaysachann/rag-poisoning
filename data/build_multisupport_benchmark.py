"""Build a disjoint two-support-passage benchmark from pinned MS MARCO v1.1.

Two labeled passages per question make corroboration experiments possible.
Passage agreement does not prove independent origin or factual truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from data.build_benchmark import _answers, _contains_answer, _tokens, normalize_text


def collect_groups(rows: Iterable[dict], *, target: int, excluded_source_ids: set[int],
                   excluded_passages: set[tuple[str, ...]] | None = None,
                   excluded_questions: set[tuple[str, ...]] | None = None) -> tuple[list[dict], int]:
    if target < 3:
        raise ValueError("target must be at least three groups")
    groups = []
    seen_queries = set(excluded_questions or ())
    seen_passages = set(excluded_passages or ())
    scanned = 0
    for row in rows:
        scanned += 1
        source_id = row.get("query_id")
        if source_id is None or int(source_id) in excluded_source_ids:
            continue
        question = normalize_text(row.get("query", ""))
        question_key = _tokens(question)
        aliases = _answers(row.get("answers"))
        if not question_key or not aliases or question_key in seen_queries:
            continue
        passages = row.get("passages") or {}
        if not isinstance(passages, dict):
            continue
        texts = passages.get("passage_text") or []
        selected = passages.get("is_selected") or []
        candidates = []
        for index, raw in enumerate(texts):
            if index >= len(selected) or not selected[index]:
                continue
            passage = normalize_text(raw)
            key = _tokens(passage)
            if not 80 <= len(passage) <= 1200 or not key or key in seen_passages:
                continue
            if _contains_answer(passage, aliases):
                candidates.append({"text": passage, "key": key, "source_passage_index": index})
        unique = []
        used = set()
        for candidate in candidates:
            if candidate["key"] not in used:
                unique.append(candidate)
                used.add(candidate["key"])
        if len(unique) < 2:
            continue
        chosen = unique[:2]
        seen_queries.add(question_key)
        seen_passages.update(item["key"] for item in chosen)
        groups.append({"source_query_id": int(source_id), "question": question,
                       "answer_aliases": aliases, "passages": chosen})
        if len(groups) >= target:
            break
    if len(groups) != target:
        raise RuntimeError(f"Found {len(groups)} two-passage groups after {scanned} rows; requested {target}")
    return groups, scanned


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(groups: list[dict], output: Path, *, revision: str, seed: int, scanned: int,
          excluded_ids: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Benchmark output already exists: {output}")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_count, validation_count = int(n * 0.70), int(n * 0.15)
    splits = {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count:train_count + validation_count],
        "test": shuffled[train_count + validation_count:],
    }
    output.mkdir(parents=True, exist_ok=True)
    next_doc_id = 0
    for split, items in splits.items():
        folder = output / split
        folder.mkdir()
        corpus, queries = [], []
        for group in items:
            support_ids = []
            for passage in group["passages"]:
                support_ids.append(next_doc_id)
                corpus.append({"doc_id": next_doc_id, "text": passage["text"],
                               "source_dataset": "microsoft/ms_marco",
                               "source_query_id": group["source_query_id"],
                               "source_passage_index": passage["source_passage_index"]})
                next_doc_id += 1
            queries.append({"qid": f"msmarco2-{group['source_query_id']}",
                            "question": group["question"],
                            "answer_aliases": group["answer_aliases"],
                            "support_doc_ids": support_ids,
                            "source_dataset": "microsoft/ms_marco",
                            "source_query_id": group["source_query_id"]})
        write_jsonl(folder / "corpus.jsonl", corpus)
        write_jsonl(folder / "queries.jsonl", queries)
    hashes = {str(path.relative_to(output)).replace("\\", "/"):
              hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.rglob("*.jsonl"))}
    manifest = {
        "schema_version": 2, "dataset": "microsoft/ms_marco", "config": "v1.1",
        "revision": revision, "source_split": "train", "seed": seed,
        "selection": "two distinct selected passages per question, each containing an exact normalized answer alias",
        "source_rows_scanned": scanned, "v1_source_ids_excluded": excluded_ids,
        "passages_per_question": 2,
        "counts": {split: {"queries": len(items), "passages": 2 * len(items)} for split, items in splits.items()},
        "file_sha256": hashes,
        "test_state": "frozen_unopened_for_model_development",
        "provenance_limit": "Two MS MARCO passages may share origin; lexical alias presence is not independent factual verification.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=200)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "benchmark_v2")
    args = parser.parse_args()
    from datasets import load_dataset

    base = Path(__file__).resolve().parent / "benchmark"
    original_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    revision = original_manifest["sources"]["ms_marco"]["revision"]
    original_docs = [
        json.loads(line)
        for split in ("train", "validation", "test")
        for line in (base / split / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    excluded = {int(row["source_query_id"]) for row in original_docs}
    excluded_passages = {_tokens(row["text"]) for row in original_docs}
    excluded_questions = {
        _tokens(json.loads(line)["question"])
        for split in ("train", "validation", "test")
        for line in (base / split / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True, revision=revision)
    groups, scanned = collect_groups(
        rows, target=args.groups, excluded_source_ids=excluded,
        excluded_passages=excluded_passages, excluded_questions=excluded_questions,
    )
    manifest = build(groups, args.output, revision=revision, seed=args.seed,
                     scanned=scanned, excluded_ids=len(excluded))
    print(json.dumps({"counts": manifest["counts"], "source_rows_scanned": scanned}, indent=2))


if __name__ == "__main__":
    main()
