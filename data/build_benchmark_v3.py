"""Build disjoint v3 single-passage benchmark excluding v1 and v2 sources.

v1 test is spent, v2 test is untouched - v3 uses new seed and exclusion sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from data.build_benchmark import (
    _answers,
    _contains_answer,
    _tokens,
    collect_nq_challenge,
    normalize_text,
)

SCHEMA_VERSION = 1
MS_MARCO_DATASET = "microsoft/ms_marco"
NQ_OPEN_DATASET = "google-research-datasets/nq_open"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_exclusions(roots: list[Path]) -> tuple[set[int], set[tuple[str, ...]], set[tuple[str, ...]]]:
    excl_ids: set[int] = set()
    excl_pass: set[tuple[str, ...]] = set()
    excl_q: set[tuple[str, ...]] = set()
    for root in roots:
        for split in ("train", "validation", "test"):
            c = root / split / "corpus.jsonl"
            q = root / split / "queries.jsonl"
            if not c.exists():
                continue
            for doc in _read_jsonl(c):
                if doc.get("source_query_id") is not None:
                    excl_ids.add(int(doc["source_query_id"]))
                excl_pass.add(_tokens(doc["text"]))
            if q.exists():
                for row in _read_jsonl(q):
                    excl_q.add(_tokens(row["question"]))
        # v2 layout: groups with passages
        for split in ("train", "validation", "test"):
            for name in (f"{split}/groups.jsonl", f"{split}.jsonl"):
                p = root / name
                if p.exists():
                    for row in _read_jsonl(p):
                        if row.get("source_query_id") is not None:
                            excl_ids.add(int(row["source_query_id"]))
                        for ps in row.get("passages", []):
                            excl_pass.add(_tokens(ps.get("text", "")))
                        excl_q.add(_tokens(row.get("question", "")))
    return excl_ids, excl_pass, excl_q


def collect_pairs_v3(
    rows: Iterable[dict],
    target: int,
    excl_ids: set[int],
    excl_pass: set[tuple[str, ...]],
    excl_q: set[tuple[str, ...]],
) -> tuple[list[dict], int]:
    pairs: list[dict] = []
    seen_q: set[tuple[str, ...]] = set(excl_q)
    seen_p: set[tuple[str, ...]] = set(excl_pass)
    scanned = 0
    for row in rows:
        scanned += 1
        sid = row.get("query_id")
        if sid is not None and int(sid) in excl_ids:
            continue
        query = normalize_text(row.get("query", ""))
        answers = _answers(row.get("answers"))
        qk = _tokens(query)
        if not qk or not answers or qk in seen_q:
            continue
        passages = row.get("passages") or {}
        if not isinstance(passages, dict):
            continue
        texts = passages.get("passage_text") or []
        selected = passages.get("is_selected") or []
        for pi, raw in enumerate(texts):
            if pi >= len(selected) or not selected[pi]:
                continue
            passage = normalize_text(raw)
            pk = _tokens(passage)
            if not 80 <= len(passage) <= 1200 or pk in seen_p:
                continue
            if not _contains_answer(passage, answers):
                continue
            seen_q.add(qk)
            seen_p.add(pk)
            pairs.append(
                {
                    "doc_id": len(pairs),
                    "text": passage,
                    "question": query,
                    "answer_aliases": answers,
                    "source_query_id": int(sid) if sid is not None else None,
                    "source_passage_index": pi,
                }
            )
            break
        if len(pairs) >= target:
            break
    if len(pairs) < target:
        raise RuntimeError(f"Found {len(pairs)} after {scanned} rows; need {target}")
    return pairs, scanned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--nq-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1042)
    ap.add_argument("--output", type=Path, default=Path("data/benchmark_v3"))
    ap.add_argument("--msmarco-revision", required=True)
    ap.add_argument("--nq-revision", required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    v1 = Path("data/benchmark")
    v2 = Path("data/benchmark_v2")
    excl_ids, excl_pass, excl_q = load_exclusions([v for v in (v1, v2) if v.exists()])
    print(f"Excluding {len(excl_ids)} source ids, {len(excl_pass)} passages, {len(excl_q)} questions")

    m_rows = load_dataset(MS_MARCO_DATASET, "v1.1", split="train", streaming=True, revision=args.msmarco_revision)
    n_rows = load_dataset(NQ_OPEN_DATASET, split="train", streaming=True, revision=args.nq_revision)
    pairs, scanned = collect_pairs_v3(m_rows, args.pairs, excl_ids, excl_pass, excl_q)

    shuffled = list(pairs)
    random.Random(args.seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    splits = {"train": shuffled[:n_train], "validation": shuffled[n_train:n_train+n_val], "test": shuffled[n_train+n_val:]}

    # NQ challenge against new test docs
    test_docs = [{"doc_id": d["doc_id"], "text": d["text"]} for d in splits["test"]]
    challenge = collect_nq_challenge(n_rows, args.nq_queries, test_docs)

    out: Path = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        sd = out / name
        sd.mkdir()
        with (sd / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as h:
            for it in items:
                h.write(json.dumps({"doc_id": it["doc_id"], "text": it["text"], "source_dataset": MS_MARCO_DATASET, "source_query_id": it["source_query_id"], "source_passage_index": it["source_passage_index"]}, ensure_ascii=False, sort_keys=True) + "\n")
        with (sd / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as h:
            for it in items:
                h.write(json.dumps({"qid": f"msmarco-v3-{it['doc_id']}", "question": it["question"], "answer_aliases": it["answer_aliases"], "support_doc_ids": [it["doc_id"]], "source_dataset": MS_MARCO_DATASET}, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "nq_challenge.jsonl").open("w", encoding="utf-8", newline="\n") as h:
        for c in challenge:
            h.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    files = {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*.jsonl"))}
    manifest = {"schema_version": SCHEMA_VERSION, "sources": {"ms_marco": {"dataset": MS_MARCO_DATASET, "config": "v1.1", "split": "train", "revision": args.msmarco_revision}, "nq_open": {"dataset": NQ_OPEN_DATASET, "split": "train", "revision": args.nq_revision}}, "seed": args.seed, "selection": "v3 disjoint single-passage, excludes v1+v2 source ids/passages/questions", "msmarco_rows_scanned": scanned, "counts": {**{k: len(v) for k, v in splits.items()}, "nq_challenge": len(challenge)}, "file_sha256": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Done v3: {out}")


if __name__ == "__main__":
    main()
