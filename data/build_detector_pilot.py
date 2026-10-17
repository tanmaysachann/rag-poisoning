"""Freeze fresh MS MARCO QA pairs and an external SQuAD domain test.

The MS MARCO rows exclude the source IDs, questions, and passages used in the
earlier three benchmarks. SQuAD validation is reserved for evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from data.build_benchmark import _contains_answer, normalize_text
from data.build_benchmark_v3 import collect_pairs_v3, load_exclusions

MS_REVISION = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"


def _write_split(root: Path, name: str, rows: list[dict]) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    with (directory / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as docs, (
        directory / "queries.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as queries:
        for row in rows:
            docs.write(json.dumps({
                "doc_id": row["doc_id"], "text": row["text"],
                "source_dataset": row["source_dataset"],
                "source_query_id": row["source_query_id"],
            }, ensure_ascii=False, sort_keys=True) + "\n")
            queries.write(json.dumps({
                "qid": row["qid"], "question": row["question"],
                "answer_aliases": row["answer_aliases"],
                "support_doc_ids": [row["doc_id"]],
                "source_dataset": row["source_dataset"],
            }, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/detector_pilot_v4"))
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--validation", type=int, default=150)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--domain-test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1417)
    parser.add_argument("--exclude", nargs="*", type=Path, default=[],
                        help="Previously exposed experiment roots to exclude")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen data: {args.output}")

    from datasets import load_dataset

    exclusions = load_exclusions([
        Path("data/benchmark"), Path("data/benchmark_v2"), Path("data/benchmark_v3")
    ] + args.exclude)
    requested = args.train + args.validation + args.test
    ms_rows = load_dataset(
        "microsoft/ms_marco", "v1.1", split="train", streaming=True,
        revision=MS_REVISION,
    )
    pairs, scanned = collect_pairs_v3(ms_rows, requested, *exclusions)
    random.Random(args.seed).shuffle(pairs)
    for row in pairs:
        row["qid"] = f"msmarco-v4-{row['doc_id']}"
        row["source_dataset"] = "microsoft/ms_marco"
    splits = {
        "train": pairs[:args.train],
        "validation": pairs[args.train:args.train + args.validation],
        "test": pairs[args.train + args.validation:],
    }

    squad_rows = load_dataset(
        "rajpurkar/squad", split="validation", streaming=True,
        revision=SQUAD_REVISION,
    )
    domain: list[dict] = []
    seen_contexts: set[str] = set()
    for root in args.exclude:
        previous = root / "domain_test" / "corpus.jsonl"
        if previous.exists():
            for line in previous.read_text(encoding="utf-8").splitlines():
                context = normalize_text(json.loads(line)["text"])
                seen_contexts.add(hashlib.sha256(context.encode("utf-8")).hexdigest())
    for row in squad_rows:
        context = normalize_text(row["context"])
        question = normalize_text(row["question"])
        aliases = [normalize_text(answer) for answer in row["answers"]["text"]]
        key = hashlib.sha256(context.encode("utf-8")).hexdigest()
        if (not 80 <= len(context) <= 1200 or key in seen_contexts
                or not _contains_answer(context, aliases)):
            continue
        seen_contexts.add(key)
        domain.append({
            "doc_id": 1000000 + len(domain), "qid": f"squad-{row['id']}",
            "text": context, "question": question,
            "answer_aliases": aliases, "source_query_id": row["id"],
            "source_dataset": "rajpurkar/squad",
        })
        if len(domain) == args.domain_test:
            break
    if len(domain) != args.domain_test:
        raise RuntimeError(f"Only {len(domain)} SQuAD cases found")
    splits["domain_test"] = domain

    for name, rows in splits.items():
        _write_split(args.output, name, rows)
    files = {
        str(path.relative_to(args.output)).replace("\\", "/"):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.rglob("*.jsonl"))
    }
    manifest = {
        "schema_version": 1, "seed": args.seed,
        "ms_marco_revision": MS_REVISION, "squad_revision": SQUAD_REVISION,
        "ms_marco_rows_scanned": scanned,
        "selection": "fresh disjoint MS MARCO; SQuAD validation domain test",
        "additional_exclusions": [str(path) for path in args.exclude],
        "counts": {name: len(rows) for name, rows in splits.items()},
        "file_sha256": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
