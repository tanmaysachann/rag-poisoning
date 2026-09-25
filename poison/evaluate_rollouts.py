"""Reproducible random/greedy edit-MDP baseline on benchmark questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attack.baselines import choose_wrong_answer
from config import RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from poison.edit_cache import EditEffectCache
from poison.env import DocumentEditEnv
from poison.rollout import run_episode


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--strategy", choices=("random", "greedy_proxy", "greedy_repeat"), default="random")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    validate_benchmark(args.benchmark)
    queries = _read_jsonl(args.benchmark / args.split / "queries.jsonl")[:args.limit]
    answer_pool = [row["answer_aliases"][0] for row in _read_jsonl(args.benchmark / "train" / "queries.jsonl")]
    cache = EditEffectCache(774)
    records = []
    for index, query in enumerate(queries):
        wrong = choose_wrong_answer(query["answer_aliases"], answer_pool, args.seed + index)
        env = DocumentEditEnv(
            args.benchmark / args.split / "corpus.jsonl", query, wrong,
            max_steps=3, replace_existing=False,
        )
        result = run_episode(env, seed=args.seed + index, strategy=args.strategy, cache=cache)
        records.append({
            "qid": query["qid"], "split": args.split, "strategy": args.strategy,
            "wrong_answer": wrong, "attack_doc_id": env.attack_doc_id,
            "seed_doc_id": env.seed_doc_id, "seed": args.seed + index, **result,
        })
    output = args.output or RESULTS_DIR / f"edit_rollout_{args.split}_{args.strategy}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "split": args.split, "strategy": args.strategy, "cases": len(records),
        "retrieval_rate": sum(row["terminal"]["retrieved"] for row in records) / len(records),
        "attack_success_rate": sum(row["terminal"]["attack_success"] for row in records) / len(records),
        "mean_reward": sum(row["total_reward"] for row in records) / len(records),
        "cached_transitions": len(cache.rows), "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
