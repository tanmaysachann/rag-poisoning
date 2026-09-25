"""Evaluate random and greedy single-document misinformation edits locally."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack.baselines import choose_wrong_answer, insert_false_answer, substitute_answer
from attack.harness import stage_document_attack
from config import RESULTS_DIR
from data.validate_benchmark import validate_benchmark
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import HybridRetriever


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_case(
    clean_corpus: Path,
    query: dict,
    seed_doc: dict,
    wrong_answer: str,
    attack_id: int,
    positions: list[str],
    temporary: Path,
    answer_aliases: list[str] | None = None,
) -> dict:
    attempts: list[dict] = []
    for position in positions:
        attack_text = (
            substitute_answer(seed_doc["text"], answer_aliases or [], wrong_answer)
            if position == "replace_answer"
            else insert_false_answer(seed_doc["text"], query["question"], wrong_answer, position)
        )
        with tempfile.TemporaryDirectory(prefix="candidate-", dir=temporary) as candidate_dir:
            candidate_path = Path(candidate_dir)
            staged_path = candidate_path / "staged.jsonl"
            stage_document_attack(
                clean_corpus, staged_path, doc_id=attack_id, text=attack_text,
                replace_existing=position == "replace_answer",
            )
            retriever = HybridRetriever(
                staged_path, force_rebuild=True, artifact_dir=candidate_path / "index"
            )
            retrieved = retriever.retrieve(query["question"], top_k=5)
            rank = next((index for index, doc in enumerate(retrieved, 1) if doc["doc_id"] == attack_id), None)
            answer, source_doc_id, _ = _select_answer(query["question"], retrieved, retriever)
        success = source_doc_id == attack_id and wrong_answer.casefold() in answer.casefold()
        attempts.append(
            {
                "position": position,
                "text": attack_text,
                "top5_rank": rank,
                "undefended_answer": answer,
                "source_doc_id": source_doc_id,
                "attack_success": success,
            }
        )
    best = min(
        attempts,
        key=lambda attempt: (
            not attempt["attack_success"],
            attempt["top5_rank"] is None,
            attempt["top5_rank"] or 999,
        ),
    )
    return {
        "qid": query["qid"],
        "question": query["question"],
        "seed_doc_id": seed_doc["doc_id"],
        "attack_doc_id": attack_id,
        "wrong_answer": wrong_answer,
        "attempted_positions": positions,
        **best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--strategy", choices=("random", "greedy", "stealth"), default="random")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    validate_benchmark(args.benchmark)
    corpus_path = args.benchmark / args.split / "corpus.jsonl"
    docs = _read_jsonl(corpus_path)
    by_id = {doc["doc_id"]: doc for doc in docs}
    queries = _read_jsonl(args.benchmark / args.split / "queries.jsonl")[: args.limit]
    answer_pool = [query["answer_aliases"][0] for query in _read_jsonl(args.benchmark / "train" / "queries.jsonl")]
    rng = random.Random(args.seed)
    output = args.output or RESULTS_DIR / f"attack_baseline_{args.split}_{args.strategy}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    temp_parent = ROOT / "tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sentinel-attacks-", dir=temp_parent) as temporary:
        temp_path = Path(temporary)
        for index, query in enumerate(queries):
            seed_doc = by_id[query["support_doc_ids"][0]]
            wrong = choose_wrong_answer(query["answer_aliases"], answer_pool, args.seed + index)
            positions = (
                [rng.choice(("start", "middle", "end"))] if args.strategy == "random"
                else ["replace_answer"] if args.strategy == "stealth"
                else ["start", "middle", "end"]
            )
            record = evaluate_case(
                corpus_path, query, seed_doc, wrong,
                seed_doc["doc_id"] if args.strategy == "stealth" else max(by_id) + 1,
                positions, temp_path, answer_aliases=query["answer_aliases"],
            )
            record.update({
                "split": args.split, "strategy": args.strategy, "seed": args.seed,
                "answer_backend": "extractive", "attack_surface": "pre_ingest_content_poisoning",
            })
            records.append(record)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    retrieved = sum(record["top5_rank"] is not None for record in records)
    successful = sum(record["attack_success"] for record in records)
    print(
        f"{args.split}/{args.strategy}: {len(records)} attempts, "
        f"top-5 retrieval {retrieved / len(records):.3f}, "
        f"undefended attack success {successful / len(records):.3f}"
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
