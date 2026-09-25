"""CPU PPO training on the frozen training split of the local benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import torch

from attack.baselines import choose_wrong_answer
from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from poison.env import DocumentEditEnv, STATE_VERSION
from poison.policy import FactoredActorCritic
from poison.ppo import collect_episode, ppo_update
from detect.research_detector import ResearchDetector


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--episodes-per-batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--defender-aware", action="store_true", help="Use the frozen detector in step and terminal rewards")
    parser.add_argument("--replace-existing", action="store_true", help="Edit the support passage before accepted ingest")
    parser.add_argument("--detector", type=Path, default=ARTIFACTS_DIR / "research_detector_hashing.joblib")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batches < 1 or args.episodes_per_batch < 1:
        raise ValueError("Batch and episode counts must be positive")
    validate_benchmark(args.benchmark)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    corpus_path = args.benchmark / "train" / "corpus.jsonl"
    queries_path = args.benchmark / "train" / "queries.jsonl"
    queries = _read_jsonl(queries_path)
    answer_pool = [row["answer_aliases"][0] for row in queries]
    policy = FactoredActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    detector = joblib.load(args.detector) if args.defender_aware else None
    if detector is not None and not isinstance(detector, ResearchDetector):
        raise ValueError("Wrong detector artifact type")
    records = []
    for batch in range(args.batches):
        transitions = []
        terminal_rows = []
        for offset in range(args.episodes_per_batch):
            index = batch * args.episodes_per_batch + offset
            query = queries[index % len(queries)]
            wrong = choose_wrong_answer(query["answer_aliases"], answer_pool, args.seed + index)
            env = DocumentEditEnv(
                corpus_path, query, wrong, max_steps=3,
                detector=detector, replace_existing=args.replace_existing,
            )
            episode, terminal = collect_episode(env, policy, seed=args.seed + index)
            transitions.extend(episode)
            terminal_rows.append(terminal)
        update = ppo_update(policy, optimizer, transitions)
        record = {
            "batch": batch + 1, "episodes": len(terminal_rows),
            "transitions": len(transitions),
            "retrieval_rate": sum(row["retrieved"] for row in terminal_rows) / len(terminal_rows),
            "attack_success_rate": sum(row["attack_success"] for row in terminal_rows) / len(terminal_rows),
            "defended_attack_success_rate": (
                sum(row["defended_attack_success"] for row in terminal_rows) / len(terminal_rows)
                if detector is not None else None
            ),
            "attack_detection_rate": (
                sum(row["attack_doc_detected"] is True for row in terminal_rows) / len(terminal_rows)
                if detector is not None else None
            ),
            **update,
        }
        records.append(record)
        print(json.dumps(record))
    output = args.output or ARTIFACTS_DIR / ("poison_policy_ppo_defender.pt" if args.defender_aware else "poison_policy_ppo.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_version": STATE_VERSION,
        "policy_class": "FactoredActorCritic",
        "state_dict": policy.state_dict(),
        "train_corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "train_queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "config": {
            "batches": args.batches, "episodes_per_batch": args.episodes_per_batch,
            "seed": args.seed, "learning_rate": args.learning_rate,
            "max_steps": 3, "state_dim": 774,
            "defender_aware": args.defender_aware,
            "replace_existing": args.replace_existing,
            "detector_sha256": hashlib.sha256(args.detector.read_bytes()).hexdigest() if detector else None,
            "detector_threshold": detector.threshold if detector else None,
        },
    }
    torch.save(checkpoint, output)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / ("ppo_train_defender.jsonl" if args.defender_aware else "ppo_train.jsonl")
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    print(f"Saved {output} and {metrics_path}")


if __name__ == "__main__":
    main()
