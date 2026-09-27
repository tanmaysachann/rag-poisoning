"""Compare three matched PPO runs with and without detector-risk step shaping."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

from config import ROOT_DIR


SEEDS = (42, 43, 44)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: Path = ROOT_DIR, *, bootstrap_samples: int = 10000) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    runs, paired = [], {}
    for seed in SEEDS:
        original_checkpoint = root / ("artifacts/poison_policy_ppo_defender.pt" if seed == 42
                                      else f"artifacts/poison_policy_ppo_defender_seed{seed}.pt")
        ablated_checkpoint = root / f"artifacts/poison_policy_ppo_no_detection_reward_seed{seed}.pt"
        original_result = root / f"results/ppo_validation_defender_seed{seed}.json"
        ablated_result = root / f"results/ppo_validation_no_detection_reward_seed{seed}.json"
        before = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
        after = torch.load(ablated_checkpoint, map_location="cpu", weights_only=False)
        before_config, after_config = before["config"], after["config"]
        for name in ("batches", "episodes_per_batch", "seed", "learning_rate", "max_steps",
                     "state_dim", "defender_aware", "replace_existing", "detector_sha256"):
            if before_config[name] != after_config[name]:
                raise ValueError(f"Seed {seed}: ablation changed {name}")
        if (before_config["seed"] != seed or after_config["reward_weights"]["detection"] != 0
                or before_config.get("reward_weights", {}).get("detection", 0.5) != 0.5
                or before["train_corpus_sha256"] != after["train_corpus_sha256"]
                or before["train_queries_sha256"] != after["train_queries_sha256"]):
            raise ValueError(f"Seed {seed}: training input or reward contract differs")
        original = json.loads(original_result.read_text(encoding="utf-8"))
        ablated = json.loads(ablated_result.read_text(encoding="utf-8"))
        baseline_rows = {row["qid"]: row for row in original["cases"]}
        ablated_rows = {row["qid"]: row for row in ablated["cases"]}
        if (len(baseline_rows) != 75 or len(ablated_rows) != 75
                or set(baseline_rows) != set(ablated_rows)
                or original["summary"]["split"] != "validation"
                or ablated["summary"]["split"] != "validation"):
            raise ValueError(f"Seed {seed}: validation cases differ")
        if any(baseline_rows[qid]["wrong_answer"] != ablated_rows[qid]["wrong_answer"]
               for qid in baseline_rows):
            raise ValueError(f"Seed {seed}: wrong-answer schedules differ")
        difference = {
            qid: int(bool(baseline_rows[qid]["terminal"]["defended_attack_success"]))
            - int(bool(ablated_rows[qid]["terminal"]["defended_attack_success"]))
            for qid in baseline_rows
        }
        paired[seed] = difference
        original_success = sum(row["terminal"]["defended_attack_success"] for row in baseline_rows.values())
        ablated_success = sum(row["terminal"]["defended_attack_success"] for row in ablated_rows.values())
        if (round(original["summary"]["defended_attack_success_rate"] * 75) != original_success
                or round(ablated["summary"]["defended_attack_success_rate"] * 75) != ablated_success):
            raise ValueError(f"Seed {seed}: summary counts differ")
        runs.append({
            "seed": seed, "cases": 75, "original_successes": original_success,
            "no_detection_reward_successes": ablated_success,
            "original_operation_counts": original["summary"]["operation_counts"],
            "no_detection_reward_operation_counts": ablated["summary"]["operation_counts"],
            "original_checkpoint_sha256": _hash(original_checkpoint),
            "ablated_checkpoint_sha256": _hash(ablated_checkpoint),
            "original_result_sha256": _hash(original_result),
            "ablated_result_sha256": _hash(ablated_result),
        })
    qids = sorted(paired[SEEDS[0]])
    if any(set(paired[seed]) != set(qids) for seed in SEEDS):
        raise ValueError("Seeds must share validation questions")
    by_query = [sum(paired[seed][qid] for seed in SEEDS) for qid in qids]
    rng = random.Random(20260929)
    boot = sorted(sum(by_query[rng.randrange(len(qids))] for _ in qids) / (len(qids) * len(SEEDS))
                  for _ in range(bootstrap_samples))
    total = len(qids) * len(SEEDS)
    return {
        "split": "validation", "seeds": list(SEEDS), "episodes_per_seed": 200,
        "ablation": "detector-risk step reward weight 0.5 to 0; terminal defended-success reward retained",
        "runs": runs, "total_paired_cases": total,
        "original_successes": sum(row["original_successes"] for row in runs),
        "no_detection_reward_successes": sum(row["no_detection_reward_successes"] for row in runs),
        "paired_original_minus_ablation_rate": sum(by_query) / total,
        "query_cluster_bootstrap_95pct_difference": [
            boot[int(.025 * (bootstrap_samples - 1))], boot[int(.975 * (bootstrap_samples - 1))],
        ],
        "bootstrap_seed": 20260929, "bootstrap_samples": bootstrap_samples,
        "note": "Development validation on the same 75 questions per seed. This tests one reward component, not general PPO superiority.",
    }


if __name__ == "__main__":
    result = summarize()
    output = ROOT_DIR / "results/ppo_detection_reward_ablation_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
