"""Compare matched PPO training with and without edit-effect cache shaping."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from config import ROOT_DIR
from evaluation.summarize_ppo_proxy_ablation import SEEDS, _hash
from poison.reward import RewardWeights


def summarize(root: Path = ROOT_DIR, *, bootstrap_samples: int = 10000) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    runs, paired = [], {}
    for seed in SEEDS:
        original_checkpoint = root / ("artifacts/poison_policy_ppo_defender.pt" if seed == 42
                                      else f"artifacts/poison_policy_ppo_defender_seed{seed}.pt")
        cache_checkpoint = root / f"artifacts/poison_policy_ppo_cache_seed{seed}.pt"
        original_result = root / f"results/ppo_validation_defender_seed{seed}.json"
        cache_result = root / f"results/ppo_validation_cache_seed{seed}.json"
        before = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
        after = torch.load(cache_checkpoint, map_location="cpu", weights_only=False)
        baseline, cache_config = before["config"], after["config"]
        for name in ("batches", "episodes_per_batch", "seed", "learning_rate", "max_steps",
                     "state_dim", "defender_aware", "replace_existing", "detector_sha256"):
            if baseline[name] != cache_config[name]:
                raise ValueError(f"Seed {seed}: cache run changed {name}")
        if (baseline["seed"] != seed or baseline.get("cache_weight", 0) != 0
                or cache_config["cache_weight"] != 0.5
                or baseline.get("proxy_value_weight", 0.1) != cache_config["proxy_value_weight"]
                or baseline.get("reward_weights", vars(RewardWeights())) != cache_config["reward_weights"]
                or before["train_corpus_sha256"] != after["train_corpus_sha256"]
                or before["train_queries_sha256"] != after["train_queries_sha256"]):
            raise ValueError(f"Seed {seed}: training input or cache contract differs")
        original = json.loads(original_result.read_text(encoding="utf-8"))
        cached = json.loads(cache_result.read_text(encoding="utf-8"))
        before_rows = {row["qid"]: row for row in original["cases"]}
        after_rows = {row["qid"]: row for row in cached["cases"]}
        if (len(before_rows) != 75 or len(after_rows) != 75
                or set(before_rows) != set(after_rows)
                or original["summary"]["split"] != "validation"
                or cached["summary"]["split"] != "validation"
                or any(before_rows[qid]["wrong_answer"] != after_rows[qid]["wrong_answer"]
                       for qid in before_rows)):
            raise ValueError(f"Seed {seed}: validation cases or wrong-answer schedules differ")
        differences = {
            qid: int(bool(after_rows[qid]["terminal"]["defended_attack_success"]))
            - int(bool(before_rows[qid]["terminal"]["defended_attack_success"]))
            for qid in before_rows
        }
        paired[seed] = differences
        before_success = sum(row["terminal"]["defended_attack_success"] for row in before_rows.values())
        after_success = sum(row["terminal"]["defended_attack_success"] for row in after_rows.values())
        if (round(original["summary"]["defended_attack_success_rate"] * 75) != before_success
                or round(cached["summary"]["defended_attack_success_rate"] * 75) != after_success):
            raise ValueError(f"Seed {seed}: summary counts differ")
        history = [json.loads(line) for line in (root / f"results/ppo_train_cache_seed{seed}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(history) != 10 or history[-1]["edit_cache_rows"] < 1:
            raise ValueError(f"Seed {seed}: cache was not populated")
        runs.append({
            "seed": seed, "cases": 75, "original_successes": before_success,
            "cache_successes": after_success,
            "changed_case_outcomes": sum(value != 0 for value in differences.values()),
            "final_cache_rows": history[-1]["edit_cache_rows"],
            "original_operation_counts": original["summary"]["operation_counts"],
            "cache_operation_counts": cached["summary"]["operation_counts"],
            "original_checkpoint_sha256": _hash(original_checkpoint),
            "cache_checkpoint_sha256": _hash(cache_checkpoint),
            "original_result_sha256": _hash(original_result),
            "cache_result_sha256": _hash(cache_result),
        })
    qids = sorted(paired[SEEDS[0]])
    if any(set(paired[seed]) != set(qids) for seed in SEEDS):
        raise ValueError("Seeds must share validation questions")
    by_query = [sum(paired[seed][qid] for seed in SEEDS) for qid in qids]
    rng = random.Random(20261001)
    boot = sorted(sum(by_query[rng.randrange(len(qids))] for _ in qids) / (len(qids) * len(SEEDS))
                  for _ in range(bootstrap_samples))
    total = len(qids) * len(SEEDS)
    return {
        "split": "validation", "seeds": list(SEEDS), "episodes_per_seed": 200,
        "ablation": "edit-effect cache reward weight 0 to 0.5; all other losses and rewards retained",
        "runs": runs, "total_paired_cases": total,
        "original_successes": sum(row["original_successes"] for row in runs),
        "cache_successes": sum(row["cache_successes"] for row in runs),
        "changed_case_outcomes": sum(row["changed_case_outcomes"] for row in runs),
        "paired_cache_minus_original_rate": sum(by_query) / total,
        "query_cluster_bootstrap_95pct_difference": [
            boot[int(.025 * (bootstrap_samples - 1))], boot[int(.975 * (bootstrap_samples - 1))],
        ],
        "bootstrap_seed": 20261001, "bootstrap_samples": bootstrap_samples,
        "note": "Development validation. Cache predicts the proxy step reward of nearby state-action edits; it does not predict terminal attack success. Exact NumPy neighbor search was used during training only.",
    }


if __name__ == "__main__":
    result = summarize()
    output = ROOT_DIR / "results/ppo_cache_ablation_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
