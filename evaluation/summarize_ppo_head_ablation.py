"""Verify matched PPO runs with and without action-head conditioning."""

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
        before_checkpoint = root / ("artifacts/poison_policy_ppo_defender.pt" if seed == 42
                                    else f"artifacts/poison_policy_ppo_defender_seed{seed}.pt")
        after_checkpoint = root / f"artifacts/poison_policy_ppo_no_head_conditioning_seed{seed}.pt"
        before_result = root / f"results/ppo_validation_defender_seed{seed}.json"
        after_result = root / f"results/ppo_validation_no_head_conditioning_seed{seed}.json"
        before = torch.load(before_checkpoint, map_location="cpu", weights_only=False)
        after = torch.load(after_checkpoint, map_location="cpu", weights_only=False)
        initial, ablated = before["config"], after["config"]
        for key in ("batches", "episodes_per_batch", "seed", "learning_rate", "max_steps",
                    "state_dim", "defender_aware", "replace_existing", "detector_sha256"):
            if initial[key] != ablated[key]:
                raise ValueError(f"Seed {seed}: head ablation changed {key}")
        if (initial["seed"] != seed or initial.get("conditional_heads", True) is not True
                or ablated["conditional_heads"] is not False
                or initial.get("cache_weight", 0) != ablated["cache_weight"]
                or initial.get("proxy_value_weight", 0.1) != ablated["proxy_value_weight"]
                or initial.get("reward_weights", vars(RewardWeights())) != ablated["reward_weights"]
                or before["train_corpus_sha256"] != after["train_corpus_sha256"]
                or before["train_queries_sha256"] != after["train_queries_sha256"]):
            raise ValueError(f"Seed {seed}: training input or head contract differs")
        original = json.loads(before_result.read_text(encoding="utf-8"))
        changed = json.loads(after_result.read_text(encoding="utf-8"))
        first = {row["qid"]: row for row in original["cases"]}
        second = {row["qid"]: row for row in changed["cases"]}
        if (len(first) != 75 or len(second) != 75 or set(first) != set(second)
                or original["summary"]["split"] != "validation"
                or changed["summary"]["split"] != "validation"
                or any(first[qid]["wrong_answer"] != second[qid]["wrong_answer"] for qid in first)):
            raise ValueError(f"Seed {seed}: validation cases differ")
        differences = {
            qid: int(bool(first[qid]["terminal"]["defended_attack_success"]))
            - int(bool(second[qid]["terminal"]["defended_attack_success"]))
            for qid in first
        }
        paired[seed] = differences
        original_success = sum(row["terminal"]["defended_attack_success"] for row in first.values())
        ablated_success = sum(row["terminal"]["defended_attack_success"] for row in second.values())
        if (round(original["summary"]["defended_attack_success_rate"] * 75) != original_success
                or round(changed["summary"]["defended_attack_success_rate"] * 75) != ablated_success):
            raise ValueError(f"Seed {seed}: summary counts differ")
        runs.append({
            "seed": seed, "cases": 75, "original_successes": original_success,
            "no_head_conditioning_successes": ablated_success,
            "changed_case_outcomes": sum(value != 0 for value in differences.values()),
            "original_operation_counts": original["summary"]["operation_counts"],
            "no_head_conditioning_operation_counts": changed["summary"]["operation_counts"],
            "original_checkpoint_sha256": _hash(before_checkpoint),
            "ablated_checkpoint_sha256": _hash(after_checkpoint),
            "original_result_sha256": _hash(before_result),
            "ablated_result_sha256": _hash(after_result),
        })
    qids = sorted(paired[SEEDS[0]])
    if any(set(paired[seed]) != set(qids) for seed in SEEDS):
        raise ValueError("Seeds must share validation questions")
    by_query = [sum(paired[seed][qid] for seed in SEEDS) for qid in qids]
    rng = random.Random(20261002)
    boot = sorted(sum(by_query[rng.randrange(len(qids))] for _ in qids) / (len(qids) * len(SEEDS))
                  for _ in range(bootstrap_samples))
    total = len(qids) * len(SEEDS)
    return {
        "split": "validation", "seeds": list(SEEDS), "episodes_per_seed": 200,
        "ablation": "zero chosen-operation and chosen-position inputs to downstream action heads",
        "runs": runs, "total_paired_cases": total,
        "original_successes": sum(row["original_successes"] for row in runs),
        "no_head_conditioning_successes": sum(row["no_head_conditioning_successes"] for row in runs),
        "changed_case_outcomes": sum(row["changed_case_outcomes"] for row in runs),
        "paired_original_minus_ablation_rate": sum(by_query) / total,
        "query_cluster_bootstrap_95pct_difference": [
            boot[int(.025 * (bootstrap_samples - 1))], boot[int(.975 * (bootstrap_samples - 1))],
        ],
        "bootstrap_seed": 20261002, "bootstrap_samples": bootstrap_samples,
        "note": "Development validation. Masks still constrain legal actions; this removes only the previous chosen-action inputs to the position and payload heads.",
    }


if __name__ == "__main__":
    result = summarize()
    output = ROOT_DIR / "results/ppo_head_conditioning_ablation_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
