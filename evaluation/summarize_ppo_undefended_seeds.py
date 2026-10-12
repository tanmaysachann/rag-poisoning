"""Audit matched undefended PPO, greedy-insertion, and random validation runs by seed."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

from config import ROOT_DIR


SEEDS = (42, 43, 44)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _paths(seed: int, root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    if seed == 42:
        policy = root / "results/ppo_validation_evaluation.json"
        checkpoint = root / "artifacts/poison_policy_ppo.pt"
        greedy = root / "results/edit_rollout_validation_greedy_proxy.jsonl"
        greedy_summary = root / "results/edit_rollout_validation_greedy_proxy.summary.json"
        random_rows = root / "results/edit_rollout_validation_random.jsonl"
        random_summary = root / "results/edit_rollout_validation_random.summary.json"
    else:
        policy = root / f"results/ppo_validation_seed{seed}.json"
        checkpoint = root / f"artifacts/poison_policy_ppo_seed{seed}.pt"
        greedy = root / f"results/edit_rollout_validation_greedy_proxy_seed{seed}.jsonl"
        greedy_summary = greedy.with_suffix(".summary.json")
        random_rows = root / f"results/edit_rollout_validation_random_seed{seed}.jsonl"
        random_summary = random_rows.with_suffix(".summary.json")
    return policy, checkpoint, greedy, greedy_summary, random_rows, random_summary


def compare(seed: int, root: Path = ROOT_DIR) -> tuple[dict, dict[str, int], dict[str, int]]:
    policy_path, checkpoint_path, greedy_path, greedy_summary_path, random_path, random_summary_path = _paths(seed, root)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    greedy_rows = _jsonl(greedy_path)
    random_rows = _jsonl(random_path)
    greedy_summary = json.loads(greedy_summary_path.read_text(encoding="utf-8"))
    random_summary = json.loads(random_summary_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (config["seed"] != seed or config["defender_aware"] or config.get("replace_existing", False)
            or policy["summary"]["split"] != "validation" or policy["summary"]["defender_aware"]
            or policy["summary"].get("replace_existing", False)):
        raise ValueError(f"Seed {seed}: experiment configuration differs")
    policies = {row["qid"]: row for row in policy["cases"]}
    greedies = {row["qid"]: row for row in greedy_rows}
    randoms = {row["qid"]: row for row in random_rows}
    if (len(policies) != 75 or len(greedies) != 75 or len(randoms) != 75
            or set(policies) != set(greedies) or set(policies) != set(randoms)):
        raise ValueError(f"Seed {seed}: missing or duplicate validation questions")
    order = list(greedies)
    if any(policies[qid]["wrong_answer"] != greedies[qid]["wrong_answer"]
           or policies[qid]["wrong_answer"] != randoms[qid]["wrong_answer"]
           or greedies[qid].get("seed", seed + index) != seed + index
           or randoms[qid].get("seed", seed + index) != seed + index
           for index, qid in enumerate(order)):
        raise ValueError(f"Seed {seed}: wrong-answer or rollout schedule differs")
    paired_greedy = {
        qid: int(bool(row["terminal"]["attack_success"])) - int(bool(greedies[qid]["terminal"]["attack_success"]))
        for qid, row in policies.items()
    }
    paired_random = {
        qid: int(bool(row["terminal"]["attack_success"])) - int(bool(randoms[qid]["terminal"]["attack_success"]))
        for qid, row in policies.items()
    }
    policy_success = sum(row["terminal"]["attack_success"] for row in policies.values())
    greedy_success = sum(row["terminal"]["attack_success"] for row in greedies.values())
    random_success = sum(row["terminal"]["attack_success"] for row in randoms.values())
    if (greedy_summary["strategy"] != "greedy_proxy" or greedy_summary["split"] != "validation"
            or greedy_summary["cases"] != 75
            or round(greedy_summary["attack_success_rate"] * 75) != greedy_success):
        raise ValueError(f"Seed {seed}: greedy-insertion summary differs")
    if (random_summary["strategy"] != "random" or random_summary["split"] != "validation"
            or random_summary["cases"] != 75
            or round(random_summary["attack_success_rate"] * 75) != random_success):
        raise ValueError(f"Seed {seed}: random-edit summary differs")
    if round(policy["summary"]["attack_success_rate"] * 75) != policy_success:
        raise ValueError(f"Seed {seed}: PPO summary disagrees with its cases")
    summary = {
        "seed": seed, "cases": 75, "ppo_successes": policy_success,
        "greedy_insertion_successes": greedy_success,
        "random_edit_successes": random_success,
        "ppo_only_successes": sum(value == 1 for value in paired_greedy.values()),
        "greedy_only_successes": sum(value == -1 for value in paired_greedy.values()),
        "paired_difference_count": policy_success - greedy_success,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "policy_result_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "greedy_result_sha256": hashlib.sha256(greedy_path.read_bytes()).hexdigest(),
        "random_result_sha256": hashlib.sha256(random_path.read_bytes()).hexdigest(),
    }
    return summary, paired_greedy, paired_random


def summarize(root: Path = ROOT_DIR, *, bootstrap_samples: int = 10000) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rows_and_pairs = [compare(seed, root) for seed in SEEDS]
    runs = [item[0] for item in rows_and_pairs]
    pairs = [item[1] for item in rows_and_pairs]
    random_pairs = [item[2] for item in rows_and_pairs]
    qids = sorted(pairs[0])
    if any(set(pair) != set(qids) for pair in pairs):
        raise ValueError("Seeds do not evaluate the same validation questions")
    per_query_differences = [sum(pair[qid] for pair in pairs) for qid in qids]
    per_query_random_differences = [sum(pair[qid] for pair in random_pairs) for qid in qids]
    rng = random.Random(20261012)
    boot = sorted(
        sum(per_query_differences[rng.randrange(len(qids))] for _ in qids)
        / (len(qids) * len(SEEDS))
        for _ in range(bootstrap_samples)
    )
    random_rng = random.Random(20261013)
    random_boot = sorted(
        sum(per_query_random_differences[random_rng.randrange(len(qids))] for _ in qids)
        / (len(qids) * len(SEEDS))
        for _ in range(bootstrap_samples)
    )
    total = 75 * len(SEEDS)
    return {
        "split": "validation", "defender": "none_undefended_insertion",
        "attack_surface": "insertion", "cases_per_seed": 75,
        "seeds": list(SEEDS), "runs": runs,
        "ppo_successes": sum(row["ppo_successes"] for row in runs),
        "greedy_insertion_successes": sum(row["greedy_insertion_successes"] for row in runs),
        "random_edit_successes": sum(row["random_edit_successes"] for row in runs),
        "total_paired_cases": total,
        "ppo_success_rate": sum(row["ppo_successes"] for row in runs) / total,
        "greedy_insertion_success_rate": sum(row["greedy_insertion_successes"] for row in runs) / total,
        "random_edit_success_rate": sum(row["random_edit_successes"] for row in runs) / total,
        "paired_difference_rate": sum(per_query_differences) / total,
        "ppo_minus_random_difference_rate": sum(per_query_random_differences) / total,
        "query_cluster_bootstrap_95pct_difference": [
            boot[int(0.025 * (bootstrap_samples - 1))],
            boot[int(0.975 * (bootstrap_samples - 1))],
        ],
        "query_cluster_bootstrap_95pct_ppo_minus_random": [
            random_boot[int(0.025 * (bootstrap_samples - 1))],
            random_boot[int(0.975 * (bootstrap_samples - 1))],
        ],
        "bootstrap_seed": 20261012, "bootstrap_samples": bootstrap_samples,
        "note": "Development validation; the three seeds reuse the same 75 questions. The interval resamples questions as clusters and describes this comparison, not unseen-corpus generalization.",
    }


def main() -> None:
    result = summarize()
    output = ROOT_DIR / "results/ppo_multiseed_undefended_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
