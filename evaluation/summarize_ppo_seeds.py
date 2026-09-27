"""Audit matched defender-aware PPO, fixed, and random validation runs by seed."""

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


def compare(seed: int, root: Path = ROOT_DIR) -> tuple[dict, dict[str, int], dict[str, int]]:
    policy_path = root / f"results/ppo_validation_defender_seed{seed}.json"
    baseline_path = root / f"results/fixed_substitution_validation_seed{seed}.jsonl"
    random_path = root / f"results/random_rollout_validation_seed{seed}.jsonl"
    checkpoint_path = root / ("artifacts/poison_policy_ppo_defender.pt" if seed == 42
                              else f"artifacts/poison_policy_ppo_defender_seed{seed}.pt")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    baseline = _jsonl(baseline_path)
    random_rows = _jsonl(random_path)
    random_summary = json.loads(random_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (config["seed"] != seed or not config["defender_aware"] or not config["replace_existing"]
            or policy["summary"]["split"] != "validation" or not policy["summary"]["defender_aware"]
            or not policy["summary"]["replace_existing"]):
        raise ValueError(f"Seed {seed}: experiment configuration differs")
    policies = {row["qid"]: row for row in policy["cases"]}
    baselines = {row["qid"]: row for row in baseline}
    randoms = {row["qid"]: row for row in random_rows}
    if (len(policies) != 75 or len(baselines) != 75 or len(randoms) != 75
            or set(policies) != set(baselines) or set(policies) != set(randoms)):
        raise ValueError(f"Seed {seed}: missing or duplicate validation questions")
    if any(policies[qid]["wrong_answer"] != baselines[qid]["wrong_answer"]
           or policies[qid]["wrong_answer"] != randoms[qid]["wrong_answer"]
           or baselines[qid]["seed"] != seed + index
           or randoms[qid]["seed"] != seed + index
           or randoms[qid]["strategy"] != "random"
           for index, qid in enumerate(baselines)):
        raise ValueError(f"Seed {seed}: wrong-answer or rollout schedule differs")
    paired = {
        qid: int(bool(row["terminal"]["defended_attack_success"]))
        - int(bool(baselines[qid]["terminal"]["defended_attack_success"]))
        for qid, row in policies.items()
    }
    paired_random = {
        qid: int(bool(row["terminal"]["defended_attack_success"]))
        - int(bool(randoms[qid]["terminal"]["defended_attack_success"]))
        for qid, row in policies.items()
    }
    policy_success = sum(row["terminal"]["defended_attack_success"] for row in policies.values())
    baseline_success = sum(row["terminal"]["defended_attack_success"] for row in baselines.values())
    random_success = sum(row["terminal"]["defended_attack_success"] for row in randoms.values())
    if (random_summary["strategy"] != "random" or random_summary["seed"] != seed
            or random_summary["split"] != "validation" or random_summary["cases"] != 75
            or not random_summary["defender_aware"] or not random_summary["replace_existing"]
            or round(random_summary["defended_attack_success_rate"] * 75) != random_success):
        raise ValueError(f"Seed {seed}: random-edit configuration or summary differs")
    if round(policy["summary"]["defended_attack_success_rate"] * 75) != policy_success:
        raise ValueError(f"Seed {seed}: PPO summary disagrees with its cases")
    summary = {
        "seed": seed, "cases": 75, "ppo_successes": policy_success,
        "fixed_substitution_successes": baseline_success,
        "random_edit_successes": random_success,
        "ppo_only_successes": sum(value == 1 for value in paired.values()),
        "fixed_only_successes": sum(value == -1 for value in paired.values()),
        "paired_difference_count": policy_success - baseline_success,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "policy_result_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "baseline_result_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "random_result_sha256": hashlib.sha256(random_path.read_bytes()).hexdigest(),
    }
    return summary, paired, paired_random


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
    rng = random.Random(20260927)
    boot = sorted(
        sum(per_query_differences[rng.randrange(len(qids))] for _ in qids)
        / (len(qids) * len(SEEDS))
        for _ in range(bootstrap_samples)
    )
    random_rng = random.Random(20260928)
    random_boot = sorted(
        sum(per_query_random_differences[random_rng.randrange(len(qids))] for _ in qids)
        / (len(qids) * len(SEEDS))
        for _ in range(bootstrap_samples)
    )
    total = 75 * len(SEEDS)
    result = {
        "split": "validation", "defender": "frozen hashing research detector",
        "attack_surface": "accepted_ingest_replacement", "cases_per_seed": 75,
        "seeds": list(SEEDS), "runs": runs,
        "ppo_successes": sum(row["ppo_successes"] for row in runs),
        "fixed_substitution_successes": sum(row["fixed_substitution_successes"] for row in runs),
        "random_edit_successes": sum(row["random_edit_successes"] for row in runs),
        "total_paired_cases": total,
        "ppo_success_rate": sum(row["ppo_successes"] for row in runs) / total,
        "fixed_substitution_success_rate": sum(row["fixed_substitution_successes"] for row in runs) / total,
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
        "bootstrap_seed": 20260927, "bootstrap_samples": bootstrap_samples,
        "note": "Development validation; the three seeds reuse the same 75 questions. The interval resamples questions as clusters and describes this comparison, not unseen-corpus generalization.",
    }
    return result


def main() -> None:
    result = summarize()
    output = ROOT_DIR / "results/ppo_multiseed_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
