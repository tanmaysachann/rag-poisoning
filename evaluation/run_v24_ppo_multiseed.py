"""Run two more PPO attacker seeds against one frozen V23 defender.

Together with the existing 20261021 run, this gives three comparable seeds.
Each seed gets its own checkpoint, validation replay, and fixed/random edit
baselines. V24 known-ID identity is a separate frozen-text replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from evaluation.evaluate_adaptive_ppo_v24 import same_attacks


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
TRAIN_SNAPSHOT = ROOT / "data/benchmark_snapshot_v22_train"
VALIDATION_SNAPSHOT = ROOT / "data/benchmark_snapshot_v22"
SEEDS = (20261021, 20261022, 20261023)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_if_absent(command: list[str], outputs: list[Path]) -> None:
    exists = [path.is_file() for path in outputs]
    if all(exists):
        return
    if any(exists):
        raise ValueError(f"Partial artifact pair requires manual inspection: {outputs}")
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not all(path.is_file() for path in outputs):
        raise RuntimeError("Command returned without its required output artifacts")


def _paths(seed: int) -> dict[str, Path]:
    if seed == 20261021:
        return {
            "checkpoint": ARTIFACTS / "poison_policy_ppo_snapshot_v23.pt",
            "training": RESULTS / "ppo_train_snapshot_v23.jsonl",
            "v23": RESULTS / "ppo_validation_snapshot_v23_retrained_evaluation.json",
            "v24": RESULTS / "ppo_validation_snapshot_v24_identity_replay.json",
            "fixed": RESULTS / "edit_rollout_validation_fixed_substitution_snapshot_v23_seed20261021.summary.json",
            "random": RESULTS / "edit_rollout_validation_random_snapshot_v23_seed20261021.summary.json",
        }
    stem = f"snapshot_v23_seed{seed}"
    return {
        "checkpoint": ARTIFACTS / f"poison_policy_ppo_{stem}.pt",
        "training": RESULTS / f"ppo_train_{stem}.jsonl",
        "v23": RESULTS / f"ppo_validation_{stem}.json",
        "v24": RESULTS / f"ppo_validation_{stem}_identity.json",
        "fixed": RESULTS / f"edit_rollout_validation_fixed_substitution_{stem}.summary.json",
        "random": RESULTS / f"edit_rollout_validation_random_{stem}.summary.json",
    }


def _new_seed(seed: int) -> None:
    paths = _paths(seed)
    common = ["--defender-aware", "--replace-existing", "--snapshot-containment"]
    _run_if_absent([
        sys.executable, "-m", "poison.train", "--seed", str(seed),
        "--batches", "10", "--episodes-per-batch", "20", *common,
        "--snapshot-detector", str(TRAIN_SNAPSHOT),
        "--output", str(paths["checkpoint"]),
        "--metrics-output", str(paths["training"]),
    ], [paths["checkpoint"], paths["training"]])
    for key, identity in (("v23", False), ("v24", True)):
        _run_if_absent([
            sys.executable, "-m", "poison.evaluate_policy", "--seed", str(seed),
            "--limit", "75", "--checkpoint", str(paths["checkpoint"]),
            "--snapshot-detector", str(VALIDATION_SNAPSHOT),
            "--snapshot-containment", *( ["--snapshot-identity"] if identity else [] ),
            "--output", str(paths[key]),
        ], [paths[key]])
    for strategy in ("fixed_substitution", "random"):
        target = RESULTS / f"edit_rollout_validation_{strategy}_snapshot_v23_seed{seed}.jsonl"
        _run_if_absent([
            sys.executable, "-m", "poison.evaluate_rollouts", "--split", "validation",
            "--limit", "75", "--seed", str(seed), "--strategy", strategy,
            *common, "--snapshot-detector", str(VALIDATION_SNAPSHOT),
            "--output", str(target),
        ], [target, target.with_suffix(".summary.json")])


def _count(data: dict, key: str) -> int:
    return sum(bool(row["terminal"][key]) for row in data["cases"])


def report(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Multiseed report already exists: {output}")
    rounds = []
    for seed in SEEDS:
        paths = _paths(seed)
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Missing seed {seed} artifact")
        before = json.loads(paths["v23"].read_text(encoding="utf-8"))
        after = json.loads(paths["v24"].read_text(encoding="utf-8"))
        same_attacks(before, after)
        if len(before["cases"]) != 75:
            raise ValueError("Unexpected validation case count")
        fixed = json.loads(paths["fixed"].read_text(encoding="utf-8"))
        random = json.loads(paths["random"].read_text(encoding="utf-8"))
        if any(row["seed"] != seed or row["cases"] != 75 for row in (fixed, random)):
            raise ValueError("Baseline schedule mismatch")
        rounds.append({
            "seed": seed, "training_episodes": 200,
            "v23_defended_successes": _count(before, "defended_attack_success"),
            "v24_known_id_defended_successes": _count(after, "defended_attack_success"),
            "fixed_defended_successes": round(fixed["defended_attack_success_rate"] * 75),
            "random_defended_successes": round(random["defended_attack_success_rate"] * 75),
            "artifact_sha256": {name: sha(path) for name, path in paths.items()},
        })
    successes = [row["v23_defended_successes"] for row in rounds]
    payload = {
        "protocol": "v24-ppo-attacker-three-seed-v23-containment-v1",
        "status": "development_validation; no independent blind RL evaluation",
        "training_questions_available": 350, "episodes_per_seed": 200,
        "validation_questions": 75, "rounds": rounds,
        "v23_ppo_defended_successes_mean": statistics.mean(successes),
        "v23_ppo_defended_successes_range": [min(successes), max(successes)],
        "v23_ppo_defended_successes_sample_stdev": statistics.stdev(successes),
        "frozen_defender_snapshot_sha256": sha(VALIDATION_SNAPSHOT / "manifest.json"),
        "limits": [
            "Three seeds against V23 are a variability check, not a statistical superiority proof.",
            "The 75 validation questions were used during prior development.",
            "The V24 replay uses a pre-attack known-ID hash; new inserted documents lack it.",
            "PPO optimizes the attacker; the V24 detector is supervised plus source integrity.",
        ],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "v24_ppo_multiseed.json")
    args = parser.parse_args()
    for seed in SEEDS[1:]:
        _new_seed(seed)
    print(json.dumps(report(args.output), indent=2))
