"""Train/evaluate the frozen v14 one-step PPO quarantine policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from detect.research_detector import threshold_for_fpr

TRAIN_FAMILIES = ("advdecoding", "content_poison", "corpus_poison", "garag")
HELDOUT = ("gaslite", "poisonedrag_black")
SEEDS = (11, 22, 33)


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scale_features(raw: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    transformed = np.sign(raw) * np.log1p(np.abs(raw))
    scaler = StandardScaler().fit(transformed[train_indices])
    return np.clip(scaler.transform(transformed), -8, 8).astype(np.float32), scaler


def train_supervised(x: torch.Tensor, labels: torch.Tensor, output: Path) -> Network:
    torch.manual_seed(20261018)
    np.random.seed(20261018)
    actor = Network()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(20261018)
    for _ in range(80):
        for indices in torch.randperm(len(x), generator=generator).split(128):
            logits = actor(x[indices])
            weights = torch.where(labels[indices] == 0, 10.0, 1.0)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logits, labels[indices], reduction="none") * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(actor.state_dict(), output)
    return actor


def train_ppo(warmstart: Network, x: torch.Tensor, labels: torch.Tensor,
              seed: int, output: Path) -> tuple[Network, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    actor = Network()
    actor.load_state_dict(warmstart.state_dict())
    critic = Network()
    optimizer = torch.optim.AdamW([*actor.parameters(), *critic.parameters()], lr=0.0003)
    generator = torch.Generator().manual_seed(seed)
    rewards = []
    for _ in range(60):
        chosen = torch.randint(len(x), (1024,), generator=generator)
        batch_x, batch_y = x[chosen], labels[chosen]
        with torch.no_grad():
            old_logits = actor(batch_x)
            old_distribution = torch.distributions.Bernoulli(logits=old_logits)
            actions = old_distribution.sample()
            old_logp = old_distribution.log_prob(actions)
            reward = torch.where(batch_y == 1,
                                 torch.where(actions == 1, 1.0, -1.0),
                                 torch.where(actions == 1, -10.0, 0.0))
            advantages = reward - critic(batch_x)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            rewards.append(float(reward.mean()))
        for _ in range(4):
            for positions in torch.randperm(1024, generator=generator).split(256):
                distribution = torch.distributions.Bernoulli(logits=actor(batch_x[positions]))
                logp = distribution.log_prob(actions[positions])
                ratio = (logp - old_logp[positions]).exp()
                advantage = advantages[positions]
                policy_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(0.8, 1.2) * advantage,
                ).mean()
                value_loss = nn.functional.mse_loss(critic(batch_x[positions]), reward[positions])
                loss = policy_loss + 0.5 * value_loss - 0.003 * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    torch.save(actor.state_dict(), output)
    return actor, {"sampled_mean_reward_first_10": round(float(np.mean(rewards[:10])), 4),
                   "sampled_mean_reward_last_10": round(float(np.mean(rewards[-10:])), 4),
                   "sampled_actions": 60 * 1024}


def evaluate(rows: list[dict], scores: np.ndarray, cutoff: float) -> dict:
    selected = defaultdict(list)
    for row, score in zip(rows, scores):
        if row["split"] in {"development", "external"}:
            selected[row["kind"]].append((row["qid"], bool(score >= cutoff)))
    result = {"threshold": float(cutoff), "clean": {}, "reference": {}, "families": {}}
    for kind in ("clean", "reference"):
        values = selected[kind]
        result[kind] = {"documents": len(values), "false_flags": sum(flag for _, flag in values)}
    for kind in (*TRAIN_FAMILIES, *HELDOUT, "hotpot_poisonedrag"):
        values = selected[kind]
        questions = defaultdict(bool)
        for qid, flag in values:
            questions[qid] |= flag
        result["families"][kind] = {
            "documents": len(values), "caught": sum(flag for _, flag in values),
            "questions": len(questions), "questions_with_any_caught": sum(questions.values()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/defender_bandit_v14"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/defender_bandit_v14"))
    parser.add_argument("--output", type=Path, default=Path("results/defender_bandit_v14.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    feature_file = args.data / "features.jsonl"
    if sha(feature_file) != manifest["feature_sha256"]:
        raise ValueError("Frozen features changed")
    rows = [json.loads(line) for line in feature_file.open(encoding="utf-8")]
    raw = np.asarray([row["features"] for row in rows], dtype=np.float64)
    if raw.shape[1] != 9 or not np.isfinite(raw).all():
        raise ValueError("Invalid feature matrix")
    train_indices = np.asarray([i for i, row in enumerate(rows)
                                if row["split"] == "train" and row["kind"] != "reference"])
    cal_clean = np.asarray([i for i, row in enumerate(rows)
                            if row["split"] == "calibration" and row["kind"] == "clean"])
    if len(train_indices) < 500 or len(cal_clean) != 210:
        raise ValueError(f"Unexpected training/calibration rows: {len(train_indices)}/{len(cal_clean)}")
    x_all, scaler = scale_features(raw, train_indices)
    x = torch.from_numpy(x_all[train_indices])
    labels = torch.tensor([rows[i]["label"] for i in train_indices], dtype=torch.float32)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    args.models.mkdir(parents=True, exist_ok=True)
    preprocessing_path = args.models / "preprocessing.json"
    preprocessing_path.write_text(json.dumps({
        "feature_names": ["mahalanobis", "isolation_forest", "relevance_spike",
                          "instruction_pattern", "url_pattern", "authority_cue",
                          "query_doc_cosine", "query_copy_ratio", "v3_risk"],
        "transform": "sign(x) * log1p(abs(x)); StandardScaler fit on train rows; clip [-8,8]",
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "training_feature_sha256": manifest["feature_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    warm_path = args.models / "warmstart.pt"
    warmstart = train_supervised(x, labels, warm_path)
    trained = {"supervised_warmstart": warmstart}
    training_log = {}
    for seed in SEEDS:
        name = f"ppo_seed_{seed}"
        model_path = args.models / f"{name}.pt"
        trained[name], training_log[name] = train_ppo(warmstart, x, labels, seed, model_path)
        print(f"trained {name}: {training_log[name]}", flush=True)
    # Policy weights and thresholds are now fixed before reading development outcomes.
    scores = {"frozen_v3": raw[:, -1]}
    with torch.no_grad():
        all_x = torch.from_numpy(x_all)
        for name, model in trained.items():
            scores[name] = torch.sigmoid(model(all_x)).numpy().astype(np.float64)
    metrics = {}
    thresholds = {}
    for name, values in scores.items():
        cutoff = threshold_for_fpr(values[cal_clean], 0.01)
        thresholds[name] = cutoff
        metrics[name] = evaluate(rows, values, cutoff)
        metrics[name]["calibration_clean_false_flags"] = int(np.sum(values[cal_clean] >= cutoff))
    baseline_por = max(metrics[name]["families"]["poisonedrag_black"]["caught"]
                       for name in ("frozen_v3", "supervised_warmstart"))
    baseline_gas = max(metrics[name]["families"]["gaslite"]["caught"]
                       for name in ("frozen_v3", "supervised_warmstart"))
    seed_pass = {}
    for seed in SEEDS:
        name = f"ppo_seed_{seed}"
        value = metrics[name]
        seed_pass[name] = (
            value["families"]["poisonedrag_black"]["caught"] > baseline_por
            and value["clean"]["false_flags"] <= 2
            and value["families"]["gaslite"]["caught"] >= baseline_gas - 10
        )
    report = {
        "protocol": "research_protocols/defender_bandit_v14.md",
        "feature_sha256": manifest["feature_sha256"],
        "feature_manifest_sha256": sha(args.data / "manifest.json"),
        "preprocessing_sha256": sha(preprocessing_path),
        "model_sha256": {"supervised_warmstart": sha(warm_path),
                         **{f"ppo_seed_{seed}": sha(args.models / f"ppo_seed_{seed}.pt")
                            for seed in SEEDS}},
        "rows": len(rows), "train_rows": len(train_indices), "calibration_clean_rows": len(cal_clean),
        "training_log": training_log, "models": metrics,
        "gate": {"baseline_poisonedrag_black_caught": baseline_por,
                 "baseline_gaslite_caught": baseline_gas,
                 "per_seed_pass": seed_pass,
                 "passed": sum(seed_pass.values()) >= 2},
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "limitations": ["Development questions were inspected in earlier v9-v13 experiments.",
                        "Source labels do not establish factual contradiction.",
                        "HotpotQA attack-only rows have no matched clean false-positive estimate.",
                        "One-step reward does not measure downstream answer compromise."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
