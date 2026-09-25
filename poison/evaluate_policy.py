"""Evaluate a frozen PPO checkpoint on train or validation edit episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hashlib
import joblib
import torch

from attack.baselines import choose_wrong_answer
from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from poison.actions import OPERATIONS
from poison.env import DocumentEditEnv, STATE_VERSION
from poison.policy import FactoredActorCritic
from detect.research_detector import ResearchDetector


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS_DIR / "poison_policy_ppo.pt")
    parser.add_argument("--detector", type=Path, default=ARTIFACTS_DIR / "research_detector_hashing.joblib")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("Limit must be positive")
    validate_benchmark(args.benchmark)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["state_version"] != STATE_VERSION:
        raise ValueError("Policy state schema does not match environment")
    policy = FactoredActorCritic()
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    defender_aware = checkpoint["config"].get("defender_aware", False)
    replace_existing = checkpoint["config"].get("replace_existing", False)
    detector = None
    if defender_aware:
        if hashlib.sha256(args.detector.read_bytes()).hexdigest() != checkpoint["config"]["detector_sha256"]:
            raise ValueError("Detector artifact differs from frozen training defender")
        detector = joblib.load(args.detector)
        if not isinstance(detector, ResearchDetector):
            raise ValueError("Wrong detector artifact type")
    queries = _read_jsonl(args.benchmark / args.split / "queries.jsonl")[:args.limit]
    answer_pool = [row["answer_aliases"][0] for row in _read_jsonl(args.benchmark / "train" / "queries.jsonl")]
    records = []
    for index, query in enumerate(queries):
        wrong = choose_wrong_answer(query["answer_aliases"], answer_pool, args.seed + index)
        env = DocumentEditEnv(
            args.benchmark / args.split / "corpus.jsonl", query, wrong,
            max_steps=3, detector=detector, replace_existing=replace_existing,
        )
        state, info = env.reset(seed=args.seed + index)
        trace = []
        while True:
            action, _, _, _ = policy.act(state, info["action_mask"], deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            trace.append({"operation": action.operation, "position": action.position, "payload": action.payload, "reward": reward})
            if terminated or truncated:
                records.append({
                    "qid": query["qid"], "wrong_answer": wrong, "trace": trace,
                    "terminal": info["terminal"], "final_document": info["document"],
                })
                break
    output = RESULTS_DIR / f"ppo_{args.split}{'_defender' if defender_aware else ''}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split, "cases": len(records), "policy_state_version": STATE_VERSION,
        "retrieval_rate": sum(row["terminal"]["retrieved"] for row in records) / len(records),
        "attack_success_rate": sum(row["terminal"]["attack_success"] for row in records) / len(records),
        "defended_attack_success_rate": (
            sum(row["terminal"]["defended_attack_success"] for row in records) / len(records)
            if defender_aware else None
        ),
        "attack_detection_rate": (
            sum(row["terminal"]["attack_doc_detected"] is True for row in records) / len(records)
            if defender_aware else None
        ),
        "defender_aware": defender_aware, "replace_existing": replace_existing,
        "operation_counts": {operation: sum(step["operation"] == operation for row in records for step in row["trace"]) for operation in OPERATIONS},
        "checkpoint": str(args.checkpoint),
    }
    output.write_text(json.dumps({"summary": summary, "cases": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
