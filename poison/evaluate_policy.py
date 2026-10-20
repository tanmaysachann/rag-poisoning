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
from detect.snapshot_integrity_detector_v22 import SnapshotIntegrityDetector


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
    parser.add_argument("--snapshot-detector", type=Path,
                        help="Pinned pre-attack corpus snapshot for this evaluation split")
    parser.add_argument("--snapshot-containment", action="store_true",
                        help="Replay a snapshot-trained policy against V23 containment")
    parser.add_argument("--snapshot-identity", action="store_true",
                        help="Enforce known document ID/hash matches from the pinned corpus")
    parser.add_argument("--output", type=Path, help="Keep each seed's validation evidence separate")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("Limit must be positive")
    if args.snapshot_containment and args.snapshot_detector is None:
        raise ValueError("--snapshot-containment requires --snapshot-detector")
    if args.snapshot_identity and args.snapshot_detector is None:
        raise ValueError("--snapshot-identity requires --snapshot-detector")
    validate_benchmark(args.benchmark)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["state_version"] != STATE_VERSION:
        raise ValueError("Policy state schema does not match environment")
    policy = FactoredActorCritic(
        conditional_heads=checkpoint["config"].get("conditional_heads", True)
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    defender_aware = checkpoint["config"].get("defender_aware", False)
    detector_kind = checkpoint["config"].get("detector_kind", "research_detector_v3")
    replace_existing = checkpoint["config"].get("replace_existing", False)
    detector = None
    evaluation_detector_sha256 = None
    if defender_aware:
        if detector_kind in {"snapshot_integrity_v22", "snapshot_containment_v23"}:
            if args.snapshot_detector is None:
                raise ValueError("Snapshot-trained checkpoint requires --snapshot-detector")
            corpus = args.benchmark / args.split / "corpus.jsonl"
            detector = SnapshotIntegrityDetector(
                args.snapshot_detector,
                expected_corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                threshold=checkpoint["config"]["detector_threshold"],
                containment=(args.snapshot_containment or
                             detector_kind == "snapshot_containment_v23"),
                identity_corpus=corpus if args.snapshot_identity else None)
            evaluation_detector_sha256 = detector.manifest_sha256
            if args.split == "train" and evaluation_detector_sha256 != checkpoint["config"]["detector_sha256"]:
                raise ValueError("Training snapshot differs from checkpoint")
        elif detector_kind == "research_detector_v3":
            evaluation_detector_sha256 = hashlib.sha256(args.detector.read_bytes()).hexdigest()
            if evaluation_detector_sha256 != checkpoint["config"]["detector_sha256"]:
                raise ValueError("Detector artifact differs from frozen training defender")
            detector = joblib.load(args.detector)
            if not isinstance(detector, ResearchDetector):
                raise ValueError("Wrong detector artifact type")
        else:
            raise ValueError(f"Unsupported defender: {detector_kind}")
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
    output = args.output or RESULTS_DIR / (
        f"ppo_{args.split}_snapshot_{'v24_identity' if args.snapshot_identity else 'v23_containment' if (args.snapshot_containment or detector_kind == 'snapshot_containment_v23') else 'v22'}_evaluation.json"
        if detector_kind in {"snapshot_integrity_v22", "snapshot_containment_v23"}
        else f"ppo_{args.split}{'_defender' if defender_aware else ''}_evaluation.json")
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
        "defender_aware": defender_aware,
        "detector_kind": ("snapshot_identity_v24" if args.snapshot_identity
                          else "snapshot_containment_v23" if (args.snapshot_containment or detector_kind == "snapshot_containment_v23")
                          else detector_kind if defender_aware else "none"),
        "evaluation_detector_sha256": evaluation_detector_sha256,
        "replace_existing": replace_existing,
        "operation_counts": {operation: sum(step["operation"] == operation for row in records for step in row["trace"]) for operation in OPERATIONS},
        "checkpoint": str(args.checkpoint),
    }
    output.write_text(json.dumps({"summary": summary, "cases": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if isinstance(detector, SnapshotIntegrityDetector):
        detector.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
