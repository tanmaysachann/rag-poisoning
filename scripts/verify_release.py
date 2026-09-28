"""Run the local product and evidence checks without evaluating the v2 test split."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.summarize_ppo_seeds import summarize
from evaluation.summarize_ppo_detection_ablation import summarize as summarize_detection_ablation
from evaluation.summarize_ppo_proxy_ablation import summarize as summarize_proxy_ablation
from evaluation.summarize_ppo_cache_ablation import summarize as summarize_cache_ablation
from evaluation.summarize_ppo_head_ablation import summarize as summarize_head_ablation
from evaluation.verify_v2_development import verify as verify_v2_development
from evaluation.verify_contriever_result import verify as verify_contriever_result
from evaluation.verify_unseen_templates import verify as verify_unseen_templates
from evaluation.probe_v2_peer_failover import summarize as summarize_v2_peer_failover
from evaluation.verify_poison_budget import verify as verify_poison_budget


def run(*command: str) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the isolated offline rebuild")
    args = parser.parse_args()

    run(sys.executable, "scripts/doctor.py")
    run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q")
    run(sys.executable, "-m", "evaluation.verify_test_v1")
    run(sys.executable, "-m", "data.validate_multisupport_benchmark")
    print(json.dumps(verify_v2_development(), indent=2), flush=True)
    print(json.dumps(verify_contriever_result(), indent=2), flush=True)
    print(json.dumps(verify_unseen_templates(), indent=2), flush=True)
    print(json.dumps(verify_poison_budget(), indent=2), flush=True)
    peer_probe_path = ROOT / "results/v2_peer_failover_validation_probe.json"
    if json.loads(peer_probe_path.read_text(encoding="utf-8")) != summarize_v2_peer_failover():
        raise ValueError(f"Saved v2 peer-failover probe is stale: {peer_probe_path}")
    print("V2 peer-failover diagnostic: verified against corrected validation cases", flush=True)

    saved_path = ROOT / "results/ppo_multiseed_validation.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    regenerated = summarize()
    if saved != regenerated:
        raise ValueError(f"Saved PPO comparison is stale: {saved_path}")
    print("PPO three-seed comparison: verified against raw runs and checkpoints", flush=True)
    ablation_path = ROOT / "results/ppo_detection_reward_ablation_validation.json"
    if json.loads(ablation_path.read_text(encoding="utf-8")) != summarize_detection_ablation():
        raise ValueError(f"Saved PPO detection-reward ablation is stale: {ablation_path}")
    print("PPO detection-reward ablation: verified against raw runs and checkpoints", flush=True)
    proxy_path = ROOT / "results/ppo_proxy_value_ablation_validation.json"
    if json.loads(proxy_path.read_text(encoding="utf-8")) != summarize_proxy_ablation():
        raise ValueError(f"Saved PPO proxy-value ablation is stale: {proxy_path}")
    print("PPO proxy-value ablation: verified against raw runs and checkpoints", flush=True)
    cache_path = ROOT / "results/ppo_cache_ablation_validation.json"
    if json.loads(cache_path.read_text(encoding="utf-8")) != summarize_cache_ablation():
        raise ValueError(f"Saved PPO cache ablation is stale: {cache_path}")
    print("PPO edit-cache ablation: verified against raw runs and checkpoints", flush=True)
    head_path = ROOT / "results/ppo_head_conditioning_ablation_validation.json"
    if json.loads(head_path.read_text(encoding="utf-8")) != summarize_head_ablation():
        raise ValueError(f"Saved PPO head-conditioning ablation is stale: {head_path}")
    print("PPO action-head ablation: verified against raw runs and checkpoints", flush=True)

    node = shutil.which("node")
    if node:
        run(node, "--check", "frontend/app.js")
        run(node, "--check", "frontend/research.js")
    else:
        print("Node.js unavailable; JavaScript syntax check skipped", flush=True)

    if not args.skip_smoke:
        run(sys.executable, "scripts/smoke_rebuild.py")
    print("\nRelease verification passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
