"""Stage the hosted site plus the V24/V25/V26 research view and remote proxies."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The 91 MB scanner checkpoint belongs in the separate container service.
# Including it in the FastAPI site function would needlessly inflate redeploys.
SITE_EXCLUDED_FILES = {"artifacts/multifamily_v20/model.safetensors"}
OVERLAYS = (
    "backend/api.py",
    "backend/research_lab.py",
    "backend/v24_service.py",
    "backend/v25_service.py",
    "backend/v26_service.py",
    "pipeline/v24_answer_gate.py",
    # git archive stores CRLF for this JSONL on Windows; the pinned digest is LF.
    "data/benchmark_v3/validation/corpus.jsonl",
    "frontend/index.html",
    "frontend/v24.css",
    "frontend/v24.js",
    "frontend/v24_standalone.html",
    "results/v26_rule_frozen.json",
    "results/v26_fiqa_holdout.json",
    "results/v26_fiqa_answer_path.json",
    "results/v26_fiqa_role_stress_frozen.json",
    "results/v26_fiqa_role_stress.json",
    "results/v26_nfcorpus_development_replay.json",
    "results/v26_presentation_sample.json",
    "data/multifamily_v20/manifest.json",
    "data/multifamily_v20/evaluation.jsonl",
    "results/adaptive_ppo_v24.json",
    "results/product_v24.json",
    "results/v24_blind_scifact.json",
    "results/v24_answer_path_validation.json",
    "results/v24_answer_path_validation_admitted.json",
    "results/v24_ppo_multiseed.json",
)

# git archive expands some pinned LF files to CRLF on Windows. The older v1
# and v3 evidence verifiers hash their exact bytes, so restore those files
# from the locally verified checkout after staging the committed archive.
PINNED_EVIDENCE_OVERLAYS = (
    "attack/baselines.py",
    "attack/heldout_families.py",
    "data/benchmark/test/corpus.jsonl",
    "data/benchmark/test/queries.jsonl",
    "data/benchmark_v3/nq_challenge.jsonl",
    "data/benchmark_v3/test/corpus.jsonl",
    "data/benchmark_v3/test/queries.jsonl",
    "data/benchmark_v3/train/corpus.jsonl",
    "data/benchmark_v3/train/queries.jsonl",
    "data/benchmark_v3/validation/queries.jsonl",
    "evaluation/evaluate_v3_generalization.py",
    "evaluation/run_attack_baselines.py",
    "evaluation/run_defense.py",
    "evaluation/verify_v3_generalization.py",
    "pipeline/secure_rag.py",
    "research_protocols/v3_heldout_generalization.json",
    "retrieval/hybrid_retriever.py",
    "results/attack_baseline_test_greedy.jsonl",
    "results/attack_baseline_test_stealth.jsonl",
    "results/clean_answers_test_minilm_ranker_strict.cases.jsonl",
    "results/defense_test_greedy_accepted_ingest_minilm_ranker.cases.jsonl",
    "results/defense_test_stealth_accepted_ingest_minilm_ranker.cases.jsonl",
    "results/v3_attack_train_random.jsonl",
    "results/v3_attack_train_stealth.jsonl",
    "results/v3_attack_validation_greedy.jsonl",
    "results/v3_generalization_test.cases.jsonl",
    "results/v3_generalization_test_clean.cases.jsonl",
    "results/v3_generalization_validation.cases.jsonl",
    "results/v3_generalization_validation_clean.cases.jsonl",
)


def build(output: Path) -> dict:
    output = output.resolve()
    if ROOT.resolve() not in output.parents or output.exists():
        raise ValueError("Output must be a new directory inside this workspace")
    archive = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=zip", "HEAD"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for item in source.infolist():
            if item.filename in SITE_EXCLUDED_FILES:
                continue
            target = (output / item.filename).resolve()
            if output not in target.parents and target != output:
                raise ValueError("Unexpected path in frozen site archive")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(item))
    for name in (*OVERLAYS, *PINNED_EVIDENCE_OVERLAYS):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    report = {"bundle": str(output), "source_bytes": total,
              "source": "HEAD plus explicit V24/V25/V26 serving and evidence overlays",
              "overlays": list((*OVERLAYS, *PINNED_EVIDENCE_OVERLAYS))}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/site_v24_bundle")
    build(parser.parse_args().output)
