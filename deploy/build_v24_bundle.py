"""Stage a small V24 source bundle; the exact large snapshot is fetched at build."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "config.py",
    "backend/v24_service.py",
    "backend/v25_service.py",
    "data/build_trusted_snapshot_v12.py",
    "data/multifamily_v20/manifest.json",
    "data/multifamily_v20/evaluation.jsonl",
    "data/trusted_snapshot_v12/manifest.json",
    "detect/char_fluency.py",
    "detect/product_v22.py",
    "detect/product_v25.py",
    "detect/reference_conditioned_v9.py",
    "detect/research_detector.py",
    "detect/signals.py",
    "detect/snapshot_containment_v23.py",
    "detect/snapshot_near_duplicate.py",
    "retrieval/hybrid_retriever.py",
    "retrieval/trusted_snapshot_v12.py",
    "security/integrity.py",
    "security/provenance.py",
    "artifacts/multifamily_v20/config.json",
    "artifacts/multifamily_v20/model.safetensors",
    "artifacts/multifamily_v20/special_tokens_map.json",
    "artifacts/multifamily_v20/tokenizer.json",
    "artifacts/multifamily_v20/tokenizer_config.json",
    "artifacts/multifamily_v20/vocab.txt",
    "artifacts/fluency_v21.joblib",
    "artifacts/product_v24_calibration.joblib",
    "artifacts/research_detector_v3_hashing.joblib",
    "artifacts/research_detector_v3_hashing_web.joblib",
    "research_protocols/v3_web_detector.json",
    "results/adaptive_ppo_v24.json",
    "results/fusion_v21.json",
    "results/product_v24.json",
    "results/snapshot_containment_v23.json",
    "results/v25_rule_frozen.json",
    "results/v25_nfcorpus_holdout.json",
    "frontend/research.css",
    "frontend/styles.css",
    "frontend/v24.css",
    "frontend/v24.js",
    "frontend/v24_standalone.html",
)
TEMPLATE = ROOT / "deploy/v24_src"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(output: Path) -> dict:
    output = output.resolve()
    if ROOT.resolve() not in output.parents or output.exists():
        raise ValueError("Output must be a new directory inside this workspace")
    calibration = __import__("joblib").load(ROOT / "artifacts/product_v24_calibration.joblib")
    expected = {
        "artifacts/multifamily_v20/model.safetensors": calibration["neural_model_sha256"],
        "artifacts/fluency_v21.joblib": calibration["signal_model_sha256"],
        "artifacts/research_detector_v3_hashing.joblib": calibration["v3_model_sha256"],
        "data/multifamily_v20/manifest.json": calibration["data_manifest_sha256"],
        "data/trusted_snapshot_v12/manifest.json": calibration["snapshot_manifest_sha256"],
    }
    for name, digest in expected.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Pinned V24 input changed: {name}")
    for source in TEMPLATE.rglob("*"):
        if source.is_file():
            target = output / source.relative_to(TEMPLATE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in FILES:
        source = ROOT / name
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    report = {"bundle": str(output), "files": len(FILES) + 4,
              "source_bytes": size, "snapshot_sha256": calibration["snapshot_manifest_sha256"]}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/v24_bundle")
    build(parser.parse_args().output)
