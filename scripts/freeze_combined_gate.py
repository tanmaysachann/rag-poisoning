"""Seal the locally trained v5 gate after the confirmation protocol passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from detect.research_detector import ResearchDetector

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "detector_pilot_v4"
MANIFEST = ROOT / "research_protocols" / "combined_gate_v5.json"


def _entry(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"Frozen manifest already exists: {MANIFEST}")
    training_path = ROOT / "results" / "detector_pilot_v4_training.json"
    confirmation_path = ROOT / "results" / "detector_confirmation_v5_test.json"
    old_path = ROOT / "artifacts" / "research_detector_v3_hashing_web.joblib"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if not confirmation["combined_adoption_gate"]["pass"]:
        raise ValueError("The combined gate failed the predeclared confirmation rule")
    if confirmation.get("aggregate_attack_interval_method") != "paired bootstrap clustered by question (four layouts per question)":
        raise ValueError("Question-clustered uncertainty check is missing")
    old = joblib.load(old_path)
    if not isinstance(old, ResearchDetector):
        raise ValueError("Wrong baseline detector type")
    model_files = sorted(path for path in MODEL.iterdir() if path.is_file())
    required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"}
    if not required.issubset({path.name for path in model_files}):
        raise FileNotFoundError("Trained model export is incomplete")
    inputs = {"old_detector": _entry(old_path),
              "training_report": _entry(training_path),
              "confirmation_report": _entry(confirmation_path)}
    inputs.update({f"model_{path.name}": _entry(path) for path in model_files})
    manifest = {
        "schema_version": 1, "policy": "quarantine_if_v3_or_trained_cross_encoder_flags",
        "model_directory": MODEL.relative_to(ROOT).as_posix(),
        "old_threshold": old.threshold, "new_threshold": training["threshold"],
        "inputs": inputs,
        "confirmation_scope": "synthetic attacks on new MS MARCO questions and new SQuAD contexts",
        "runtime_requirements": "requirements-dev.txt (torch, transformers); local artifact is not bundled with the public web deployment",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "inputs": len(inputs),
                      "old_threshold": old.threshold,
                      "new_threshold": training["threshold"]}, indent=2))


if __name__ == "__main__":
    main()
