"""Rebuild the offline demo in an isolated temporary project copy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPY_DIRS = ("attack", "backend", "data", "detect", "evaluation", "frontend", "generation", "pipeline", "poison", "retrieval", "scripts", "security", "tests")


def _ignore_generated(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def _run(project: Path, *args: str, env: dict[str, str]) -> None:
    print(f"\n> python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=project, env=env, check=True)


def main() -> int:
    temp_parent = ROOT / "tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sentinel-rebuild-", dir=temp_parent) as temporary:
        project = Path(temporary)
        for name in ("config.py", "requirements.txt", "requirements-dev.txt"):
            shutil.copy2(ROOT / name, project / name)
        for name in COPY_DIRS:
            shutil.copytree(ROOT / name, project / name, ignore=_ignore_generated)

        env = os.environ.copy()
        env["RAG_USE_MINILM"] = "0"
        env["RAG_USE_LLM"] = "0"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        _run(project, "scripts/build_index.py", "--reuse-data", env=env)
        _run(project, "scripts/inject_poison.py", env=env)
        _run(project, "-m", "detect.train_fusion_classifier", env=env)
        _run(project, "scripts/build_pdf_reports.py", env=env)
        _run(project, "scripts/doctor.py", env=env)
        _run(project, "-m", "unittest", "discover", "-s", "tests", "-v", env=env)

        metrics_path = project / "results" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        report_count = len(list((project / "output" / "pdf").glob("*.pdf")))
        if metrics["clean_samples"] != 25 or metrics["poisoned_samples"] != 5:
            raise RuntimeError(f"Unexpected offline sample counts: {metrics}")
        if report_count < 5:
            raise RuntimeError(f"Expected at least five PDF reports, found {report_count}")
        print(
            f"\nOffline rebuild passed: {metrics['clean_samples']} clean, "
            f"{metrics['poisoned_samples']} poisoned, {report_count} PDFs."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
