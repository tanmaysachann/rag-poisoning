"""Read-only checks for the local Sentinel RAG demo and research setup."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ARTIFACTS_DIR,
    BASE_CORPUS_PATH,
    DEMO_CORPUS_PATH,
    POISONED_DOCS_PATH,
    QUERIES_PATH,
)
from security.integrity import load_runtime_manifest


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(item)
    return rows


def _check_corpus(path: Path, prefix: str, errors: list[str]) -> int:
    if not path.is_file():
        errors.append(f"Missing corpus: {path}")
        return 0

    try:
        rows = _read_jsonl(path)
        ids = [int(row["doc_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate document IDs in {path}")
        if any(not str(row["text"]).strip() for row in rows):
            errors.append(f"Empty document text in {path}")
    except (ValueError, KeyError, TypeError) as error:
        errors.append(f"Invalid corpus {path}: {error}")
        return 0

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path = ARTIFACTS_DIR / f"{prefix}_embeddings.meta.json"
    embedding_path = ARTIFACTS_DIR / f"{prefix}_embeddings.npy"
    manifest_path = ARTIFACTS_DIR / f"{prefix}_hashes.json"

    if not meta_path.is_file() or not embedding_path.is_file():
        errors.append(f"Missing embedding cache for {path.name}")
    else:
        try:
            meta = _read_json(meta_path)
            embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            if not isinstance(meta, dict):
                raise ValueError("metadata must be a JSON object")
            if meta.get("corpus_sha256") != digest:
                errors.append(f"Stale embedding cache for {path.name}")
            if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
                errors.append(f"Embedding row count differs from {path.name}")
            if list(embeddings.shape) != meta.get("shape"):
                errors.append(f"Embedding shape differs from metadata for {path.name}")
            print(f"  {path.name}: {len(rows)} docs, {meta.get('model', 'unknown')} cache")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            errors.append(f"Invalid embedding cache for {path.name}: {error}")

    if not manifest_path.is_file():
        errors.append(f"Missing integrity manifest for {path.name}")
    else:
        try:
            manifest = load_runtime_manifest(manifest_path)
            assert manifest is not None
            mismatches = [
                row["doc_id"]
                for row in rows
                if manifest.check(row["doc_id"], str(row["text"]))["status"] != "verified"
            ]
            if mismatches:
                errors.append(f"Integrity manifest mismatch for {path.name}: {mismatches[:5]}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"Invalid integrity manifest for {path.name}: {error}")
    return len(rows)


def main() -> int:
    errors: list[str] = []
    print("Sentinel RAG local readiness")
    for module in ("data.build_queries", "detect.train_fusion_classifier"):
        try:
            importlib.import_module(module)
            print(f"  import {module}: OK")
        except ImportError as error:
            errors.append(f"Cannot import {module}: {error}")

    clean_count = _check_corpus(BASE_CORPUS_PATH, "corpus", errors)
    demo_count = _check_corpus(DEMO_CORPUS_PATH, "demo_corpus", errors)

    for path in (QUERIES_PATH, POISONED_DOCS_PATH):
        if not path.is_file():
            errors.append(f"Missing data file: {path}")
            continue
        try:
            print(f"  {path.name}: {len(_read_jsonl(path))} rows")
        except ValueError as error:
            errors.append(str(error))

    if clean_count and demo_count and demo_count < clean_count:
        errors.append("Demo corpus has fewer documents than the clean corpus")

    backend = "sentence-transformers" if os.getenv("RAG_USE_MINILM", "0").strip().lower() in {
        "1", "true", "yes", "on"
    } else "hashing"
    classifier_path = ARTIFACTS_DIR / f"fusion_classifier_{backend}.joblib"
    if classifier_path.is_file():
        print(f"  requested backend: {backend}; classifier artifact: {classifier_path.name}")
    else:
        errors.append(f"Missing classifier for requested {backend} backend: {classifier_path}")

    if errors:
        print("\nIssues:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nReady. Run: python -m unittest discover -s tests -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
