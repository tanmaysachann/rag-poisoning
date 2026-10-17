"""Freeze page-disjoint, real-revision VitaminC claim/evidence pairs.

This is factual verification data, not labeled RAG poisoning data. The official
archive is read in place so its much larger JSONL files need not be extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

SOURCE_URL = "https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip"
SOURCE_SHA256 = "49d82dc1690cbee420d18e2c26f687a7937710bb211845d2571430dfd4dc0337"
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
QUOTAS = {
    "train": {"SUPPORTS": 15000, "REFUTES": 10000, "NOT ENOUGH INFO": 5000},
    "validation": {"SUPPORTS": 2000, "REFUTES": 1500, "NOT ENOUGH INFO": 2000},
    "calibration": {"SUPPORTS": 5000, "REFUTES": 1000, "NOT ENOUGH INFO": 1000},
    "test": {"SUPPORTS": 5000, "REFUTES": 1000, "NOT ENOUGH INFO": 1000},
}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _partition(source_split: str, page: str) -> str:
    if source_split == "train":
        return "train"
    if source_split == "test":
        return "test"
    return "calibration" if int(_hash("v6-dev-page:" + page)[:8], 16) % 10 < 7 else "validation"


def _select(rows: list[dict], quotas: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    used_cases: set[str] = set()
    used_pairs: set[str] = set()
    for row in rows:
        label = row["label"]
        pair = _hash(row["claim"] + "\0" + row["evidence"])
        if counts[label] >= quotas[label] or row["case_id"] in used_cases or pair in used_pairs:
            continue
        used_cases.add(row["case_id"])
        used_pairs.add(pair)
        selected.append(row)
        counts[label] += 1
        if all(counts[key] == number for key, number in quotas.items()):
            break
    if any(counts[key] != number for key, number in quotas.items()):
        raise ValueError(f"Insufficient distinct source cases: expected {quotas}, got {dict(counts)}")
    selected.sort(key=lambda row: (row["page"], row["case_id"], row["unique_id"]))
    return selected


def build(archive: Path, output: Path, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen data: {output}")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("VitaminC archive SHA256 differs from the pinned source")
    pools: dict[str, list[dict]] = {name: [] for name in QUOTAS}
    with zipfile.ZipFile(archive) as source:
        for source_split in ("train", "dev", "test"):
            with source.open(f"vitaminc/{source_split}.jsonl") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("revision_type") != "real" or row.get("label") not in LABELS:
                        continue
                    if not all(isinstance(row.get(key), str) and row[key].strip()
                               for key in ("unique_id", "case_id", "page", "claim", "evidence")):
                        continue
                    if len(row["claim"]) > 400 or len(row["evidence"]) > 1800:
                        continue
                    name = _partition(source_split, row["page"])
                    pools[name].append({key: row[key] for key in
                                        ("unique_id", "case_id", "page", "claim", "evidence", "label")})
    selected = {
        name: _select(pools[name], quotas, seed + index)
        for index, (name, quotas) in enumerate(QUOTAS.items())
    }
    pages = {name: {row["page"] for row in rows} for name, rows in selected.items()}
    cases = {name: {row["case_id"] for row in rows} for name, rows in selected.items()}
    pairs = {name: {_hash(row["claim"] + "\0" + row["evidence"]) for row in rows}
             for name, rows in selected.items()}
    for left in QUOTAS:
        for right in QUOTAS:
            if left < right and (pages[left] & pages[right] or cases[left] & cases[right]
                                 or pairs[left] & pairs[right]):
                raise ValueError(f"Source or text leakage between {left} and {right}")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in selected.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
        "source_subset": "real Wikipedia revisions only", "seed": seed,
        "counts": {name: dict(Counter(row["label"] for row in rows))
                   for name, rows in selected.items()},
        "distinct_pages": {name: len(values) for name, values in pages.items()},
        "split_rule": "official train/test pages; dev pages hashed to validation/calibration; one case per split",
        "file_sha256": files,
        "task": "claim/evidence factual verification, not RAG poison classification",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("artifacts/vendor/vitaminc.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/factual_verifier_v6"))
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
