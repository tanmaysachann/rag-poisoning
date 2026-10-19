"""Freeze v19 training, calibration, and untouched evaluation question IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from data.build_full_corpus_v17 import ROOT, exclusions, normalized
from data.build_trusted_snapshot_v12 import digest


TRAIN = ROOT / "data" / "full_corpus_v19" / "NQ-open.train.jsonl"
DEV = ROOT / "data" / "full_corpus_v17" / "NQ-open.dev.jsonl"
V17 = ROOT / "research_protocols" / "full_corpus_v17_matched_split.json"
V18 = ROOT / "research_protocols" / "answer_selector_v18_split.json"
OUTPUT = ROOT / "research_protocols" / "answer_selector_v19_split.json"
TRAIN_SHA = "0c93aaf13b12f855628489665e4ef85ffb8573ad444646f11f4fabdc57a0aabf"
DEV_SHA = "f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616"


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ordered(rows: list[dict], prefix: str, key: str) -> list[dict]:
    return sorted(rows, key=lambda row: (hashlib.sha256(
        (prefix + row[key]).encode("utf-8")).hexdigest(), row[key]))


def build(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split: {output}")
    if digest(TRAIN, "sha256") != TRAIN_SHA or digest(DEV, "sha256") != DEV_SHA:
        raise ValueError("Official NQ Open input hash changed")
    v17 = json.loads(V17.read_text(encoding="utf-8"))
    v18 = json.loads(V18.read_text(encoding="utf-8"))
    used = {normalized(row["question"]) for split in (v17, v18)
            for name in ("calibration", "evaluation") for row in split[name]}
    _, external_texts, exclusion_hashes = exclusions()
    train_rows, dev_rows = records(TRAIN), records(DEV)
    train_count = Counter(normalized(row["question"]) for row in train_rows)
    dev_count = Counter(normalized(row["question"]) for row in dev_rows)
    dev_texts = set(dev_count)
    eligible_train = [{"question": row["question"], "key": normalized(row["question"])}
                      for row in train_rows if train_count[normalized(row["question"])] == 1
                      and normalized(row["question"]) not in dev_texts
                      and normalized(row["question"]) not in external_texts]
    eligible_eval = [{"question": row["question"], "key": normalized(row["question"])}
                     for row in dev_rows if dev_count[normalized(row["question"])] == 1
                     and normalized(row["question"]) not in used
                     and normalized(row["question"]) not in external_texts]
    if len(eligible_train) < 3000 or len(eligible_eval) < 500:
        raise ValueError("Insufficient disjoint questions")
    training = ordered(eligible_train, "v19-train:", "key")[:3000]
    evaluation = ordered(eligible_eval, "v19-eval:", "key")[:500]
    spent_v18 = ordered(v18["evaluation"], "v19-cal:", "qid")
    split = {
        "protocol": "research_protocols/answer_selector_v19.md",
        "nq_train_sha256": TRAIN_SHA, "nq_dev_sha256": DEV_SHA,
        "v17_split_sha256": digest(V17, "sha256"),
        "v18_split_sha256": digest(V18, "sha256"),
        "snapshot_manifest_sha256": digest(ROOT / "data" / "trusted_snapshot_v12" /
                                           "manifest.json", "sha256"),
        "exclusion_hashes": exclusion_hashes,
        "eligible_train_questions": len(eligible_train),
        "eligible_evaluation_questions": len(eligible_eval),
        "training": training,
        "spent_calibration": spent_v18[:200],
        "spent_diagnostic": spent_v18[200:],
        "calibration": [],
        "evaluation": [{"qid": f"v19-{i:04d}", "question": row["question"]}
                       for i, row in enumerate(evaluation, 1)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return {"training": len(training), "fresh_evaluation": len(evaluation),
            "eligible_evaluation": len(eligible_eval),
            "split_sha256": digest(output, "sha256")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
