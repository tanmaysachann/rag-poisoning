"""Build a question-disjoint six-family PoisonArena fine-tuning dataset.

The V9 MiniLM checkpoint supplies the NQ/Architecture Matters pretraining.
Only the published PoisonArena development archive is used here. All results
from this collection are development evidence, not a fresh blind test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from data.build_reference_detector_v9 import BEIR_MD5, digest
from evaluation.evaluate_poisonarena_v9 import build_cases, normalized_question

FAMILIES = ("advdecoding", "content_poison", "corpus_poison", "garag",
            "gaslite", "poisonedrag_black")


def split_for_question(question: str) -> str:
    """Keep V10's 29 evaluation questions; split the other 71 for calibration."""
    key = normalized_question(question)
    old_bucket = int(hashlib.sha256(("v10-pilot:" + key).encode()).hexdigest()[:8], 16) % 10
    if old_bucket >= 7:
        return "evaluation"
    calibration_bucket = int(hashlib.sha256(("v20-cal:" + key).encode()).hexdigest()[:8], 16) % 5
    return "calibration" if calibration_bucket == 0 else "train"


def rows_for_case(case: dict, split: str) -> list[dict]:
    base = {"query_id": case["query_id"], "query": case["query"],
            "reference_id": case["reference_id"], "reference": case["reference"],
            "split": split}
    rows = [{**base, "candidate_id": case["reference_id"],
             "candidate": case["reference"], "label": "reference", "family": "reference"}]
    rows.extend({**base, "candidate_id": doc["id"], "candidate": doc["text"],
                 "label": "clean_other", "family": "clean"}
                for doc in case["clean"] if doc["id"] != case["reference_id"])
    for family in FAMILIES:
        rows.extend({**base, "candidate_id": f"{family}:{index}",
                     "candidate": candidate, "label": "poison", "family": family}
                    for index, candidate in enumerate(case["attacks"].get(family, [])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/multifamily_v20"))
    args = parser.parse_args()
    if digest(args.beir_archive, "md5") != BEIR_MD5:
        raise ValueError("BEIR archive changed")
    cases, source_counts = build_cases(
        Path("data/poisonarena_v9"),
        Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip"),
        args.beir_archive,
    )
    by_split = {split: [] for split in ("train", "calibration", "evaluation")}
    qids = {split: set() for split in by_split}
    for case in cases:
        split = split_for_question(case["query"])
        by_split[split].extend(rows_for_case(case, split))
        qids[split].add(case["query_id"])
    if len(set.union(*qids.values())) != len(cases) or any(
        qids[a] & qids[b] for a in qids for b in qids if a != b
    ):
        raise AssertionError("Question leakage between splits")
    if min(map(len, qids.values())) < 10:
        raise ValueError("Question split is too small")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset: {args.output}")
    manifest = {
        "schema": "multifamily-v20-source-labels",
        "source": "Pinned PoisonArena six-family files with BEIR NQ clean retrievals/qrels",
        "poisonarena_manifest_sha256": digest(Path("data/poisonarena_v9/manifest.json"), "sha256"),
        "beir_md5": BEIR_MD5,
        "parent_v9_training_sha256": json.loads(
            Path("data/reference_detector_v9/manifest.json").read_text(encoding="utf-8")
        )["train_sha256"],
        "families": FAMILIES,
        "splits": {},
        "source_counts": source_counts,
        "limitations": [
            "These are source labels, not independently reviewed factual-contradiction labels.",
            "All PoisonArena questions have previously been inspected in this project; evaluation is development-only.",
            "BEIR qrels are benchmark-selected references, not authenticated live sources.",
        ],
    }
    for split, rows in by_split.items():
        path = args.output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "questions": len(qids[split]), "rows": len(rows),
            "label_counts": dict(Counter(row["label"] for row in rows)),
            "family_counts": dict(Counter(row["family"] for row in rows)),
            "query_ids_sha256": hashlib.sha256(
                "\n".join(sorted(qids[split])).encode()
            ).hexdigest(),
            "sha256": digest(path, "sha256"),
        }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
