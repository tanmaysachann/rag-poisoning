"""Add disjoint PoisonArena training questions to the v9 source-labelled rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from data.build_reference_detector_v9 import digest
from evaluation.evaluate_poisonarena_v9 import build_cases, normalized_question


TRAIN_FAMILIES = ("advdecoding", "corpus_poison", "garag", "content_poison")


def v10_train_question(question: str) -> bool:
    normalized = normalized_question(question)
    return int(hashlib.sha256(("v10-pilot:" + normalized).encode()).hexdigest()[:8], 16) % 10 < 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/multifamily_v10"))
    args = parser.parse_args()
    cases, selection = build_cases(Path("data/poisonarena_v9"),
                                   Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip"),
                                   args.beir_archive)
    old_manifest = json.loads(Path("data/reference_detector_v9/manifest.json").read_text())
    old_path = Path("data/reference_detector_v9/train.jsonl")
    if digest(old_path, "sha256") != old_manifest["train_sha256"]:
        raise ValueError("V9 training data changed")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "train.jsonl"
    counts = Counter()
    train_questions = set()
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for line in old_path.open(encoding="utf-8"):
            sink.write(line)
            counts[json.loads(line)["label"]] += 1
        for case in cases:
            if not v10_train_question(case["query"]):
                continue
            qid = "poisonarena:" + case["query_id"]
            train_questions.add(qid)
            rows = [(case["reference_id"], case["reference"], "reference")]
            rows += [(doc["id"], doc["text"], "clean_other")
                     for doc in case["clean"] if doc["id"] != case["reference_id"]]
            for family in TRAIN_FAMILIES:
                for index, candidate in enumerate(case["attacks"].get(family, [])):
                    rows.append((f"{family}:{index}", candidate, "poison"))
            for candidate_id, candidate, label in rows:
                sink.write(json.dumps({"query_id": qid, "query": case["query"],
                                       "reference_id": case["reference_id"],
                                       "reference": case["reference"],
                                       "candidate_id": candidate_id,
                                       "candidate": candidate,
                                       "label": label}, ensure_ascii=False) + "\n")
                counts[label] += 1
    manifest = {"protocol": "research_protocols/multifamily_detector_v10.md",
                "v9_train_sha256": old_manifest["train_sha256"],
                "poisonarena_manifest_sha256": digest(
                    Path("data/poisonarena_v9/manifest.json"), "sha256"),
                "training_families": TRAIN_FAMILIES,
                "poisonarena_training_questions": len(train_questions),
                "poisonarena_training_qids_sha256": hashlib.sha256(
                    "\n".join(sorted(train_questions)).encode()).hexdigest(),
                "label_counts": dict(counts), "train_sha256": digest(output, "sha256"),
                "source_selection": selection}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
