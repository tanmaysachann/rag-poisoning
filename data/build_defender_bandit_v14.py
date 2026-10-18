"""Freeze v3 feature rows for the predeclared v14 contextual-bandit pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import joblib

from data.build_multifamily_v10 import TRAIN_FAMILIES, v10_train_question
from data.build_reference_detector_v9 import ATTACK_SHA256, BEIR_MD5, digest as source_digest
from detect.research_detector import ResearchDetector
from evaluation.evaluate_poisonarena_v9 import build_cases, normalized_question

HOT_SHA = "5119d6f9fd53cb0ecb3f33de939bd940d77cbaf80284b7ccd3d256f63a430537"
HELDOUT = ("gaslite", "poisonedrag_black")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/defender_bandit_v14"))
    args = parser.parse_args()
    output = args.output / "features.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen feature rows: {output}")
    attack_archive = Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip")
    if source_digest(attack_archive, "sha256") != ATTACK_SHA256:
        raise ValueError("Architecture Matters source archive changed")
    if source_digest(args.beir_archive, "md5") != BEIR_MD5:
        raise ValueError("BEIR NQ source archive changed")
    cases, selection = build_cases(
        Path("data/poisonarena_v9"),
        attack_archive,
        args.beir_archive,
    )
    train_pool = sorted(
        (case for case in cases if v10_train_question(case["query"])),
        key=lambda case: hashlib.sha256(
            ("v14-cal:" + normalized_question(case["query"])).encode()
        ).hexdigest(),
    )
    if len(cases) != 100 or len(train_pool) != 71:
        raise ValueError(f"Unexpected question counts: {len(cases)}, {len(train_pool)}")
    training = {case["query_id"] for case in train_pool[:50]}
    calibration = {case["query_id"] for case in train_pool[50:]}
    development = {case["query_id"] for case in cases if not v10_train_question(case["query"])}
    if (len(training), len(calibration), len(development)) != (50, 21, 29):
        raise ValueError("Unexpected split counts")
    model_path = Path("artifacts/research_detector_v3_hashing.joblib")
    model = joblib.load(model_path)
    if not isinstance(model, ResearchDetector):
        raise TypeError("Unexpected v3 detector")
    hot_path = Path("artifacts/vendor/poisonedrag-hotpotqa-v14.json")
    if digest(hot_path) != HOT_SHA:
        raise ValueError("HotpotQA attack file changed")
    hot = json.loads(hot_path.read_text(encoding="utf-8"))
    if len(hot) != 100:
        raise ValueError("Unexpected HotpotQA question count")
    arena_questions = {normalized_question(case["query"]) for case in cases}
    overlap = sum(normalized_question(row["question"]) in arena_questions for row in hot.values())
    if overlap:
        raise ValueError(f"External question overlap: {overlap}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    started = time.perf_counter()

    def score_row(sink, *, split: str, qid: str, kind: str, index: int, query: str,
                  passage: str, label: int) -> None:
        features = model.features(query, passage)
        if not all(math.isfinite(float(value)) for value in features):
            raise ValueError(f"Nonfinite features: {split}/{qid}/{kind}/{index}")
        risk = float(model.classifier.predict_proba(features.reshape(1, -1))[0, 1])
        if not math.isfinite(risk):
            raise ValueError("Nonfinite v3 risk")
        sink.write(json.dumps({"split": split, "qid": qid, "kind": kind,
                               "index": index, "label": label,
                               "features": [*map(float, features), risk]},
                              separators=(",", ":")) + "\n")
        counts[f"{split}:{kind}"] += 1

    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for case in sorted(cases, key=lambda item: item["query_id"]):
            qid = case["query_id"]
            split = ("train" if qid in training else "calibration" if qid in calibration
                     else "development")
            for index, doc in enumerate(case["clean"]):
                score_row(sink, split=split, qid=qid, kind="clean", index=index,
                          query=case["query"], passage=doc["text"], label=0)
            score_row(sink, split=split, qid=qid, kind="reference", index=0,
                      query=case["query"], passage=case["reference"], label=0)
            families = TRAIN_FAMILIES if split != "development" else (*TRAIN_FAMILIES, *HELDOUT)
            for family in families:
                texts = case["attacks"].get(family)
                if texts is None:
                    continue  # Content Poisoning was released for only 20/100 questions.
                if len(texts) != 5:
                    raise ValueError(f"Unexpected attack count for {qid}/{family}")
                for index, passage in enumerate(texts):
                    score_row(sink, split=split, qid=qid, kind=family, index=index,
                              query=case["query"], passage=passage, label=1)
            print(f"features: {sum(counts.values())} rows / {qid}", flush=True)
        for qid, row in sorted(hot.items()):
            for index, passage in enumerate(row["adv_texts"]):
                score_row(sink, split="external", qid=qid, kind="hotpot_poisonedrag",
                          index=index, query=row["question"], passage=passage, label=1)
    manifest = {
        "protocol": "research_protocols/defender_bandit_v14.md",
        "feature_sha256": digest(output), "v3_model_sha256": digest(model_path),
        "poisonarena_manifest_sha256": digest(Path("data/poisonarena_v9/manifest.json")),
        "hotpot_attack_sha256": HOT_SHA,
        "question_counts": {"train": 50, "calibration": 21, "development": 29,
                            "external": 100},
        "question_ids_sha256": {name: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
                                for name, ids in (("train", training), ("calibration", calibration),
                                                  ("development", development))},
        "external_question_overlap": overlap, "row_counts": dict(sorted(counts.items())),
        "selection": selection, "feature_extraction_seconds": round(time.perf_counter()-started, 2),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
