"""Generate answer-blind training candidates from the pinned NQ passage index."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from data.build_answer_selector_v19 import OUTPUT, ROOT
from data.build_trusted_snapshot_v12 import digest
from evaluation.evaluate_full_corpus_v17 import QA_PATH, pin_models
from generation.trusted_reference import ReferenceReader
from retrieval.trusted_snapshot_v12 import TrustedSnapshotSearch


LOCAL = ROOT / "data" / "full_corpus_v19"
RETRIEVALS = LOCAL / "train_retrievals.jsonl"
PREDICTIONS = LOCAL / "train_predictions.jsonl"
SNAPSHOT = ROOT / "data" / "trusted_snapshot_v12"


def training_questions() -> list[dict]:
    split = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if split["snapshot_manifest_sha256"] != digest(SNAPSHOT / "manifest.json", "sha256"):
        raise ValueError("Pinned passage snapshot changed")
    return [{"qid": f"train-{i:04d}", "question": row["question"],
             "key": row["key"]} for i, row in enumerate(split["training"], 1)]


def completed_rows(path: Path, questions: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if len(rows) >= len(questions) or row["qid"] != questions[len(rows)]["qid"]:
            raise ValueError(f"Candidate cache differs from frozen split: {path}")
        rows.append(row)
    return rows


def retrieve() -> dict:
    questions = training_questions()
    prior = completed_rows(RETRIEVALS, questions)
    for case in prior:
        for passage in case["pool"]:
            if hashlib.sha256(passage["text"].encode("utf-8")).hexdigest() != passage["sha256"]:
                raise ValueError("Changed indexed source in retrieval cache")
    search = TrustedSnapshotSearch(SNAPSHOT)
    started = time.perf_counter()
    try:
        with RETRIEVALS.open("a", encoding="utf-8", newline="\n") as out:
            for index in range(len(prior), len(questions)):
                row = questions[index]
                pool = search.search(row["question"], limit=10)
                out.write(json.dumps({"qid": row["qid"], "question": row["question"],
                                      "pool": pool}, ensure_ascii=False) + "\n")
                out.flush()
                if (index + 1) % 100 == 0:
                    print(f"retrieved {index + 1}/{len(questions)} in "
                          f"{time.perf_counter() - started:.0f}s", flush=True)
    finally:
        search.close()
    return {"questions": len(questions), "retrievals_sha256": digest(RETRIEVALS, "sha256")}


def read() -> dict:
    questions = training_questions()
    retrievals = completed_rows(RETRIEVALS, questions)
    if len(retrievals) != len(questions):
        raise ValueError("Complete training retrievals before reader extraction")
    prior = completed_rows(PREDICTIONS, questions)
    pin_models()
    reader = ReferenceReader(QA_PATH)
    started = time.perf_counter()
    with PREDICTIONS.open("a", encoding="utf-8", newline="\n") as out:
        for start in range(len(prior), len(questions), 32):
            block = retrievals[start:start + 32]
            pairs = [(case["question"], passage) for case in block
                     for passage in case["pool"]]
            extracted = reader.extract_batch(
                [question for question, _ in pairs],
                [passage["text"] for _, passage in pairs], batch_size=8)
            lookup = {}
            for (question, passage), result in zip(pairs, extracted):
                source_ok = (hashlib.sha256(passage["text"].encode("utf-8")).hexdigest()
                             == passage["sha256"])
                span = passage["text"][result.start:result.end]
                verified = bool(source_ok and span and span == result.answer)
                lookup[question, passage["doc_id"]] = {
                    "doc_id": passage["doc_id"], "sha256": passage["sha256"],
                    "answer": result.answer if verified else "",
                    "qa_score": result.score if verified else 0.0,
                    "span_verified": verified,
                }
            for case in block:
                out.write(json.dumps({"qid": case["qid"], "question": case["question"],
                                      "candidates": [lookup[case["question"], p["doc_id"]]
                                                     for p in case["pool"]]},
                                     ensure_ascii=False) + "\n")
            out.flush()
            if (start // 32) % 5 == 0:
                print(f"read {min(start + 32, len(questions))}/{len(questions)} in "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
    return {"questions": len(questions), "predictions_sha256": digest(PREDICTIONS, "sha256")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("retrieve", "read"))
    args = parser.parse_args()
    print(json.dumps(retrieve() if args.phase == "retrieve" else read(), indent=2))


if __name__ == "__main__":
    main()
