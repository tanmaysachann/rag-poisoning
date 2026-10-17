"""Secondary answer impact of the frozen v3 detector on external RAG cases.

This replays archived top-10 retrieval through the project's local extractive
answerer. It does not rerun or reproduce the archive's original LLM answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import joblib

from detect.research_detector import ResearchDetector
from evaluation.evaluate_external_rag_v7 import candidates, read_cases
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import TextEmbedder


class FixedRetriever:
    def __init__(self, documents: list[dict], embedder: TextEmbedder):
        self.documents = documents
        self.embedder = embedder

    def encode(self, texts):
        return self.embedder.encode(texts)


def contains_answer(text: str, answer: str) -> bool:
    normalized = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
    expected = normalized(answer)
    return bool(expected) and expected in normalized(text)


def answer_for(case: dict, family: str, scores: dict, threshold: float,
               embedder: TextEmbedder, defended: bool) -> str:
    documents = []
    for candidate in case["retrievals"][family]["candidates"]:
        if defended and scores[case["query_id"], candidate["id"]] >= threshold:
            continue
        documents.append({"doc_id": len(documents), "text": candidate["text"]})
    retriever = FixedRetriever(documents, embedder)
    return _select_answer(case["query"], documents, retriever)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/external_rag_v7"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/external_rag_v7_answer_v3.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    cases = read_cases(args.data / "test.jsonl", manifest["file_sha256"]["test.jsonl"])
    old = joblib.load("artifacts/research_detector_v3_hashing.joblib")
    if not isinstance(old, ResearchDetector):
        raise TypeError("Unexpected v3 detector artifact")
    rows = candidates(cases)
    started = time.perf_counter()
    scores = {key: old.score(row["query"], row["text"])["risk_score"]
              for key, row in rows.items()}
    embedder = TextEmbedder(preferred_backend="hashing")
    report = {"scope": "secondary extractive answer replay, frozen v3 content detector",
              "source_manifest_sha256": hashlib.sha256(
                  (args.data / "manifest.json").read_bytes()
              ).hexdigest(), "threshold": old.threshold, "questions": len(cases),
              "answer_match": "normalized exact substring; aliases and paraphrases can be missed",
              "families": {family: {variant: {"correct_match": 0, "target_match": 0,
                                               "no_evidence": 0, "questions": 0}
                                    for variant in ("before", "after")}
                           for family in ("clean", "naive", "corruptrag_ak")}}
    for index, case in enumerate(cases, 1):
        for family in report["families"]:
            for variant in ("before", "after"):
                answer = answer_for(case, family, scores, old.threshold, embedder,
                                    defended=variant == "after")
                counts = report["families"][family][variant]
                counts["questions"] += 1
                counts["correct_match"] += contains_answer(answer, case["correct_answer"])
                if family != "clean":
                    counts["target_match"] += contains_answer(answer, case["target_answer"])
                counts["no_evidence"] += answer.startswith("Insufficient relevant evidence")
        if index % 100 == 0:
            print(f"answered {index}/{len(cases)} questions", flush=True)
    for family in report["families"].values():
        for counts in family.values():
            counts["correct_match_rate"] = counts["correct_match"] / counts["questions"]
            counts["target_match_rate"] = counts["target_match"] / counts["questions"]
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
