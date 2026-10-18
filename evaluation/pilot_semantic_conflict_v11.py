"""Predeclared QA plus factual-refutation feasibility check on PoisonArena."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.build_multifamily_v10 import v10_train_question
from detect.research_detector import threshold_for_fpr
from evaluation.evaluate_poisonarena_v9 import build_cases
from generation.trusted_reference import ReferenceReader, answer_overlap


QA_PATH = Path("artifacts/vendor/roberta-base-squad2-nq")
QA_SHA256 = "b2f61c7b1710fcc6826fc0cc12dca8c59c0fbdadd82c05bb6ebff2028583fcf6"
NLI_PATH = Path("artifacts/factual_verifier_v6")
NLI_SHA256 = "d686901fef439dcc8a8c7342818d1c7d58910277d61672e471a188692d8a285a"
FAMILIES = ("advdecoding", "content_poison", "corpus_poison", "garag",
            "gaslite", "poisonedrag_black")


def candidate_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        qid = case["query_id"]
        rows.append({"qid": qid, "kind": "reference", "query": case["query"],
                     "context": case["reference"], "reference": case["reference"]})
        for document in case["clean"]:
            rows.append({"qid": qid, "kind": "clean", "query": case["query"],
                         "context": document["text"], "reference": case["reference"]})
        for family in FAMILIES:
            for context in case["attacks"].get(family, []):
                rows.append({"qid": qid, "kind": family, "query": case["query"],
                             "context": context, "reference": case["reference"]})
    return rows


@torch.inference_mode()
def refutation_probabilities(rows: list[dict], device: torch.device,
                             batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(NLI_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_PATH, local_files_only=True).to(device).eval()
    if model.config.label2id.get("REFUTES") != 1:
        raise ValueError("Unexpected v6 fact-verifier labels")
    output = np.zeros(len(rows), dtype=np.float64)
    eligible = [i for i, row in enumerate(rows) if row["candidate_span"]]
    for start in range(0, len(eligible), batch_size):
        indexes = eligible[start:start + batch_size]
        claims = [f"The answer to {rows[i]['query'].rstrip('?')} is "
                  f"{rows[i]['candidate_span']}." for i in indexes]
        evidence = [rows[i]["reference"] for i in indexes]
        tokens = tokenizer(claims, evidence, padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(device)
        probabilities = torch.softmax(model(**tokens).logits.float(), dim=1)
        output[indexes] = probabilities[:, 1].cpu().numpy()
    return output


def summarize(rows: list[dict], threshold: float, field: str) -> dict:
    result = {}
    for family in ("clean", "reference", *FAMILIES):
        subset = [row for row in rows if row["kind"] == family]
        flags = sum(row[field] >= threshold for row in subset)
        result[family] = {"flagged": flags, "total": len(subset),
                          "rate": flags / len(subset) if subset else None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/semantic_conflict_v11_pilot.json"))
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if hashlib.sha256((QA_PATH / "pytorch_model.bin").read_bytes()).hexdigest() != QA_SHA256:
        raise ValueError("Pinned QA weights changed")
    if hashlib.sha256((NLI_PATH / "model.safetensors").read_bytes()).hexdigest() != NLI_SHA256:
        raise ValueError("Pinned fact-verifier weights changed")
    cases, selection = build_cases(Path("data/poisonarena_v9"),
                                   Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip"),
                                   args.beir_archive)
    train_ids = {case["query_id"] for case in cases if v10_train_question(case["query"])}
    development_ids = {case["query_id"] for case in cases} - train_ids
    rows = candidate_rows(cases)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    reader = ReferenceReader(QA_PATH)
    predictions = reader.extract_batch([row["query"] for row in rows],
                                       [row["context"] for row in rows],
                                       batch_size=args.batch_size)
    refs = {row["qid"]: prediction for row, prediction in zip(rows, predictions)
            if row["kind"] == "reference"}
    for row, prediction in zip(rows, predictions):
        reference = refs[row["qid"]]
        row["candidate_span"] = prediction.answer.strip()
        row["candidate_qa_score"] = prediction.score
        row["reference_span"] = reference.answer.strip()
        row["reference_qa_score"] = reference.score
        conflict = bool(row["candidate_span"] and row["reference_span"] and
                        not answer_overlap(row["candidate_span"], row["reference_span"]))
        row["answer_difference"] = (min(prediction.score, reference.score)
                                    if conflict else 0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refute = refutation_probabilities(rows, device, args.batch_size)
    for row, probability in zip(rows, refute):
        row["factual_refutation"] = (float(probability) * row["answer_difference"]
                                     if row["answer_difference"] else 0.0)
    calibration = [row for row in rows if row["qid"] in train_ids]
    development = [row for row in rows if row["qid"] in development_ids]
    signals = {}
    for field in ("answer_difference", "factual_refutation"):
        clean = np.asarray([row[field] for row in calibration if row["kind"] == "clean"],
                           dtype=np.float64)
        threshold = float(threshold_for_fpr(clean, 0.01))
        signals[field] = {"threshold": threshold,
                          "calibration": summarize(calibration, threshold, field),
                          "development": summarize(development, threshold, field)}
    arena = json.loads(Path("data/poisonarena_v9/nq_arena.json").read_text(encoding="utf-8"))
    targets = {}
    for family in FAMILIES:
        source = json.loads((Path("data/poisonarena_v9") / f"{family}.json")
                            .read_text(encoding="utf-8"))
        targets[family] = {row["id"]: row["incorrect_answer"] for row in source}
    quality = defaultdict(lambda: {"correct_span_match": 0, "target_span_match": 0,
                                   "total": 0})
    for row in development:
        kind = row["kind"]
        correct = arena[row["qid"]]["correct answer"]
        quality[kind]["correct_span_match"] += answer_overlap(row["candidate_span"], correct)
        if kind in targets and isinstance(targets[kind][row["qid"]], str):
            quality[kind]["target_span_match"] += answer_overlap(
                row["candidate_span"], targets[kind][row["qid"]])
        quality[kind]["total"] += 1
    earlier = json.loads(Path("results/multifamily_v10_poisonarena_heldout.json")
                         .read_text(encoding="utf-8"))
    report = {"protocol": "research_protocols/semantic_conflict_v11.md",
              "qa_weights_sha256": QA_SHA256, "nli_weights_sha256": NLI_SHA256,
              "poisonarena_manifest_sha256": hashlib.sha256(
                  Path("data/poisonarena_v9/manifest.json").read_bytes()).hexdigest(),
              "selection": selection, "calibration_questions": len(train_ids),
              "development_questions": len(development_ids), "signals": signals,
              "development_span_audit": dict(quality),
              "frozen_v10_same_questions": earlier["candidate_model"],
              "frozen_v3_same_questions": earlier["frozen_v3"],
              "gate": "At least 50% PoisonedRAG-black and <=1% clean flags on development; otherwise stop.",
              "limitation": "Previously inspected PoisonArena development split; qrel is an oracle-like reference."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
