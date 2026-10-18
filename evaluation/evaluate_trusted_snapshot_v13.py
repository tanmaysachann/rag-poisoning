"""Pinned Contriever reranking of qrel-free BEIR NQ snapshot candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from evaluation.evaluate_poisonarena_v9 import verified_json
from evaluation.evaluate_trusted_snapshot_v12 import (
    QA_PATH, QA_SHA256, POLICIES, calibrate, load_questions, outcomes,
    policy_prediction,
)
from generation.trusted_reference import ReferenceReader, answer_overlap
from retrieval.trusted_snapshot_v12 import TrustedSnapshotSearch


CONTRIEVER_PATH = Path("artifacts/models/contriever_msmarco")
CONTRIEVER_SHA256 = "08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8"
RANKINGS = ("dense", "fusion")


@torch.inference_mode()
def embed(texts: list[str], tokenizer, model, device: torch.device,
          batch_size: int = 32) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[start:start + batch_size], padding=True,
                           truncation=True, max_length=256,
                           return_tensors="pt").to(device)
        output = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        chunks.append(torch.nn.functional.normalize(pooled, dim=1)
                      .float().cpu().numpy())
        if start % 2048 == 0:
            print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(chunks)


def rankings(case: dict, vectors: dict[str, np.ndarray],
             query_vector: np.ndarray) -> dict[str, list[dict]]:
    pool = case["pool"]
    scores = np.asarray([float(vectors[row["sha256"]] @ query_vector)
                         for row in pool], dtype=np.float64)
    dense_order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    dense_ranks = np.empty(len(pool), dtype=np.int32)
    for rank, index in enumerate(dense_order, 1):
        dense_ranks[index] = rank
    fusion = np.asarray([1 / (60 + index + 1) + 1 / (60 + dense_ranks[index])
                         for index in range(len(pool))])
    fusion_order = sorted(range(len(pool)), key=lambda i: (-fusion[i], i))
    return {"dense": [pool[index] for index in dense_order[:10]],
            "fusion": [pool[index] for index in fusion_order[:10]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/trusted_snapshot_v12"))
    parser.add_argument("--poisonarena", type=Path, default=Path("data/poisonarena_v9"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trusted_snapshot_v13.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite already scored v13 evaluation")
    if hashlib.sha256((CONTRIEVER_PATH / "pytorch_model.bin").read_bytes()).hexdigest() != CONTRIEVER_SHA256:
        raise ValueError("Pinned Contriever weights changed")
    if hashlib.sha256((QA_PATH / "pytorch_model.bin").read_bytes()).hexdigest() != QA_SHA256:
        raise ValueError("Pinned QA weights changed")
    questions, source_manifest = load_questions(args.poisonarena)
    search = TrustedSnapshotSearch(args.snapshot)
    try:
        cases = [{**row, "pool": search.search(row["question"], limit=100)}
                 for row in questions]
    finally:
        search.close()
    print("Retrieved 100 snapshot passages per question", flush=True)
    unique = {passage["sha256"]: passage["text"]
              for case in cases for passage in case["pool"]}
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(CONTRIEVER_PATH,
                                      local_files_only=True).to(device).eval()
    keys = list(unique)
    vectors = embed([unique[key] for key in keys], tokenizer, model, device)
    vector_map = dict(zip(keys, vectors))
    questions_vector = embed([case["question"] for case in cases],
                             tokenizer, model, device)
    for case, query_vector in zip(cases, questions_vector):
        case["rankings"] = rankings(case, vector_map, query_vector)
        case.pop("pool")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Reranked candidates without qrels or answer keys", flush=True)
    reader = ReferenceReader(QA_PATH)
    pairs = {}
    for case in cases:
        for ranking in RANKINGS:
            for passage in case["rankings"][ranking]:
                pairs[case["qid"], passage["doc_id"]] = (case["question"], passage)
    flat = list(pairs.values())
    extracted = reader.extract_batch([query for query, _ in flat],
                                     [passage["text"] for _, passage in flat],
                                     batch_size=24)
    for (_, passage), result in zip(flat, extracted):
        passage["answer"] = result.answer.strip()
        passage["qa_score"] = result.score
    prediction_path = args.snapshot / "v13_predictions.jsonl"
    if prediction_path.exists():
        raise FileExistsError("Refusing to overwrite prior v13 predictions")
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps({**case, "rankings": {
                name: [{key: value for key, value in passage.items() if key != "text"}
                       for passage in case["rankings"][name]]
                for name in RANKINGS}}, ensure_ascii=False) + "\n")
    print("Saved v13 predictions before opening answers or qrels", flush=True)
    arena = verified_json(args.poisonarena / "nq_arena.json",
                          source_manifest["files"]["nq_arena"]["sha256"])
    answers = {qid: row["correct answer"] for qid, row in arena.items()}
    calibration = [case for case in cases if case["split"] == "calibration"]
    development = [case for case in cases if case["split"] == "development"]
    options = []
    for ranking in RANKINGS:
        for policy in POLICIES:
            predictions = {case["qid"]: policy_prediction(
                case["rankings"][ranking], policy) for case in cases}
            cutoff, counts = calibrate(calibration, predictions, answers)
            options.append((counts["correct_match"], -counts["wrong_emitted"],
                            -RANKINGS.index(ranking), -POLICIES.index(policy),
                            ranking, policy, cutoff, counts))
    _, _, _, _, selected_ranking, selected_policy, cutoff, calibration_counts = max(options)
    predictions = {case["qid"]: policy_prediction(
        case["rankings"][selected_ranking], selected_policy) for case in cases}
    development_counts = outcomes(development, predictions, cutoff, answers)
    qrels_data = (args.poisonarena / "nq_qrels.json").read_bytes()
    if hashlib.sha256(qrels_data).hexdigest() != source_manifest["files"]["nq_qrels"]["sha256"]:
        raise ValueError("PoisonArena qrels changed")
    relevant = {case["qid"]: set() for case in cases}
    with io.StringIO(qrels_data.decode("utf-8")) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["query-id"] in relevant and int(row["score"]) > 0:
                relevant[row["query-id"]].add(row["corpus-id"])
    retrieval = {}
    for name, subset in (("calibration", calibration), ("development", development)):
        retrieval[name] = {}
        for ranking in RANKINGS:
            retrieval[name][ranking] = {f"qrel_at_{k}": sum(bool(
                relevant[case["qid"]] & {passage["doc_id"]
                for passage in case["rankings"][ranking][:k]})
                for case in subset) for k in (1, 5, 10)}
    result_cases = []
    for case in cases:
        prediction = predictions[case["qid"]]
        emitted = prediction["answer"] if prediction["score"] >= cutoff else ""
        result_cases.append({"qid": case["qid"], "split": case["split"],
                             "question": case["question"], "answer": emitted,
                             "correct_answer": answers[case["qid"]],
                             "correct_strict_match": answer_overlap(emitted, answers[case["qid"]]),
                             "selected_doc_ids": prediction["doc_ids"],
                             "selected_score": prediction["score"],
                             "retrieved_doc_ids": [row["doc_id"] for row in
                                                    case["rankings"][selected_ranking]]})
    with args.output.with_suffix(".cases.jsonl").open(
            "w", encoding="utf-8", newline="\n") as handle:
        for row in result_cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    earlier = json.loads(Path("results/trusted_snapshot_v12.json").read_text(encoding="utf-8"))
    report = {"protocol": "research_protocols/trusted_snapshot_v13.md",
              "snapshot_manifest_sha256": hashlib.sha256(
                  (args.snapshot / "manifest.json").read_bytes()).hexdigest(),
              "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              "contriever_weights_sha256": CONTRIEVER_SHA256,
              "qa_weights_sha256": QA_SHA256,
              "poisonarena_manifest_sha256": hashlib.sha256(
                  (args.poisonarena / "manifest.json").read_bytes()).hexdigest(),
              "calibration_questions": len(calibration),
              "development_questions": len(development),
              "calibration_comparison": [{"ranking": row[4], "policy": row[5],
                                          "cutoff": row[6], **row[7]} for row in options],
              "selected_ranking": selected_ranking, "selected_policy": selected_policy,
              "frozen_cutoff": cutoff, "calibration": calibration_counts,
              "development": development_counts, "retrieval": retrieval,
              "prior_v12_development": earlier["development"],
              "prior_v12_development_retrieval": earlier["retrieval"]["development"],
              "gate": {"development_correct_at_least": 9,
                       "development_wrong_at_most": 2,
                       "development_qrel_at_10_at_least": 9},
              "limitation": "Same already inspected development questions; unreviewed benchmark corpus, no live source registry."}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
