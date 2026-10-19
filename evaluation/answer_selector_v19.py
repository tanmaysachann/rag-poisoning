"""Train a text answer selector; freeze predictions before fresh scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.build_answer_selector_v19 import DEV, OUTPUT, ROOT, TRAIN, V17, records
from data.generate_answer_selector_v19 import training_questions
from data.build_full_corpus_v17 import normalized
from data.build_trusted_snapshot_v12 import digest
from evaluation.answer_selector_v18 import calibrate, read_predictions
from evaluation.evaluate_full_corpus_v17 import answer_match, outcomes, split_rows
from evaluation.evaluate_trusted_snapshot_v12 import policy_prediction


LOCAL = ROOT / "data" / "full_corpus_v19"
MODEL = ROOT / "artifacts" / "answer_selector_v19"
EXAMPLES = LOCAL / "training_examples.jsonl"
TRAIN_PREDICTIONS = LOCAL / "train_predictions.jsonl"
TRAIN_RETRIEVALS = LOCAL / "train_retrievals.jsonl"
V17_PREDICTIONS = ROOT / "data" / "full_corpus_v17" / "predictions.jsonl"
V17_RETRIEVALS = ROOT / "data" / "full_corpus_v17" / "retrievals.jsonl"
V18_PREDICTIONS = ROOT / "data" / "full_corpus_v18" / "predictions.jsonl"
V18_RETRIEVALS = ROOT / "data" / "full_corpus_v18" / "retrievals.jsonl"
EVAL_PREDICTIONS = LOCAL / "evaluation" / "predictions.jsonl"
EVAL_RETRIEVALS = LOCAL / "evaluation" / "retrievals.jsonl"
SELECTED = LOCAL / "selected_predictions.jsonl"
TRAIN_REPORT = ROOT / "results" / "answer_selector_v19_training.json"
EVAL_REPORT = ROOT / "results" / "answer_selector_v19.json"
CASES = ROOT / "results" / "answer_selector_v19.cases.jsonl"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BASE_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
SEED = 19
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 2


def source_map(path: Path) -> dict[str, dict[str, dict]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if case["qid"] in result:
            raise ValueError(f"Duplicate source question in {path}")
        result[case["qid"]] = {str(row["doc_id"]): row for row in case["pool"]}
    return result


def verified_options(case: dict, sources: dict[str, dict]) -> list[dict]:
    rows = (case["candidates"] if "candidates" in case else
            [row for method in ("lexical", "dense", "fusion")
             for row in case["rankings"][method]])
    found = {}
    for row in rows:
        if not row["span_verified"] or not row["answer"].strip():
            continue
        doc_id = str(row["doc_id"])
        source = sources.get(doc_id)
        if source is None or source["sha256"] != row["sha256"]:
            raise ValueError("Reader source differs from pinned retrieved passage")
        body = source["text"]
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != row["sha256"]:
            raise ValueError("Retrieved passage hash changed")
        if row["answer"] not in body:
            raise ValueError("Verified answer span absent from passage")
        found[doc_id] = {"doc_id": doc_id, "answer": row["answer"],
                         "qa_score": float(row["qa_score"]), "text": body,
                         "sha256": row["sha256"]}
    return list(found.values())


def window(text: str, answer: str) -> str:
    start = text.find(answer)
    if start < 0:
        raise ValueError("Proposed answer is absent from its source")
    return text[max(0, start - 400):min(len(text), start + len(answer) + 400)]


def prompt(question: str, answer: str) -> str:
    return f"{question}\nProposed answer: {answer}"


def train_aliases() -> dict[str, list[str]]:
    split = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if digest(TRAIN, "sha256") != split["nq_train_sha256"]:
        raise ValueError("Official training answers changed")
    result = {}
    for row in records(TRAIN):
        key = normalized(row["question"])
        if key in result:
            result[key] = []
        else:
            result[key] = row["answer"]
    return result


def aliases_for(path: Path, expected_sha: str,
                questions: list[str]) -> dict[str, list[str]]:
    """Retain labels only for the explicitly requested question subset."""
    if digest(path, "sha256") != expected_sha:
        raise ValueError("Official answer source changed")
    wanted = {normalized(question) for question in questions}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        key = normalized(item["question"])
        if key in wanted:
            if key in found:
                raise ValueError(f"Ambiguous official answer for {key}")
            found[key] = item["answer"]
    if set(found) != wanted:
        raise ValueError("Missing official answer for selected question")
    return found


def build_examples() -> dict:
    if EXAMPLES.exists():
        raise FileExistsError("Refusing to overwrite training examples")
    split = json.loads(OUTPUT.read_text(encoding="utf-8"))
    selected = training_questions()
    new = read_predictions(TRAIN_PREDICTIONS, {row["qid"] for row in selected})
    sources = source_map(TRAIN_RETRIEVALS)
    if set(sources) != set(new):
        raise ValueError("Training source IDs differ from predictions")
    official_train = train_aliases()
    v17_split, v17_rows = split_rows(V17)
    old = read_predictions(V17_PREDICTIONS, {row["qid"] for row in v17_rows})
    old_sources = source_map(V17_RETRIEVALS)
    if set(old_sources) != set(old):
        raise ValueError("V17 source IDs differ from predictions")
    official_dev = aliases_for(DEV, split["nq_dev_sha256"],
                               [row["question"] for row in v17_rows])
    counts = Counter()
    with EXAMPLES.open("w", encoding="utf-8", newline="\n") as out:
        for origin, rows, predictions, passages, aliases in (
            ("nq_train", selected, new, sources, official_train),
            ("spent_v17", v17_rows, old, old_sources, official_dev),
        ):
            for row in rows:
                qid, question = row["qid"], row["question"]
                gold = aliases[normalized(question)]
                if not gold:
                    raise ValueError(f"Missing or ambiguous answer: {qid}")
                options = verified_options(predictions[qid], passages[qid])
                positives = [option for option in options if answer_match(
                    option["answer"], gold, substring=False)]
                negatives = sorted((option for option in options if not answer_match(
                    option["answer"], gold, substring=False)),
                                   key=lambda option: -option["qa_score"])[:10]
                counts[f"{origin}_questions"] += 1
                counts[f"{origin}_questions_with_positive"] += bool(positives)
                for option, label in ([(option, 1) for option in positives] +
                                      [(option, 0) for option in negatives]):
                    example = {"qid": qid, "origin": origin,
                               "question": question, "answer": option["answer"],
                               "context": window(option["text"], option["answer"]),
                               "doc_id": option["doc_id"],
                               "source_sha256": option["sha256"],
                               "reader_score": option["qa_score"], "label": label}
                    out.write(json.dumps(example, ensure_ascii=False) + "\n")
                    counts[f"{origin}_candidates"] += 1
                    counts[f"{origin}_positive_candidates"] += label
    result = {"questions": len(selected) + len(v17_rows),
              "examples_sha256": digest(EXAMPLES, "sha256"), **counts}
    (LOCAL / "training_examples.meta.json").write_text(
        json.dumps({"split_sha256": digest(OUTPUT, "sha256"),
                    "train_predictions_sha256": digest(TRAIN_PREDICTIONS, "sha256"),
                    "v17_predictions_sha256": digest(V17_PREDICTIONS, "sha256"),
                    **result}, indent=2) + "\n", encoding="utf-8")
    return result


def load_examples() -> list[dict]:
    meta = json.loads((LOCAL / "training_examples.meta.json").read_text(encoding="utf-8"))
    if meta["split_sha256"] != digest(OUTPUT, "sha256") or meta["examples_sha256"] != digest(EXAMPLES, "sha256"):
        raise ValueError("Training example manifest changed")
    return records(EXAMPLES)


def infer(model, tokenizer, rows: list[dict], device: torch.device,
          *, batch_size: int = 32) -> list[float]:
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            tokens = tokenizer([prompt(row["question"], row["answer"]) for row in batch],
                               [row["context"] for row in batch], padding=True,
                               truncation=True, max_length=MAX_LENGTH,
                               return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**tokens).logits
            scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
    return scores


def choose_cases(model, tokenizer, prediction_path: Path, retrieval_path: Path,
                 expected_qids: set[str], device: torch.device) -> dict[str, dict]:
    predictions = read_predictions(prediction_path, expected_qids)
    passages = source_map(retrieval_path)
    if set(passages) != expected_qids:
        raise ValueError("Candidate sources differ from frozen questions")
    flat = []
    for qid, case in predictions.items():
        for option in verified_options(case, passages[qid]):
            flat.append({"qid": qid, "question": case["question"],
                         "answer": option["answer"],
                         "context": window(option["text"], option["answer"]),
                         "qa_score": option["qa_score"],
                         "doc_id": option["doc_id"]})
    scores = infer(model, tokenizer, flat, device)
    chosen = {qid: {"answer": "", "score": 0.0, "doc_ids": []}
              for qid in expected_qids}
    for row, score in zip(flat, scores):
        current = chosen[row["qid"]]
        if (float(score), row["qa_score"]) > (current["score"],
                                              current.get("qa_score", -1.0)):
            chosen[row["qid"]] = {"answer": row["answer"], "score": float(score),
                                  "doc_ids": [row["doc_id"]],
                                  "qa_score": row["qa_score"]}
    return chosen


def train() -> dict:
    if MODEL.exists() or TRAIN_REPORT.exists():
        raise FileExistsError("Refusing to overwrite v19 model or report")
    examples = load_examples()
    example_meta = json.loads((LOCAL / "training_examples.meta.json").read_text(
        encoding="utf-8"))
    positives = sum(row["label"] for row in examples)
    negatives = len(examples) - positives
    if positives < 100 or negatives < 100:
        raise ValueError("Insufficient positive or negative training candidates")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION,
                                       local_files_only=True))
    if digest(base_path / "model.safetensors", "sha256") != BASE_SHA256:
        raise ValueError("Base model weights changed")
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path, num_labels=2, local_files_only=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED),
                        collate_fn=lambda batch: batch)
    steps = EPOCHS * len(loader)
    warmup = max(1, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup,
                                    max((steps - step) / max(steps - warmup, 1), 0.0)))
    weight = torch.tensor([1.0, min(10.0, negatives / positives)], device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for step, batch in enumerate(loader, 1):
            tokens = tokenizer([prompt(row["question"], row["answer"]) for row in batch],
                               [row["context"] for row in batch], padding=True,
                               truncation=True, max_length=MAX_LENGTH,
                               return_tensors="pt").to(device)
            labels = torch.tensor([row["label"] for row in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**tokens).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels,
                                                         weight=weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach().cpu())
            if step % 200 == 0:
                print(f"epoch {epoch + 1}/{EPOCHS} step {step}/{len(loader)} "
                      f"mean_loss {loss_sum / step:.4f}", flush=True)
        history.append({"epoch": epoch + 1, "mean_loss": loss_sum / len(loader),
                        "elapsed_seconds": round(time.perf_counter() - started, 2)})
    MODEL.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    split = json.loads(OUTPUT.read_text(encoding="utf-8"))
    calibration = split["spent_calibration"]
    chosen = choose_cases(model, tokenizer, V18_PREDICTIONS, V18_RETRIEVALS,
                          {row["qid"] for row in calibration} |
                          {row["qid"] for row in split["spent_diagnostic"]}, device)
    aliases = aliases_for(DEV, split["nq_dev_sha256"],
                          [row["question"] for row in calibration])
    answers = {row["qid"]: aliases[normalized(row["question"])]
               for row in calibration}
    cutoff, counts = calibrate(calibration, chosen, answers)
    frozen_policy_cutoff = json.loads((ROOT / "results" / "full_corpus_v17.json").read_text(
        encoding="utf-8"))["calibrated_score_cutoff"]
    paired_predictions = read_predictions(V18_PREDICTIONS,
        {row["qid"] for row in calibration + split["spent_diagnostic"]})
    frozen_policy = {row["qid"]: policy_prediction(
        paired_predictions[row["qid"]]["rankings"]["dense"], "consensus10")
        for row in calibration}
    paired_baseline = outcomes(calibration, frozen_policy, answers,
                               frozen_policy_cutoff)
    (MODEL / "selection.json").write_text(
        json.dumps({"cutoff": cutoff, "split_sha256": digest(OUTPUT, "sha256"),
                    "training_examples_sha256": digest(EXAMPLES, "sha256")}, indent=2) + "\n",
        encoding="utf-8")
    weights = MODEL / "model.safetensors"
    report = {"protocol": "research_protocols/answer_selector_v19.md",
              "split_sha256": digest(OUTPUT, "sha256"),
              "training_examples_sha256": digest(EXAMPLES, "sha256"),
              "nq_open_train_questions": len(split["training"]),
              "spent_v17_questions": 700,
              "training_data_counts": {key: value for key, value in example_meta.items()
                                       if key.startswith(("nq_train_", "spent_v17_"))},
              "training_candidates": len(examples), "positive_candidates": positives,
              "negative_candidates": negatives,
              "model": BASE_MODEL, "base_revision": BASE_REVISION,
              "base_weights_sha256": BASE_SHA256,
              "model_weights_sha256": digest(weights, "sha256"),
              "device": str(device), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "max_length": MAX_LENGTH, "seed": SEED,
              "positive_class_weight": float(weight[1].item()),
              "history": history, "calibration_questions": len(calibration),
              "calibrated_cutoff": cutoff, "calibration": counts,
              "paired_frozen_baseline_calibration": paired_baseline,
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    TRAIN_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_model() -> tuple:
    report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    if report["split_sha256"] != digest(OUTPUT, "sha256") or report["model_weights_sha256"] != digest(MODEL / "model.safetensors", "sha256"):
        raise ValueError("Trained model or frozen split changed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True).to(device).eval()
    return model, tokenizer, device, report


def predict_eval() -> dict:
    metadata = SELECTED.with_suffix(".meta.json")
    if SELECTED.exists() or metadata.exists():
        raise FileExistsError("Refusing to overwrite fresh model predictions")
    started = time.perf_counter()
    model, tokenizer, device, report = load_model()
    split = json.loads(OUTPUT.read_text(encoding="utf-8"))
    rows = split["evaluation"]
    chosen = choose_cases(model, tokenizer, EVAL_PREDICTIONS, EVAL_RETRIEVALS,
                          {row["qid"] for row in rows}, device)
    with SELECTED.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps({"qid": row["qid"], **chosen[row["qid"]]},
                                 ensure_ascii=False) + "\n")
    result = {"questions": len(rows), "selected_sha256": digest(SELECTED, "sha256"),
              "model_weights_sha256": report["model_weights_sha256"],
              "selector_seconds": round(time.perf_counter() - started, 2)}
    metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def score_eval() -> dict:
    if EVAL_REPORT.exists() or CASES.exists():
        raise FileExistsError("Refusing to overwrite fresh evaluation or case rows")
    split = json.loads(OUTPUT.read_text(encoding="utf-8"))
    rows = split["evaluation"]
    selected = read_predictions(SELECTED, {row["qid"] for row in rows})
    predictions = read_predictions(EVAL_PREDICTIONS, {row["qid"] for row in rows})
    answers_by_text = aliases_for(DEV, split["nq_dev_sha256"],
                                  [row["question"] for row in rows])
    answers = {row["qid"]: answers_by_text[normalized(row["question"])] for row in rows}
    report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    baseline_cutoff = json.loads((ROOT / "results" / "full_corpus_v17.json").read_text(
        encoding="utf-8"))["calibrated_score_cutoff"]
    baseline = {qid: policy_prediction(case["rankings"]["dense"], "consensus10")
                for qid, case in predictions.items()}
    selected_counts = outcomes(rows, selected, answers, report["calibrated_cutoff"])
    baseline_counts = outcomes(rows, baseline, answers, baseline_cutoff)
    candidate_presence = 0
    unverified = 0
    reader_span_failures = 0
    with CASES.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            qid = row["qid"]
            reader_span_failures += sum(not item["span_verified"]
                                        for method in ("lexical", "dense", "fusion")
                                        for item in predictions[qid]["rankings"][method])
            options = [item for method in ("lexical", "dense", "fusion")
                       for item in predictions[qid]["rankings"][method]
                       if item["span_verified"]]
            available = any(answer_match(item["answer"], answers[qid],
                                         substring=False) for item in options)
            candidate_presence += available
            chosen = selected[qid]
            answer = (chosen["answer"] if chosen["score"] >=
                      report["calibrated_cutoff"] else "")
            old = baseline[qid]
            old_answer = old["answer"] if old["score"] >= baseline_cutoff else ""
            if answer and not any(item["doc_id"] in chosen["doc_ids"] and
                                  item["answer"] == answer for item in options):
                unverified += 1
            out.write(json.dumps({"qid": qid, "question": row["question"],
                                  "answer_aliases": answers[qid],
                                  "reader_exact_candidate": available,
                                  "trained_answer": answer,
                                  "trained_source_ids": chosen["doc_ids"] if answer else [],
                                  "trained_strict_match": answer_match(
                                      answer, answers[qid], substring=False),
                                  "baseline_answer": old_answer,
                                  "baseline_strict_match": answer_match(
                                      old_answer, answers[qid], substring=False)},
                                 ensure_ascii=False) + "\n")
    gate = (selected_counts["strict_match"] >= 150 and
            selected_counts["nonmatching_emitted"] <= 50 and unverified == 0 and
            selected_counts["strict_match"] > baseline_counts["strict_match"])
    result = {"protocol": "research_protocols/answer_selector_v19.md",
              "split_sha256": digest(OUTPUT, "sha256"),
              "model_weights_sha256": report["model_weights_sha256"],
              "training_examples_sha256": report["training_examples_sha256"],
              "reader_predictions_sha256": digest(EVAL_PREDICTIONS, "sha256"),
              "selected_predictions_sha256": digest(SELECTED, "sha256"),
              "case_rows_sha256": digest(CASES, "sha256"),
              "corpus_passages": 2681468, "questions": len(rows),
              "baseline_v17_frozen_policy": baseline_counts,
              "trained_v19_selector": selected_counts,
              "reader_exact_candidate_questions": candidate_presence,
              "reader_source_span_failures": reader_span_failures,
              "unverified_emissions": unverified, "gate_passed": gate,
              "baseline_cutoff": baseline_cutoff,
              "trained_cutoff": report["calibrated_cutoff"],
              "retrieval_reader_timing": json.loads(EVAL_PREDICTIONS.with_suffix(
                  ".meta.json").read_text(encoding="utf-8")),
              "selector_timing": json.loads(SELECTED.with_suffix(
                  ".meta.json").read_text(encoding="utf-8")),
              "limitations": ["Public NQ benchmark clean QA, not poisoning resistance",
                              "Answer-alias labels may miss valid equivalent answers",
                              "Corpus passages are hash-pinned but not authority-reviewed"]}
    EVAL_REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "train", "predict-eval", "score-eval"))
    args = parser.parse_args()
    fn = {"build": build_examples, "train": train,
          "predict-eval": predict_eval, "score-eval": score_eval}[args.phase]
    print(json.dumps(fn(), indent=2), flush=True)


if __name__ == "__main__":
    main()
