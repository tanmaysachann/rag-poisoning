"""Train the source-labelled, reference-conditioned MiniLM research scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detect.reference_conditioned_v9 import LABELS, LABEL_TO_ID, token_windows


BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def validation_question(qid: str) -> bool:
    return int(hashlib.sha256(("v9-validation:" + qid).encode()).hexdigest()[:8], 16) % 10 == 0


def encode_rows(rows: list[dict], tokenizer) -> list[dict]:
    encoded = []
    for row in rows:
        for tokens in token_windows(tokenizer, row["query"], row["reference"],
                                    row["candidate"]):
            encoded.append({"tokens": tokens, "label": LABEL_TO_ID[row["label"]],
                            "query_id": row["query_id"]})
    return encoded


@torch.inference_mode()
def validation_metrics(model, tokenizer, rows: list[dict], device: torch.device,
                       batch_size: int) -> dict:
    model.eval()
    probabilities = []
    labels = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        tokens = tokenizer.pad([row["tokens"] for row in batch],
                               return_tensors="pt").to(device)
        probabilities.extend(torch.softmax(model(**tokens).logits.float(), dim=1)
                             .cpu().numpy())
        labels.extend(row["label"] for row in batch)
    prediction = np.asarray(probabilities).argmax(axis=1)
    return {"window_macro_f1": float(f1_score(labels, prediction, average="macro")),
            "window_poison_recall": float(np.mean(prediction[np.asarray(labels) ==
                                                        LABEL_TO_ID["poison"]] ==
                                                  LABEL_TO_ID["poison"])),
            "window_counts": dict(Counter(LABELS[label] for label in labels))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/reference_detector_v9"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference_detector_v9"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/reference_detector_v9_training.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Invalid training dimensions")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite model: {args.output}")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = args.data / "train.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["train_sha256"]:
        raise ValueError("Training data changed since manifest")
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    train_rows = [row for row in rows if not validation_question(row["query_id"])]
    validation_rows = [row for row in rows if validation_question(row["query_id"])]
    if {row["query_id"] for row in train_rows} & {
            row["query_id"] for row in validation_rows}:
        raise ValueError("Question leaked into validation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, local_files_only=True,
        id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID).to(device)
    train = encode_rows(train_rows, tokenizer)
    validation = encode_rows(validation_rows, tokenizer)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda items: items)
    weights = torch.tensor([2.0, 1.0, 2.0], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    warmup = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup,
                                    max((total_steps - step) /
                                        max(total_steps - warmup, 1), 0.0)))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best_f1 = -1.0
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(loader, 1):
            tokens = tokenizer.pad([row["tokens"] for row in batch],
                                   return_tensors="pt").to(device)
            labels = torch.tensor([row["label"] for row in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(**tokens).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels,
                                                         weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 250 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(loader)} "
                      f"loss={np.mean(losses[-250:]):.4f}", flush=True)
        metrics = validation_metrics(model, tokenizer, validation, device,
                                     args.batch_size * 2)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        **metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics["window_macro_f1"] > best_f1:
            best_f1 = metrics["window_macro_f1"]
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
    report = {"base_model": BASE_MODEL, "data_manifest_sha256": hashlib.sha256(
        manifest_path.read_bytes()).hexdigest(), "seed": args.seed,
        "model_weights_sha256": hashlib.sha256(
            (args.output / "model.safetensors").read_bytes()).hexdigest(),
        "device": str(device), "epochs": args.epochs, "batch_size": args.batch_size,
        "train_questions": len({row["query_id"] for row in train_rows}),
        "validation_questions": len({row["query_id"] for row in validation_rows}),
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "train_windows": len(train), "validation_windows": len(validation),
        "label_counts": dict(Counter(row["label"] for row in train_rows)),
        "history": history, "selected_validation_macro_f1": best_f1,
        "seconds": round(time.perf_counter() - started, 2),
        "limitation": "Labels are passage provenance, not semantic entailment or proven malice."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
