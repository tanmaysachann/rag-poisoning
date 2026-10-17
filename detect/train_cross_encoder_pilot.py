"""Fine-tune an offline MiniLM query/passage classifier on the v4 pilot.

Only train and validation are read. The final test splits are not opened here.
This is an experimental artifact, not a replacement for the deployed detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detect.pilot_data import examples, nearest_other, read_split, wrong_answer
from detect.research_detector import threshold_for_fpr

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_examples(rows: list[dict], answer_pool: list[str], training: bool) -> list[dict]:
    others = nearest_other(rows)
    return [
        example for row in rows
        for example in examples(row, wrong_answer(row, answer_pool), others[row["qid"]], training=training)
    ]


@torch.inference_mode()
def predict(model, tokenizer, rows: list[dict], device: torch.device,
            batch_size: int, max_length: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        tokens = tokenizer(
            [row["question"] for row in batch], [row["text"] for row in batch],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        logits = model(**tokens).logits
        outputs.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
    return np.asarray(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/detector_pilot_v4"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/detector_pilot_v4"))
    parser.add_argument("--report", type=Path, default=Path("results/detector_pilot_v4_training.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1417)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = read_split(args.data, "train")
    validation = read_split(args.data, "validation")
    answer_pool = [alias for row in train for alias in row["answer_aliases"]]
    train_examples = make_examples(train, answer_pool, training=True)
    validation_examples = make_examples(validation, answer_pool, training=True)
    labels = np.asarray([row["label"] for row in validation_examples])
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, local_files_only=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda batch: batch)
    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps,
                         max((total_steps - step) / max(total_steps - warmup_steps, 1), 0.0)),
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best_auc = -1.0
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(loader, 1):
            tokens = tokenizer(
                [row["question"] for row in batch],
                [row["text"] for row in batch],
                padding=True, truncation=True, max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            target = torch.tensor([row["label"] for row in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = model(**tokens, labels=target).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(loader)} loss={np.mean(losses[-100:]):.4f}", flush=True)
        scores = predict(model, tokenizer, validation_examples, device,
                         args.batch_size * 2, args.max_length)
        auc = float(roc_auc_score(labels, scores))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_roc_auc": auc,
                        "validation_pr_auc": float(average_precision_score(labels, scores))})
        print(json.dumps(history[-1]), flush=True)
        if auc > best_auc:
            best_auc = auc
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)

    selected = AutoModelForSequenceClassification.from_pretrained(
        args.output, local_files_only=True
    ).to(device)
    scores = predict(selected, tokenizer, validation_examples, device,
                     args.batch_size * 2, args.max_length)
    threshold = threshold_for_fpr(scores[labels == 0], 0.05)
    predictions = scores >= threshold
    family_metrics = {}
    for family in sorted({row["family"] for row in validation_examples}):
        indices = [i for i, row in enumerate(validation_examples) if row["family"] == family]
        family_metrics[family] = {
            "count": len(indices),
            "flag_rate": float(np.mean(predictions[indices])),
            "mean_score": float(np.mean(scores[indices])),
        }
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "model": BASE_MODEL, "model_type": "fine_tuned_query_passage_cross_encoder",
        "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        "data_counts": manifest["counts"], "train_examples": len(train_examples),
        "validation_examples": len(validation_examples), "device": str(device),
        "seed": args.seed, "batch_size": args.batch_size,
        "max_length": args.max_length, "epochs": args.epochs,
        "threshold_target_fpr": 0.05, "threshold": threshold,
        "observed_validation_fpr": float(np.mean(predictions[labels == 0])),
        "observed_validation_recall": float(np.mean(predictions[labels == 1])),
        "validation_roc_auc": float(roc_auc_score(labels, scores)),
        "validation_pr_auc": float(average_precision_score(labels, scores)),
        "validation_by_family": family_metrics,
        "epoch_history": history, "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitations": [
            "Training poisons are synthetic and locally generated.",
            "The model reads only query and candidate, without a trusted factual reference.",
            "The clean FAQ control and false FAQ differ mainly in factual answer content.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
