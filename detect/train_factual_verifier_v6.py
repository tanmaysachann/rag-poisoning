"""Train a small, reference-aware factual verifier on real VitaminC revisions.

The model predicts whether a claim is supported, refuted, or unsupported by a
separately supplied evidence passage. It does not classify a RAG document as
poisoned in isolation. The sealed test split is never read by this module.
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detect.research_detector import threshold_for_fpr

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@torch.inference_mode()
def predict(model, tokenizer, rows: list[dict], device: torch.device,
            batch_size: int, max_length: int, *, evidence_first: bool = False) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        first = "evidence" if evidence_first else "claim"
        second = "claim" if evidence_first else "evidence"
        tokens = tokenizer(
            [row[first] for row in batch], [row[second] for row in batch],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        output.append(torch.softmax(model(**tokens).logits.float(), dim=1).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/factual_verifier_v6"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/factual_verifier_v6"))
    parser.add_argument("--report", type=Path, default=Path("results/factual_verifier_v6_training.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a model: {args.output}")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 32:
        raise ValueError("Invalid training dimensions")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("train", "validation", "calibration"):
        path = args.data / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["file_sha256"][path.name]:
            raise ValueError(f"Frozen data changed: {path}")
    train = read_rows(args.data / "train.jsonl")
    validation = read_rows(args.data / "validation.jsonl")
    calibration = read_rows(args.data / "calibration.jsonl")
    if set(row["page"] for row in train) & set(row["page"] for row in validation + calibration):
        raise ValueError("Training page appears in validation/calibration")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, local_files_only=True,
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda rows: rows)
    total_steps = args.epochs * len(loader)
    warmup = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup,
                         max((total_steps - step) / max(total_steps - warmup, 1), 0.0)),
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_f1 = -1.0
    history = []
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(loader, 1):
            tokens = tokenizer(
                [row["claim"] for row in batch],
                [row["evidence"] for row in batch],
                padding=True, truncation=True, max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor([LABEL_TO_ID[row["label"]] for row in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = model(**tokens, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 500 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(loader)} loss={np.mean(losses[-500:]):.4f}", flush=True)
        probabilities = predict(model, tokenizer, validation, device,
                                args.batch_size * 2, args.max_length)
        truth = [LABEL_TO_ID[row["label"]] for row in validation]
        macro_f1 = float(f1_score(truth, probabilities.argmax(axis=1), average="macro"))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_macro_f1": macro_f1})
        print(json.dumps(history[-1]), flush=True)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
    selected = AutoModelForSequenceClassification.from_pretrained(
        args.output, local_files_only=True
    ).to(device)
    probabilities = predict(selected, tokenizer, calibration, device,
                            args.batch_size * 2, args.max_length)
    # Compare in float64: nextafter at the calibrated boundary must not be
    # rounded back to float32, or a tied boundary can exceed the FPR budget.
    refute_scores = probabilities[:, LABEL_TO_ID["REFUTES"]].astype(np.float64)
    thresholds = {
        label: threshold_for_fpr(
            refute_scores[[row["label"] == label for row in calibration]], 0.01
        ) for label in ("SUPPORTS", "NOT ENOUGH INFO")
    }
    threshold = max(thresholds.values())
    report = {
        "model": BASE_MODEL, "task": "trusted-evidence claim verification",
        "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_directory": str(args.output).replace("\\", "/"),
        "device": str(device), "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "train_examples": len(train), "validation_examples": len(validation),
        "calibration_examples": len(calibration), "history": history,
        "selected_validation_macro_f1": best_f1,
        "quarantine_score": "P(REFUTES | claim, trusted evidence)",
        "target_fpr_each_non_refuted_class": 0.01,
        "calibration_thresholds": thresholds, "quarantine_threshold": threshold,
        "calibration_refutes_recall": float(np.mean(
            refute_scores[[row["label"] == "REFUTES" for row in calibration]] >= threshold
        )),
        "calibration_supports_false_quarantine": float(np.mean(
            refute_scores[[row["label"] == "SUPPORTS" for row in calibration]] >= threshold
        )),
        "calibration_unsupported_false_quarantine": float(np.mean(
            refute_scores[[row["label"] == "NOT ENOUGH INFO" for row in calibration]] >= threshold
        )),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitation": "Contradiction is not proof of malicious origin; runtime requires independent trusted evidence.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
