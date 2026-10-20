"""Fine-tune the V9 reference scorer on six PoisonArena attack families."""

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
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detect.reference_conditioned_v9 import LABELS, LABEL_TO_ID
from detect.train_reference_conditioned_v9 import encode_rows, validation_metrics


def read_verified_rows(directory: Path, split: str, manifest: dict) -> list[dict]:
    path = directory / f"{split}.jsonl"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest["splits"][split]["sha256"]:
        raise ValueError(f"{split} data changed since the manifest")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if len(rows) != manifest["splits"][split]["rows"]:
        raise ValueError(f"{split} row count changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/multifamily_v20"))
    parser.add_argument("--base", type=Path, default=Path("artifacts/reference_detector_v9"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multifamily_v20"))
    parser.add_argument("--report", type=Path, default=Path("results/multifamily_v20_training.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Invalid training dimensions")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite model: {args.output}")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows = read_verified_rows(args.data, "train", manifest)
    calibration_rows = read_verified_rows(args.data, "calibration", manifest)
    if {row["query_id"] for row in train_rows} & {
        row["query_id"] for row in calibration_rows
    }:
        raise ValueError("Question leakage between training and calibration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, local_files_only=True
    ).to(device)
    labels = {int(index): label for index, label in model.config.id2label.items()}
    if labels != dict(enumerate(LABELS)):
        raise ValueError("Unexpected base-model labels")
    train = encode_rows(train_rows, tokenizer)
    calibration = encode_rows(calibration_rows, tokenizer)
    loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda items: items,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    warmup = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup,
                                    max((total_steps - step) /
                                        max(total_steps - warmup, 1), 0.0)),
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(loader, 1):
            tokens = tokenizer.pad(
                [row["tokens"] for row in batch], return_tensors="pt"
            ).to(device)
            targets = torch.tensor([row["label"] for row in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(**tokens).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.float(), targets,
                    weight=torch.tensor([2.0, 1.0, 2.0], device=device),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(loader)} "
                      f"loss={np.mean(losses[-100:]):.4f}", flush=True)
        metrics = validation_metrics(model, tokenizer, calibration, device,
                                     args.batch_size * 2)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        **metrics})
        print(json.dumps(history[-1]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    weights = args.output / "model.safetensors"
    report = {
        "model": "MiniLM reference-conditioned, V9 warm start, six-family fine-tune",
        "training_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "v9_base_weights_sha256": hashlib.sha256(
            (args.base / "model.safetensors").read_bytes()
        ).hexdigest(),
        "model_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "seed": args.seed, "device": str(device), "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_questions": len({row["query_id"] for row in train_rows}),
        "calibration_questions": len({row["query_id"] for row in calibration_rows}),
        "train_rows": len(train_rows), "calibration_rows": len(calibration_rows),
        "train_windows": len(train), "calibration_windows": len(calibration),
        "train_label_counts": dict(Counter(row["label"] for row in train_rows)),
        "history": history, "seconds": round(time.perf_counter() - started, 2),
        "limitations": manifest["limitations"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
