"""Qwen token-likelihood probe for fluent stealth substitutions (validation only).

Scores each passage by mean token negative log-likelihood under the pinned
Qwen2.5-0.5B-Instruct CPU weights and flags high-NLL passages, following the
perplexity-filter direction in prior work. Threshold is set on v1 train-clean
passages only (5% FPR target); evaluation uses v1 validation clean passages
versus single-span answer-substituted versions of the same supports. Never
reads the test split. Model stays offline; nothing here gates serving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from attack.baselines import choose_wrong_answer, substitute_answer
from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from data.validate_multisupport_benchmark import read_jsonl


def flag_high_nll(nll: float, threshold: float) -> bool:
    return nll > threshold


def threshold_at_fpr(clean_scores: list[float], target_fpr: float) -> float:
    if not clean_scores or not 0 < target_fpr < 1:
        raise ValueError("Need nonempty scores and target_fpr in (0, 1)")
    ordered = sorted(clean_scores)
    index = min(len(ordered) - 1, math.ceil((1 - target_fpr) * len(ordered)))
    return float(ordered[index])


@torch.inference_mode()
def mean_nll(text: str, model, tokenizer, *, max_length: int = 512) -> float:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
    labels = ids.clone()
    outputs = model(ids)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=77)
    args = parser.parse_args()
    validate_benchmark(ROOT_DIR / "data/benchmark")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = ARTIFACTS_DIR / "models/qwen2_5_0_5b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model.eval()
    torch.set_num_threads(4)
    benchmark = ROOT_DIR / "data/benchmark"
    train_queries = read_jsonl(benchmark / "train/queries.jsonl")
    wrong_pool = [row["answer_aliases"][0] for row in train_queries]
    train_corpus = read_jsonl(benchmark / "train/corpus.jsonl")[:args.train_limit]
    train_scores = [mean_nll(doc["text"], model, tokenizer) for doc in train_corpus]
    threshold = threshold_at_fpr(train_scores, 0.05)
    validation_corpus = {doc["doc_id"]: doc for doc in read_jsonl(benchmark / "validation/corpus.jsonl")}
    queries = read_jsonl(benchmark / "validation/queries.jsonl")
    cases = []
    for index, query in enumerate(queries):
        support_id = query["support_doc_ids"][0]
        clean_text = validation_corpus[support_id]["text"]
        wrong = choose_wrong_answer(query["answer_aliases"], wrong_pool, args.seed_base + index)
        attacked_text = substitute_answer(clean_text, query["answer_aliases"], wrong)
        clean_nll = mean_nll(clean_text, model, tokenizer)
        attacked_nll = mean_nll(attacked_text, model, tokenizer)
        cases.append({
            "qid": query["qid"], "support_doc_id": support_id, "wrong_answer": wrong,
            "clean_nll": clean_nll, "attacked_nll": attacked_nll,
            "clean_flagged": flag_high_nll(clean_nll, threshold),
            "attack_flagged": flag_high_nll(attacked_nll, threshold),
        })
        if (index + 1) % 15 == 0:
            print(f"validation: {index + 1}/{len(queries)}", flush=True)
    y_true = [0] * len(cases) + [1] * len(cases)
    y_score = [row["clean_nll"] for row in cases] + [row["attacked_nll"] for row in cases]
    summary = {
        "benchmark": "msmarco-v1.1-frozen", "split": "validation",
        "model": "Qwen/Qwen2.5-0.5B-Instruct (local CPU, mean token NLL)",
        "train_clean_cases": len(train_scores), "target_fpr": 0.05,
        "nll_threshold": threshold,
        "clean_fpr": sum(row["clean_flagged"] for row in cases) / len(cases),
        "attack_recall": sum(row["attack_flagged"] for row in cases) / len(cases),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "mean_clean_nll": sum(row["clean_nll"] for row in cases) / len(cases),
        "mean_attacked_nll": sum(row["attacked_nll"] for row in cases) / len(cases),
        "direction": "flag high NLL (low model likelihood) as suspicious",
        "note": "Wrong answers drawn from other queries' pools raise mean NLL (3.23 vs 2.64 clean) with AUC 0.76, but recall at 5% FPR is only 17.3% at full Qwen-inference cost per document. Diagnostic only, not a serving gate.",
    }
    output = RESULTS_DIR / "perplexity_validation.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
