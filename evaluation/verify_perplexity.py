"""Verify saved perplexity probe summary against raw validation rows."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark
from data.validate_multisupport_benchmark import read_jsonl
from evaluation.evaluate_perplexity import flag_high_nll


def verify(root: Path = ROOT_DIR) -> dict:
    validate_benchmark(root / "data/benchmark")
    queries = {row["qid"]: row for row in read_jsonl(root / "data/benchmark/validation/queries.jsonl")}
    summary = json.loads((root / "results/perplexity_validation.json").read_text(encoding="utf-8"))
    cases = read_jsonl(root / "results/perplexity_validation.cases.jsonl")
    if len(cases) != len(queries) or {row["qid"] for row in cases} != set(queries):
        raise ValueError("Perplexity cases do not cover the validation split")
    threshold = float(summary["nll_threshold"])
    for row in cases:
        if row["clean_flagged"] != flag_high_nll(row["clean_nll"], threshold):
            raise ValueError(f"Clean flag differs: {row['qid']}")
        if row["attack_flagged"] != flag_high_nll(row["attacked_nll"], threshold):
            raise ValueError(f"Attack flag differs: {row['qid']}")
    y_true = [0] * len(cases) + [1] * len(cases)
    y_score = [row["clean_nll"] for row in cases] + [row["attacked_nll"] for row in cases]
    expected = {
        "clean_fpr": sum(row["clean_flagged"] for row in cases) / len(cases),
        "attack_recall": sum(row["attack_flagged"] for row in cases) / len(cases),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "mean_clean_nll": sum(row["clean_nll"] for row in cases) / len(cases),
        "mean_attacked_nll": sum(row["attacked_nll"] for row in cases) / len(cases),
    }
    if any(abs(summary[key] - value) > 1e-9 for key, value in expected.items()):
        raise ValueError("Perplexity summary disagrees with its cases")
    if summary["split"] != "validation" or summary["train_clean_cases"] != 100:
        raise ValueError("Perplexity summary has wrong split or calibration size")
    return {"status": "verified", "cases": len(cases), "test_attacks_evaluated": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
