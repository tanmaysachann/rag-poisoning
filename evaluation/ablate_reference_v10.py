"""Diagnostic shuffled-reference ablation on held-out v10 questions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from detect.reference_conditioned_v9 import ReferenceConditionedScorer
from evaluation.evaluate_poisonarena_v9 import build_cases, normalized_question


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    args = parser.parse_args()
    cases, _ = build_cases(Path("data/poisonarena_v9"),
                           Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip"),
                           args.beir_archive)
    cases = sorted((case for case in cases if int(hashlib.sha256(
        ("v10-pilot:" + normalized_question(case["query"])).encode()
    ).hexdigest()[:8], 16) % 10 >= 7), key=lambda case: case["query_id"])
    if len(cases) != 29:
        raise ValueError("Unexpected held-out question count")
    threshold = json.loads(Path("results/multifamily_v10_calibration.json")
                           .read_text(encoding="utf-8"))["threshold"]
    torch.set_num_threads(min(8, torch.get_num_threads()))
    scorer = ReferenceConditionedScorer(
        Path("artifacts/multifamily_v10"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rows, kinds = [], []
    for index, case in enumerate(cases):
        other = cases[(index + 1) % len(cases)]["reference"]
        documents = [("clean", doc["text"]) for doc in case["clean"]]
        documents += [(family, text) for family in ("gaslite", "poisonedrag_black")
                      for text in case["attacks"][family]]
        for kind, candidate in documents:
            rows.append(({"query": case["query"], "reference": case["reference"],
                          "candidate": candidate},
                         {"query": case["query"], "reference": other,
                          "candidate": candidate}))
            kinds.append(kind)
    original = scorer.score([row[0] for row in rows], batch_size=24)
    shuffled = scorer.score([row[1] for row in rows], batch_size=24)
    report = {"protocol": "research_protocols/multifamily_detector_v10.md",
              "model_weights_sha256": hashlib.sha256(
                  Path("artifacts/multifamily_v10/model.safetensors").read_bytes()).hexdigest(),
              "questions": len(cases), "pairs": len(rows), "threshold": threshold,
              "mean_absolute_probability_change": float(np.mean(np.abs(original - shuffled))),
              "changed_flag_count": int(np.sum((original >= threshold) !=
                                                (shuffled >= threshold))),
              "by_kind": {kind: {"pairs": kinds.count(kind),
                                 "original_flags": int(sum(original[index] >= threshold
                                                       for index, value in enumerate(kinds)
                                                       if value == kind)),
                                 "shuffled_flags": int(sum(shuffled[index] >= threshold
                                                       for index, value in enumerate(kinds)
                                                       if value == kind))}
                          for kind in sorted(set(kinds))},
              "interpretation_limit": "Shuffled references are out of distribution; this measures dependence, not factual correctness."}
    output = Path("results/multifamily_v10_reference_ablation.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
