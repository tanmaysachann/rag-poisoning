"""Explore explainable scanner rules on previously inspected development data.

Never reads the NFCorpus holdout. Printed metrics are design evidence only.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def domain(name: str, path: Path) -> tuple[str, list[dict]]:
    rows = read(path)
    refs = {row["query_id"]: row["fusion_score"] for row in rows
            if row["label"] == "reference"}
    if len(refs) != len({row["query_id"] for row in rows}):
        raise ValueError(f"Missing or repeated reference: {name}")
    for row in rows:
        row["reference_fusion_score"] = refs[row["query_id"]]
    return name, rows


def metrics(rows: list[dict], rule) -> dict:
    clean = [row for row in rows if row["label"] != "poison"]
    attack = [row for row in rows if row["label"] == "poison"]
    return {
        "attack": [sum(bool(rule(row)) for row in attack), len(attack)],
        "clean": [sum(bool(rule(row)) for row in clean), len(clean)],
    }


def main() -> None:
    domains = dict((domain(
        "poisonarena", ROOT / "results/v25_poisonarena_dev_scores.jsonl"),
        domain("scifact", ROOT / "results/v24_blind_scifact.cases.jsonl"),
    ))
    for name, rows in domains.items():
        for rule_name, rule in (
            ("v24", lambda r: r["containment_flag"] or r["fusion_score"] >= 4.272025739459554),
            ("v3", lambda r: r["v3_flag"]),
            ("union", lambda r: r["v3_flag"] or r["containment_flag"] or r["fusion_score"] >= 4.272025739459554),
        ):
            print(name, rule_name, metrics(rows, rule))
    proposals = []
    for v3_threshold, fusion_threshold, margin in itertools.product(
            (0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 0.995),
            (4.272025739459554, 4.75, 5.25, 5.75, 6.25, 7.0),
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)):
        def decision(row: dict) -> bool:
            return bool(row["containment_flag"]
                        or row["v3_risk"] >= v3_threshold
                        or (row["fusion_score"] >= fusion_threshold
                            and row["fusion_score"] - row["reference_fusion_score"] >= margin))
        results = {name: metrics(rows, decision) for name, rows in domains.items()}
        fprs = {name: score["clean"][0] / score["clean"][1]
                for name, score in results.items()}
        recalls = {name: score["attack"][0] / score["attack"][1]
                   for name, score in results.items()}
        if max(fprs.values()) > 0.05:
            continue
        proposals.append({
            "v3_threshold": v3_threshold, "fusion_threshold": fusion_threshold,
            "reference_margin": margin, "results": results,
            "fpr": fprs, "recall": recalls,
            "ranking": [min(recalls.values()), sum(recalls.values()) / len(recalls),
                        -max(fprs.values())],
        })
    proposals.sort(key=lambda row: row["ranking"], reverse=True)
    print("feasible_rules", len(proposals))
    for row in proposals[:15]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
