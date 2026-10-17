"""Recompute aggregate attack intervals by question from saved v5 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.evaluate_detector_pilot import _summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases_path = Path(report["cases_file"])
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    for split in ("test", "domain_test"):
        attacks = [row for row in cases if row["split"] == split and row["case_type"] == "attack"]
        if not attacks or "combined_attack_success" not in attacks[0]:
            raise ValueError("Saved cases lack combined-gate outcomes")
        report["test_splits"][split]["attack_aggregate"] = _summarize(attacks, attack=True)

    ms = report["test_splits"]["test"]
    domain = report["test_splits"]["domain_test"]
    ms_attack, ms_clean = ms["attack_aggregate"], ms["clean_control"]
    domain_attack, domain_clean = domain["attack_aggregate"], domain["clean_control"]
    report["combined_adoption_gate"] = {
        "ms_attack_improved_with_positive_paired_ci": (
            ms_attack["old_minus_combined_attack_success"] > 0
            and ms_attack["combined_paired_difference_bootstrap_95"]["lower"] > 0
        ),
        "ms_clean_noninferior": (
            ms_clean["combined_minus_old_clean_alias_match"] >= -0.02
            and ms_clean["combined_paired_difference_bootstrap_95"]["lower"] > -0.05
        ),
        "domain_attack_not_worse": domain_attack["old_minus_combined_attack_success"] >= 0,
        "domain_clean_noninferior": domain_clean["combined_minus_old_clean_alias_match"] >= -0.02,
    }
    report["combined_adoption_gate"]["pass"] = all(report["combined_adoption_gate"].values())
    report["aggregate_attack_interval_method"] = "paired bootstrap clustered by question (four layouts per question)"
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ms_attack": ms_attack["old_minus_combined_attack_success"],
        "ms_attack_interval": ms_attack["combined_paired_difference_bootstrap_95"],
        "ms_clean": ms_clean["combined_minus_old_clean_alias_match"],
        "domain_attack": domain_attack["old_minus_combined_attack_success"],
        "domain_clean": domain_clean["combined_minus_old_clean_alias_match"],
        "adoption_gate": report["combined_adoption_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
