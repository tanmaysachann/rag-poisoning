"""Replay the previously observed v3 stealth failures through source screening."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from backend.research_lab import ROOT, case_catalog, run_case


def main() -> None:
    output = ROOT / "results" / "evidence_claim_live_validation.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite the existing live evidence replay")
    failures = [row["qid"] for row in case_catalog()["cases"]
                if row["saved_stealth_failure"]]
    totals = Counter()
    for index, qid in enumerate(failures, 1):
        result = run_case(qid, "stealth", "accepted_ingest")
        totals["cases"] += 1
        totals["previous_v3_attack_success"] += result["defended_attack_success"]
        totals["retrieved_attacks"] += result["attack_retrieved_count"]
        totals["source_excluded_attacks"] += result["evidence_attack_quarantined_count"]
        totals["source_screen_attack_success"] += result["evidence_attack_source_success"]
        totals["source_screen_clean_alias_match"] += result["evidence_clean_alias_recovered"]
        totals["source_screen_abstained"] += result["evidence_defended"]["source_doc_id"] is None
        if index % 5 == 0 or index == len(failures):
            print(f"Evidence replay: {index}/{len(failures)}", flush=True)
    report = {
        "split": "previously_inspected_v3_validation_saved_failures",
        "attack": "stealth answer substitution accepted at ingest",
        "metrics": dict(totals),
        "limitations": [
            "These were selected because the old detector had already failed; this is development evidence.",
            "The attacker cannot change the pinned preattack corpus or its manifest.",
            "Source screening detects unauthorized modification, not false claims inside approved documents.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
