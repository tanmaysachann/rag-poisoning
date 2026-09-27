"""Verify the saved development template probes against their raw cases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from backend.research_lab import _inputs
from config import ROOT_DIR


STYLES = ("qa_header", "answer_key", "postscript")
COUNTS = {
    "retrieved": "attack_retrieved",
    "quarantined": "attack_quarantined",
    "undefended_attack_successes": "undefended_attack_success",
    "defended_attack_successes": "defended_attack_success",
    "clean_alias_recovered": "clean_alias_recovered",
}


def verify(root: Path = ROOT_DIR) -> dict:
    summary_path = root / "results/unseen_templates_validation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in summary_path.with_suffix(".cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queries, _, attacks, _ = _inputs()
    expected = {(style, qid) for style in STYLES for qid in queries}
    observed = Counter((row["style"], row["qid"]) for row in cases)
    if (summary["split"] != "validation" or summary["surface"] != "accepted_ingest"
            or set(summary["styles"]) != set(STYLES)
            or set(observed) != expected or any(count != 1 for count in observed.values())):
        raise ValueError("Template probe split, styles, or query coverage differs")
    for row in cases:
        qid = row["qid"]
        if (row["wrong_answer"] != attacks["greedy"][qid]["wrong_answer"]
                or row["attack_doc_id"] != attacks["greedy"][qid]["attack_doc_id"]
                or any(type(row[field]) is not bool for field in COUNTS.values())
                or not (0 <= row["attack_risk"] <= 1)
                or row["latency_ms"] < 0
                or row["attack_quarantined"] and not row["attack_retrieved"]
                or row["defended_attack_success"] and row["attack_quarantined"]):
            raise ValueError(f"Template probe case is inconsistent: {row['style']} {qid}")
    for style in STYLES:
        rows = [row for row in cases if row["style"] == style]
        counts = {"cases": len(rows)}
        counts.update({key: sum(row[field] for row in rows) for key, field in COUNTS.items()})
        if summary["styles"][style] != counts:
            raise ValueError(f"Template probe summary differs from cases: {style}")
    return {"status": "verified", "styles": len(STYLES), "cases": len(cases)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
