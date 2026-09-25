"""Audit whether pinned MS MARCO rows contain multiple answer-bearing passages.

This does not claim independent provenance: MS MARCO passage records do not
establish that two passages came from separate trustworthy sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.build_benchmark import _answers, _contains_answer, normalize_text


def audit(rows, *, limit: int) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    counts = {"rows_scanned": 0, "answerable_rows": 0, "one_answer_passage": 0,
              "two_or_more_answer_passages": 0, "two_or_more_selected_answer_passages": 0}
    examples = []
    for row in rows:
        if counts["rows_scanned"] >= limit:
            break
        counts["rows_scanned"] += 1
        answers = _answers(row.get("answers"))
        passages = row.get("passages") or {}
        if not answers or not isinstance(passages, dict):
            continue
        texts = passages.get("passage_text") or []
        selected = passages.get("is_selected") or []
        hits = [index for index, raw in enumerate(texts)
                if _contains_answer(normalize_text(raw), answers)]
        if not hits:
            continue
        counts["answerable_rows"] += 1
        if len(hits) == 1:
            counts["one_answer_passage"] += 1
        else:
            counts["two_or_more_answer_passages"] += 1
            selected_hits = [index for index in hits if index < len(selected) and selected[index]]
            if len(selected_hits) >= 2:
                counts["two_or_more_selected_answer_passages"] += 1
            if len(examples) < 10:
                examples.append({"source_query_id": row.get("query_id"),
                                 "question": normalize_text(row.get("query", "")),
                                 "answer_passage_indices": hits,
                                 "selected_answer_passage_indices": selected_hits})
    return {"dataset": "microsoft/ms_marco", "config": "v1.1", "counts": counts,
            "examples": examples,
            "note": "Lexical answer presence is not factual agreement or independent source provenance."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results/multisupport_audit.json")
    args = parser.parse_args()
    from datasets import load_dataset

    root = Path(__file__).resolve().parent / "benchmark"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    revision = manifest["sources"]["ms_marco"]["revision"]
    rows = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True, revision=revision)
    report = audit(rows, limit=args.limit)
    report["revision"] = revision
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
