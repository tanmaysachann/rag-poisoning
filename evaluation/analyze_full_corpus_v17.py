"""Post-evaluation error funnel; v17 evaluation questions are spent after this."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from data.build_full_corpus_v17 import ROOT, normalized
from data.build_trusted_snapshot_v12 import digest
from evaluation.evaluate_full_corpus_v17 import answer_match, split_rows
from evaluation.evaluate_trusted_snapshot_v12 import policy_prediction


LOCAL = ROOT / "data" / "full_corpus_v17"
SPLIT = ROOT / "research_protocols" / "full_corpus_v17_matched_split.json"
RESULT = ROOT / "results" / "full_corpus_v17.json"
OUTPUT = ROOT / "results" / "full_corpus_v17_diagnostics.json"


def has_alias(text: str, aliases: list[str]) -> bool:
    haystack = f" {normalized(text)} "
    return any(f" {normalized(alias)} " in haystack for alias in aliases
               if normalized(alias))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite post-evaluation diagnostic: {OUTPUT}")
    split, rows = split_rows(SPLIT)
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    if report["split_sha256"] != digest(SPLIT, "sha256"):
        raise ValueError("Scored split changed")
    predictions_path = LOCAL / "predictions.jsonl"
    if report["predictions_sha256"] != digest(predictions_path, "sha256"):
        raise ValueError("Frozen predictions changed")
    answers_path = LOCAL / "NQ-open.dev.jsonl"
    if split["nq_open_answers_sha256"] != digest(answers_path, "sha256"):
        raise ValueError("Official answers changed")
    aliases_by_question = defaultdict(list)
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        aliases_by_question[normalized(row["question"])].append(row["answer"])
    predictions = {row["qid"]: row for row in
                   map(json.loads, predictions_path.read_text(encoding="utf-8").splitlines())}
    retrievals_path = LOCAL / "retrievals.jsonl"
    retrievals = {row["qid"]: row["pool"] for row in
                  map(json.loads, retrievals_path.read_text(encoding="utf-8").splitlines())}
    counts = Counter()
    for question in split["evaluation"]:
        qid = question["qid"]
        matches = aliases_by_question[normalized(question["question"])]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous answer: {qid}")
        aliases = matches[0]
        pool = retrievals[qid]
        by_id = {row["doc_id"]: row for row in pool}
        counts["alias_in_lexical_top_100"] += any(has_alias(row["text"], aliases)
                                                   for row in pool)
        candidate = predictions[qid]
        for ranking, ranked in candidate["rankings"].items():
            counts[f"alias_in_{ranking}_top_10"] += any(
                has_alias(by_id[row["doc_id"]]["text"], aliases) for row in ranked)
            counts[f"reader_exact_candidate_in_{ranking}_top_10"] += any(
                answer_match(row["answer"], aliases, substring=False) for row in ranked)
            counts[f"reader_substring_candidate_in_{ranking}_top_10"] += any(
                answer_match(row["answer"], aliases, substring=True) for row in ranked)
        chosen = policy_prediction(candidate["rankings"][report["selected_ranking"]],
                                   report["selected_policy"])
        emitted = chosen["answer"] if chosen["score"] >= report["calibrated_score_cutoff"] else ""
        counts["selected_policy_exact"] += answer_match(emitted, aliases,
                                                          substring=False)
        counts["selected_policy_abstained"] += not bool(emitted)
    result = {
        "scope": "post-evaluation diagnostic; these 500 questions are now spent",
        "protocol": "research_protocols/full_corpus_v17.md",
        "parent_result_sha256": digest(RESULT, "sha256"),
        "retrievals_sha256": digest(retrievals_path, "sha256"),
        "questions": len(split["evaluation"]),
        "counts": dict(sorted(counts.items())),
        "interpretation": "Candidate presence is an oracle ceiling, not achievable answer accuracy; alias matching may miss paraphrases or alternate valid answers.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
