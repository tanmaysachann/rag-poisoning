"""Post-hoc validation diagnostic: use the peer answer on pair disagreement."""

from __future__ import annotations

import json

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from generation.sentence_ranker import _contains_alias


def choose(case: dict, branch: str, threshold: float) -> tuple[str, int | None]:
    result = case[branch]
    if (result["answer_cosine"] is not None
            and result["answer_cosine"] < threshold
            and result["peer_doc_id"] is not None
            and result["peer_answer"]):
        return result["peer_answer"], result["peer_doc_id"]
    return result["answer"], result["source_doc_id"]


def summarize() -> dict:
    validate(ROOT_DIR / "data/benchmark_v2", ROOT_DIR / "data/benchmark")
    queries = {row["qid"]: row for row in read_jsonl(ROOT_DIR / "data/benchmark_v2/validation/queries.jsonl")}
    cases = read_jsonl(ROOT_DIR / "results/paired_gate_v2_validation_full_context.cases.jsonl")
    if len(cases) != 30 or {row["qid"] for row in cases} != set(queries):
        raise ValueError("Expected the corrected 30-case v2 validation run")
    rows = []
    for threshold in (0.45, 0.50, 0.55, 0.60, 0.65):
        clean_alias = attack_success = peer_selected_clean = peer_selected_attacked = 0
        new_attacks = removed_attacks = 0
        new_attack_qids, removed_attack_qids = [], []
        for case in cases:
            clean_answer, clean_source = choose(case, "clean", threshold)
            attacked_answer, attacked_source = choose(case, "attacked", threshold)
            clean_alias += _contains_alias(clean_answer, queries[case["qid"]]["answer_aliases"])
            success = attacked_source == case["attack_doc_id"] and _contains_alias(attacked_answer, [case["wrong_answer"]])
            attack_success += success
            new_attacks += success and not case["attack_success_before_gate"]
            removed_attacks += case["attack_success_before_gate"] and not success
            if success and not case["attack_success_before_gate"]:
                new_attack_qids.append(case["qid"])
            if case["attack_success_before_gate"] and not success:
                removed_attack_qids.append(case["qid"])
            peer_selected_clean += clean_source == case["clean"]["peer_doc_id"] and clean_source is not None
            peer_selected_attacked += attacked_source == case["attacked"]["peer_doc_id"] and attacked_source is not None
        rows.append({
            "threshold": threshold, "clean_alias_matches": clean_alias,
            "attacked_wrong_answer_successes": attack_success,
            "new_attack_successes": new_attacks, "removed_attack_successes": removed_attacks,
            "new_attack_qids": new_attack_qids, "removed_attack_qids": removed_attack_qids,
            "clean_peer_selections": peer_selected_clean,
            "attacked_peer_selections": peer_selected_attacked,
        })
    return {
        "split": "validation", "cases": len(cases),
        "baseline_clean_alias_matches": sum(case["clean_alias_before_gate"] for case in cases),
        "baseline_attack_successes": sum(case["attack_success_before_gate"] for case in cases),
        "thresholds": rows,
        "note": "Post-hoc validation probe only. Thresholds were inspected on validation, and MS MARCO pair labels do not establish independent source origins. No serving decision follows from this table.",
    }


def main() -> None:
    output = summarize()
    path = ROOT_DIR / "results/v2_peer_failover_validation_probe.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
