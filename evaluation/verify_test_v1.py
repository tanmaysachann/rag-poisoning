"""Audit the frozen v1 test snapshot and its paired case counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matches_saved_hash(path: Path, expected: str, *, portable_json: bool = False) -> bool:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    # Six result summaries were sealed with Windows CRLF bytes. Git checks out
    # those text files with LF on Linux (including Vercel). Permit only that
    # exact line-ending conversion; all other content must still match.
    return (portable_json and path.suffix == ".json" and b"\r" not in data
            and hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest() == expected)


def _check_queries(rows: list[dict], expected: set[str], label: str) -> None:
    qids = [row["qid"] for row in rows]
    if len(qids) != len(expected) or set(qids) != expected:
        raise ValueError(f"{label}: missing, duplicate, or unexpected query IDs")


def verify(root: Path = ROOT, *, check_output_hashes: bool = True) -> dict:
    manifest = json.loads((root / "TEST_EVIDENCE_V1.json").read_text(encoding="utf-8"))
    for kind in ("inputs_sha256", "outputs_sha256"):
        if kind == "outputs_sha256" and not check_output_hashes:
            continue
        for relative_path, expected in manifest[kind].items():
            path = root / relative_path
            if not path.is_file() or not _matches_saved_hash(
                path, expected, portable_json=kind == "outputs_sha256"
            ):
                raise ValueError(f"Missing or changed {kind}: {relative_path}")

    queries = _read_jsonl(root / "data/benchmark/test/queries.jsonl")
    expected_qids = {row["qid"] for row in queries}
    if len(expected_qids) != manifest["cases"]:
        raise ValueError("Frozen query count or uniqueness changed")

    clean_cases = _read_jsonl(root / "results/clean_answers_test_minilm_ranker_strict.cases.jsonl")
    _check_queries(clean_cases, expected_qids, "clean answer cases")
    counts = {"clean_alias_match": sum(bool(row["alias_match"]) for row in clean_cases)}
    clean_summary = json.loads((root / "results/clean_answers_test_minilm_ranker_strict.json").read_text(encoding="utf-8"))
    if clean_summary["cases"] != len(clean_cases) or round(clean_summary["alias_match_accuracy"] * len(clean_cases)) != counts["clean_alias_match"]:
        raise ValueError("Clean answer summary disagrees with cases")

    intervals = json.loads((root / "results/test_intervals_v1.json").read_text(encoding="utf-8"))
    if intervals["split"] != "test" or intervals["cases_per_profile"] != manifest["cases"]:
        raise ValueError("Interval report split or case count mismatch")
    for strategy in ("greedy", "stealth"):
        attack_cases = _read_jsonl(root / f"results/attack_baseline_test_{strategy}.jsonl")
        _check_queries(attack_cases, expected_qids, f"{strategy} attack cases")
        if any(row["split"] != "test" or row["strategy"] != strategy for row in attack_cases):
            raise ValueError(f"{strategy} attack metadata mismatch")
        rows = _read_jsonl(root / f"results/defense_test_{strategy}_accepted_ingest_minilm_ranker.cases.jsonl")
        _check_queries(rows, expected_qids, f"{strategy} defense cases")
        profile = intervals["profiles"][f"{strategy}_minilm_ranker"]
        summary = json.loads((root / f"results/defense_test_{strategy}_accepted_ingest_minilm_ranker.json").read_text(encoding="utf-8"))
        if summary["split"] != "test" or summary["strategy"] != strategy or summary["cases"] != len(rows):
            raise ValueError(f"{strategy} defense summary metadata mismatch")
        fields = {
            "undefended_success": "undefended_attack_success",
            "defended_success": "defended_attack_success",
            "quarantined": "attack_quarantined",
            "clean_alias_recovery": "clean_answer_recovered",
        }
        for count_name, field in fields.items():
            count = sum(bool(row[field]) for row in rows)
            counts[f"{strategy}_{count_name}"] = count
        if profile["undefended_attack_success"]["count"] != counts[f"{strategy}_undefended_success"]:
            raise ValueError(f"{strategy} interval before-count mismatch")
        if profile["defended_attack_success"]["count"] != counts[f"{strategy}_defended_success"]:
            raise ValueError(f"{strategy} interval after-count mismatch")
        if profile["attack_quarantine"]["count"] != counts[f"{strategy}_quarantined"]:
            raise ValueError(f"{strategy} interval quarantine-count mismatch")
        if profile["clean_alias_recovery"]["count"] != counts[f"{strategy}_clean_alias_recovery"]:
            raise ValueError(f"{strategy} interval recovery-count mismatch")
        for count_name, summary_name in (
            ("undefended_success", "undefended_attack_success_rate"),
            ("defended_success", "defended_attack_success_rate"),
            ("quarantined", "attack_quarantine_rate"),
            ("clean_alias_recovery", "clean_answer_recovery_rate"),
        ):
            if abs(summary[summary_name] - counts[f"{strategy}_{count_name}"] / len(rows)) > 1e-12:
                raise ValueError(f"{strategy} summary {summary_name} disagrees with cases")

    if counts != manifest["expected_counts"]:
        raise ValueError(f"Reported test counts changed: {counts}")
    return {"status": "verified", "output_hashes_checked": check_output_hashes,
            "cases": manifest["cases"], "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-output-hashes", action="store_true",
                        help="Use after rerunning timing-sensitive experiments; still check inputs and counts")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, check_output_hashes=not args.skip_output_hashes), indent=2))


if __name__ == "__main__":
    main()
