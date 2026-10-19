"""Freeze new BEIR NQ question IDs before reading NQ-Open answer labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

from data.build_trusted_snapshot_v12 import BEIR_MD5, digest


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "v17-full-corpus:"


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def verified_jsonl(path: Path, expected: str) -> list[dict]:
    if digest(path, "sha256") != expected:
        raise ValueError(f"Input hash changed: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def exclusions() -> tuple[set[str], set[str], dict]:
    excluded_ids: set[str] = set()
    excluded_text: set[str] = set()
    input_hashes: dict[str, str] = {}
    external = ROOT / "data" / "external_rag_v7"
    external_manifest = json.loads((external / "manifest.json").read_text(encoding="utf-8"))
    for name in ("calibration.jsonl", "test.jsonl"):
        path = external / name
        expected = external_manifest["file_sha256"][name]
        input_hashes[f"external/{name}"] = expected
        for row in verified_jsonl(path, expected):
            excluded_ids.add(str(row["query_id"]))
            excluded_text.add(normalized(row["query"]))
    arena = ROOT / "data" / "poisonarena_v9"
    arena_manifest = json.loads((arena / "manifest.json").read_text(encoding="utf-8"))
    arena_path = arena / "nq_queries.json"
    arena_hash = arena_manifest["files"]["nq_queries"]["sha256"]
    input_hashes["poisonarena/nq_queries.json"] = arena_hash
    attack_path = arena / "nq_arena.json"
    attack_hash = arena_manifest["files"]["nq_arena"]["sha256"]
    if digest(attack_path, "sha256") != attack_hash:
        raise ValueError("PoisonArena attack file changed")
    input_hashes["poisonarena/nq_arena.json"] = attack_hash
    attacked_ids = set(json.loads(attack_path.read_text(encoding="utf-8")))
    for row in verified_jsonl(arena_path, arena_hash):
        if row["_id"] in attacked_ids:
            excluded_ids.add(str(row["_id"]))
            excluded_text.add(normalized(row["text"]))
    challenge_path = ROOT / "data" / "benchmark_v3" / "nq_challenge.jsonl"
    benchmark = json.loads((challenge_path.parent / "manifest.json").read_text(encoding="utf-8"))
    challenge_hash = benchmark["file_sha256"]["nq_challenge.jsonl"]
    input_hashes["benchmark_v3/nq_challenge.jsonl"] = challenge_hash
    for row in verified_jsonl(challenge_path, challenge_hash):
        excluded_text.add(normalized(row["question"]))
    return excluded_ids, excluded_text, input_hashes


def build(archive: Path, answers_file: Path, output: Path, *, match_questions: bool) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split: {output}")
    if digest(archive, "md5") != BEIR_MD5:
        raise ValueError("Official BEIR NQ archive hash changed")
    if not answers_file.exists():
        raise FileNotFoundError(answers_file)
    excluded_ids, excluded_text, hashes = exclusions()
    with ZipFile(archive) as source:
        query_bytes = source.read("nq/queries.jsonl")
        qrel_bytes = source.read("nq/qrels/test.tsv")
    judged = {row["query-id"] for row in csv.DictReader(
        io.StringIO(qrel_bytes.decode("utf-8")), delimiter="\t")
        if int(row["score"]) > 0}
    all_questions = [json.loads(line) for line in query_bytes.splitlines() if line.strip()]
    eligible = [row for row in all_questions
                if row["_id"] in judged and row["_id"] not in excluded_ids
                and normalized(row["text"]) not in excluded_text]
    if match_questions:
        question_counts: dict[str, int] = {}
        for line in answers_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                key = normalized(json.loads(line)["question"])
                question_counts[key] = question_counts.get(key, 0) + 1
        eligible = [row for row in eligible
                    if question_counts.get(normalized(row["text"])) == 1]
    if len(eligible) < 700:
        raise ValueError(f"Only {len(eligible)} eligible new questions")
    eligible.sort(key=lambda row: (hashlib.sha256(
        (PREFIX + row["_id"]).encode("utf-8")).hexdigest(), row["_id"]))
    selected = eligible[:700]
    split = {
        "protocol": "research_protocols/full_corpus_v17.md",
        "source": "BEIR Natural Questions queries and official NQ-Open dev answers",
        "beir_archive_md5": BEIR_MD5,
        "beir_archive_sha256": digest(archive, "sha256"),
        "beir_queries_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "beir_qrels_sha256": hashlib.sha256(qrel_bytes).hexdigest(),
        "nq_open_answers_sha256": digest(answers_file, "sha256"),
        "exclusion_hashes": hashes,
        "excluded_ids": len(excluded_ids),
        "excluded_question_texts": len(excluded_text),
        "eligible_questions": len(eligible),
        "selection": "question-only NQ-Open join, then SHA256('v17-full-corpus:' + BEIR qid); 200 calibration, next 500 evaluation" if match_questions else "SHA256('v17-full-corpus:' + BEIR qid); 200 calibration, next 500 evaluation",
        "calibration": [{"qid": row["_id"], "question": row["text"]} for row in selected[:200]],
        "evaluation": [{"qid": row["_id"], "question": row["text"]} for row in selected[200:]],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"eligible": len(eligible), "calibration": 200, "evaluation": 500,
            "split_sha256": digest(output, "sha256"),
            "nq_open_answers_sha256": split["nq_open_answers_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--answers-file", type=Path,
                        default=ROOT / "data" / "full_corpus_v17" / "NQ-open.dev.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research_protocols" / "full_corpus_v17_matched_split.json")
    parser.add_argument("--unmatched", action="store_true",
                        help="Reproduce the original first split without the question-only join")
    args = parser.parse_args()
    print(json.dumps(build(args.beir_archive, args.answers_file, args.output,
                           match_questions=not args.unmatched), indent=2))


if __name__ == "__main__":
    main()
