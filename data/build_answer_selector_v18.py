"""Freeze 500 new NQ questions disjoint from the spent v17 questions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from data.build_full_corpus_v17 import ROOT, exclusions, normalized
from data.build_trusted_snapshot_v12 import BEIR_MD5, digest


def build(archive: Path, answers_file: Path, prior_split: Path,
          output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen v18 split: {output}")
    if digest(archive, "md5") != BEIR_MD5:
        raise ValueError("BEIR NQ archive changed")
    prior = json.loads(prior_split.read_text(encoding="utf-8"))
    if digest(answers_file, "sha256") != prior["nq_open_answers_sha256"]:
        raise ValueError("Official answer source changed")
    used = prior["calibration"] + prior["evaluation"]
    used_ids = {row["qid"] for row in used}
    used_text = {normalized(row["question"]) for row in used}
    excluded_ids, excluded_text, exclusion_hashes = exclusions()
    question_counts = Counter(normalized(json.loads(line)["question"])
                              for line in answers_file.read_text(encoding="utf-8").splitlines()
                              if line.strip())
    with ZipFile(archive) as source:
        queries_bytes = source.read("nq/queries.jsonl")
        qrels_bytes = source.read("nq/qrels/test.tsv")
    judged = {row["query-id"] for row in csv.DictReader(
        io.StringIO(qrels_bytes.decode("utf-8")), delimiter="\t")
        if int(row["score"]) > 0}
    eligible = []
    for line in queries_bytes.splitlines():
        row = json.loads(line)
        qid, question = row["_id"], normalized(row["text"])
        if (qid in judged and qid not in excluded_ids and qid not in used_ids
                and question not in excluded_text and question not in used_text
                and question_counts[question] == 1):
            eligible.append(row)
    eligible.sort(key=lambda row: (hashlib.sha256(
        ("v18-full-corpus:" + row["_id"]).encode("utf-8")).hexdigest(),
                                   row["_id"]))
    if len(eligible) < 500:
        raise ValueError(f"Only {len(eligible)} fresh questions")
    selected = eligible[:500]
    frozen = {
        "protocol": "research_protocols/answer_selector_v18.md",
        "source": "BEIR NQ queries; NQ-Open dev question-only join",
        "prior_v17_split_sha256": digest(prior_split, "sha256"),
        "beir_archive_md5": BEIR_MD5,
        "beir_archive_sha256": digest(archive, "sha256"),
        "beir_queries_sha256": hashlib.sha256(queries_bytes).hexdigest(),
        "beir_qrels_sha256": hashlib.sha256(qrels_bytes).hexdigest(),
        "nq_open_answers_sha256": digest(answers_file, "sha256"),
        "exclusion_hashes": exclusion_hashes,
        "remaining_eligible": len(eligible),
        "selection": "SHA256('v18-full-corpus:' + BEIR qid), first 500 after prior exclusions",
        "calibration": [],
        "evaluation": [{"qid": row["_id"], "question": row["text"]}
                       for row in selected],
    }
    output.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return {"remaining_eligible": len(eligible), "evaluation": 500,
            "split_sha256": digest(output, "sha256")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--answers-file", type=Path,
                        default=ROOT / "data" / "full_corpus_v17" / "NQ-open.dev.jsonl")
    parser.add_argument("--prior-split", type=Path,
                        default=ROOT / "research_protocols" / "full_corpus_v17_matched_split.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research_protocols" / "answer_selector_v18_split.json")
    args = parser.parse_args()
    print(json.dumps(build(args.beir_archive, args.answers_file,
                           args.prior_split, args.output), indent=2))


if __name__ == "__main__":
    main()
