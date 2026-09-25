"""Validate the frozen two-support benchmark without evaluating its test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.build_benchmark import _contains_answer, _tokens


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(root: Path, original: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 2 or manifest["passages_per_question"] != 2:
        raise ValueError("Wrong two-support benchmark schema")
    old_docs = [row for split in ("train", "validation", "test")
                for row in read_jsonl(original / split / "corpus.jsonl")]
    old_ids = {row["source_query_id"] for row in old_docs}
    old_passages = {_tokens(row["text"]) for row in old_docs}
    old_questions = {_tokens(row["question"]) for split in ("train", "validation", "test")
                     for row in read_jsonl(original / split / "queries.jsonl")}
    seen_qids, seen_source_ids, seen_doc_ids, seen_passages, seen_questions = set(), set(), set(), set(), set()
    counts = {}
    for split in ("train", "validation", "test"):
        corpus_path, queries_path = root / split / "corpus.jsonl", root / split / "queries.jsonl"
        corpus, queries = read_jsonl(corpus_path), read_jsonl(queries_path)
        if len(corpus) != 2 * len(queries):
            raise ValueError(f"{split}: expected two passages per query")
        docs = {row["doc_id"]: row for row in corpus}
        if len(docs) != len(corpus):
            raise ValueError(f"{split}: duplicate document ID")
        for row in corpus:
            passage_key = _tokens(row["text"])
            if row["doc_id"] in seen_doc_ids or passage_key in seen_passages or passage_key in old_passages:
                raise ValueError("Document shared across benchmark splits")
            seen_doc_ids.add(row["doc_id"])
            seen_passages.add(passage_key)
        for query in queries:
            source_id = query["source_query_id"]
            question_key = _tokens(query["question"])
            if (query["qid"] in seen_qids or source_id in seen_source_ids or source_id in old_ids
                    or question_key in seen_questions or question_key in old_questions):
                raise ValueError("Query overlaps a split or the v1 benchmark")
            seen_qids.add(query["qid"])
            seen_source_ids.add(source_id)
            seen_questions.add(question_key)
            supports = query["support_doc_ids"]
            if len(supports) != 2 or len(set(supports)) != 2:
                raise ValueError("Query must have exactly two distinct support documents")
            for doc_id in supports:
                if doc_id not in docs or docs[doc_id]["source_query_id"] != source_id:
                    raise ValueError("Support document missing or belongs to another source query")
                if not _contains_answer(docs[doc_id]["text"], query["answer_aliases"]):
                    raise ValueError("Support document does not contain an answer alias")
        counts[split] = {"queries": len(queries), "passages": len(corpus)}
        if counts[split] != manifest["counts"][split]:
            raise ValueError("Manifest counts disagree with files")
    actual_files = {str(path.relative_to(root)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(root.rglob("*.jsonl"))}
    if actual_files != manifest["file_sha256"]:
        raise ValueError("Frozen file hashes disagree with manifest")
    return {"status": "verified", "counts": counts, "v1_source_ids_excluded": len(old_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "benchmark_v2")
    parser.add_argument("--original", type=Path, default=Path(__file__).resolve().parent / "benchmark")
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.original), indent=2))


if __name__ == "__main__":
    main()
