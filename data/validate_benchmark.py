"""Validate frozen benchmark hashes, support links, and split isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.build_benchmark import _contains_answer, _tokens


SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_benchmark(root: Path) -> dict[str, int]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_files = manifest["file_sha256"]
    actual_files = {str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*.jsonl")}
    if actual_files != set(expected_files):
        raise ValueError(f"Benchmark files changed: missing={sorted(set(expected_files)-actual_files)}, extra={sorted(actual_files-set(expected_files))}")
    for name, expected_hash in expected_files.items():
        actual_hash = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Benchmark checksum mismatch: {name}")

    all_ids: set[int] = set()
    all_texts: set[tuple[str, ...]] = set()
    all_questions: set[tuple[str, ...]] = set()
    counts: dict[str, int] = {}
    test_docs: dict[int, dict] = {}
    for name in SPLITS:
        docs = _read_jsonl(root / name / "corpus.jsonl")
        queries = _read_jsonl(root / name / "queries.jsonl")
        if len(docs) != len(queries) or len(docs) != manifest["counts"][name]:
            raise ValueError(f"Unexpected document/query count in {name}")
        by_id = {int(doc["doc_id"]): doc for doc in docs}
        if len(by_id) != len(docs):
            raise ValueError(f"Duplicate document ID within {name}")
        for doc in docs:
            doc_id = int(doc["doc_id"])
            text_key = _tokens(doc["text"])
            if doc_id in all_ids or text_key in all_texts:
                raise ValueError(f"Document leaks across benchmark splits: {doc_id}")
            all_ids.add(doc_id)
            all_texts.add(text_key)
        for query in queries:
            question_key = _tokens(query["question"])
            if question_key in all_questions:
                raise ValueError(f"Question leaks across benchmark splits: {query['question']}")
            all_questions.add(question_key)
            support_ids = query["support_doc_ids"]
            if len(support_ids) != 1 or support_ids[0] not in by_id:
                raise ValueError(f"Missing support document for {query['qid']}")
            if not _contains_answer(by_id[support_ids[0]]["text"], query["answer_aliases"]):
                raise ValueError(f"Answer alias absent from support for {query['qid']}")
        counts[name] = len(docs)
        if name == "test":
            test_docs = by_id

    challenge = _read_jsonl(root / "nq_challenge.jsonl")
    if len(challenge) != manifest["counts"]["nq_challenge"]:
        raise ValueError("NQ challenge count differs from manifest")
    alias_present = 0
    for item in challenge:
        support_ids = item["support_doc_ids"]
        if any(doc_id not in test_docs for doc_id in support_ids):
            raise ValueError(f"NQ challenge references a non-test document: {item['qid']}")
        verified_ids = [
            doc_id for doc_id, doc in test_docs.items()
            if _contains_answer(doc["text"], item["answer_aliases"])
        ]
        if support_ids != verified_ids or item["answer_alias_present_in_test_corpus"] != bool(verified_ids):
            raise ValueError(f"NQ alias-presence metadata is inconsistent: {item['qid']}")
        alias_present += bool(verified_ids)
    if alias_present != manifest["counts"]["nq_alias_present"]:
        raise ValueError("NQ alias-presence count differs from manifest")
    counts["nq_challenge"] = len(challenge)
    counts["nq_alias_present"] = alias_present
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parent / "benchmark")
    args = parser.parse_args()
    print(json.dumps(validate_benchmark(args.root), indent=2))
    print("Benchmark hashes, support links, and split isolation: OK")


if __name__ == "__main__":
    main()
