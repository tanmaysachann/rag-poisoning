"""Meaningful split and provenance checks for the research benchmark builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.build_benchmark import build_benchmark, collect_msmarco_pairs
from data.validate_benchmark import validate_benchmark


def _msmarco_row(index: int) -> dict:
    answer = f"Answer{index}"
    return {
        "query_id": index,
        "query": f"What is fact number {index}?",
        "answers": [answer],
        "passages": {
            "passage_text": [
                f"This is a distractor passage about fact number {index} with no correct answer and enough extra context for the length filter.",
                f"The verified reference for fact number {index} is {answer}. This longer selected passage includes enough contextual text to qualify as an answer-supported document.",
            ],
            "is_selected": [0, 1],
        },
    }


class BenchmarkBuilderTests(unittest.TestCase):
    def test_selects_only_supported_passages_and_disjoint_splits(self) -> None:
        rows = [_msmarco_row(i) for i in range(12)]
        rows.insert(3, rows[0])  # duplicate query and source must be ignored
        pairs, scanned = collect_msmarco_pairs(rows, target_pairs=12)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(scanned, 13)
        self.assertTrue(all("Answer" in item["text"] for item in pairs))

        nq_rows = [
            {"question": "Who discovered a remote fact?", "answer": ["AbsentAlias"]},
            {"question": "What is the name of a possible answer?", "answer": ["Answer0"]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark"
            manifest = build_benchmark(
                rows, nq_rows, output, target_pairs=12, nq_limit=2, seed=42,
                msmarco_revision="fixed-ms-revision", nq_revision="fixed-nq-revision",
            )
            self.assertEqual(sum(manifest["counts"][name] for name in ("train", "validation", "test")), 12)
            self.assertEqual(manifest["sources"]["ms_marco"]["revision"], "fixed-ms-revision")
            split_ids: list[set[int]] = []
            split_queries: list[set[str]] = []
            for name in ("train", "validation", "test"):
                docs = [json.loads(line) for line in (output / name / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
                queries = [json.loads(line) for line in (output / name / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(docs), len(queries))
                self.assertEqual({doc["doc_id"] for doc in docs}, {q["support_doc_ids"][0] for q in queries})
                split_ids.append({doc["doc_id"] for doc in docs})
                split_queries.append({q["question"] for q in queries})
            for left in range(3):
                for right in range(left + 1, 3):
                    self.assertFalse(split_ids[left] & split_ids[right])
                    self.assertFalse(split_queries[left] & split_queries[right])

            challenge = [json.loads(line) for line in (output / "nq_challenge.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertFalse(challenge[0]["answer_alias_present_in_test_corpus"])
            self.assertEqual(manifest["counts"]["nq_challenge"], 2)
            self.assertEqual(validate_benchmark(output)["nq_challenge"], 2)
            with self.assertRaises(FileExistsError):
                build_benchmark(
                    rows, nq_rows, output, target_pairs=12, nq_limit=2, seed=42,
                    msmarco_revision="fixed-ms-revision", nq_revision="fixed-nq-revision",
                )
            with (output / "train" / "corpus.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_benchmark(output)


if __name__ == "__main__":
    unittest.main()
