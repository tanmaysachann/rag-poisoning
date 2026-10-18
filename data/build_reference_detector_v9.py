"""Build question-disjoint, source-labelled training data from published NQ archives.

The archived replay's 1,150 logged questions are excluded completely. The
remaining 2,302 questions supply one naive attack, a judged reference and five
lexically hard, genuinely clean passages each. Labels describe provenance,
not a claim that every clean negative is factually unsupported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from sklearn.feature_extraction.text import TfidfVectorizer

from data.build_external_rag_v7 import read_jsonl


BEIR_MD5 = "d4d3d2e48787a744b6f6e691ff534307"
ATTACK_SHA256 = "ad09a35700b93113e3f94e9888a0a289d348b2fd650d6d78b99c480654d0c9e3"
LABELS = ("reference", "clean_other", "poison")


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-archive", type=Path,
                        default=Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip"))
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/reference_detector_v9"))
    parser.add_argument("--hard-negatives", type=int, default=5)
    args = parser.parse_args()
    if args.hard_negatives < 1:
        raise ValueError("At least one clean negative is required")
    if digest(args.attack_archive, "sha256") != ATTACK_SHA256:
        raise ValueError("Unexpected attack archive")
    if digest(args.beir_archive, "md5") != BEIR_MD5:
        raise ValueError("Unexpected BEIR archive")

    with ZipFile(args.attack_archive) as attack, ZipFile(args.beir_archive) as beir:
        naive = read_jsonl(attack, "data/nq-incorrect-answers-poisoned-docs.jsonl")
        logged = {name.rsplit("/", 1)[-1][:-5] for name in attack.namelist()
                  if name.startswith("results/noise/test") and name.endswith(".json")}
        train_ids = sorted(naive.keys() - logged)
        if len(train_ids) != 2302 or len(logged) != 1150:
            raise ValueError("Published question inventory changed")

        query_lookup = {row["_id"]: row["text"] for row in map(
            json.loads, beir.open("nq/queries.jsonl")) if row["_id"] in train_ids}
        if set(query_lookup) != set(train_ids):
            raise ValueError("Training queries missing from BEIR")
        relevant: dict[str, set[str]] = {qid: set() for qid in train_ids}
        with io.TextIOWrapper(beir.open("nq/qrels/test.tsv"), encoding="utf-8") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                if row["query-id"] in relevant and int(row["score"]) > 0:
                    relevant[row["query-id"]].add(row["corpus-id"])
        if any(not ids for ids in relevant.values()):
            raise ValueError("A training question has no judged reference")

        selected = {qid: sorted(ids)[0] for qid, ids in relevant.items()}
        needed = set(selected.values())
        passages: dict[str, str] = {}
        with beir.open("nq/corpus.jsonl") as source:
            for line in source:
                row = json.loads(line)
                if row["_id"] in needed:
                    passages[row["_id"]] = "\n".join(
                        part for part in (row.get("title", ""), row["text"]) if part)
        if set(passages) != needed:
            raise ValueError("A judged source passage is missing")

    # All negatives are official clean BEIR passages. The nearest references
    # from other questions are difficult on-topic negatives, not attack labels.
    pool_ids = sorted(passages)
    pool_text = [passages[doc_id] for doc_id in pool_ids]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_features=250000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(pool_text)
    query_matrix = vectorizer.transform([query_lookup[qid] for qid in train_ids])
    similarity = query_matrix @ matrix.T
    output = args.output / "train.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for row_index, qid in enumerate(train_ids):
            query = query_lookup[qid]
            ref_id = selected[qid]
            reference = passages[ref_id]
            poison = naive[qid]["poisoned_doc"]
            if not poison.strip():
                raise ValueError(f"Empty poison for {qid}")
            ranking = sorted(zip(similarity[row_index].indices,
                                 similarity[row_index].data),
                             key=lambda item: (-item[1], pool_ids[item[0]]))
            negative_ids = [pool_ids[index] for index, _ in ranking
                            if pool_ids[index] not in relevant[qid]][:args.hard_negatives]
            if len(negative_ids) != args.hard_negatives:
                raise ValueError(f"Not enough clean negatives for {qid}")
            rows = [(ref_id, reference, "reference"),
                    *[(doc_id, passages[doc_id], "clean_other") for doc_id in negative_ids],
                    (f"poisoned-naive-q:{qid}", poison, "poison")]
            for candidate_id, candidate, label in rows:
                sink.write(json.dumps({"query_id": qid, "query": query,
                                       "reference_id": ref_id,
                                       "reference": reference,
                                       "candidate_id": candidate_id,
                                       "candidate": candidate,
                                       "label": label}, ensure_ascii=False) + "\n")
                counts[label] += 1
    manifest = {"source": "Architecture Matters attack archive and BEIR NQ",
                "attack_sha256": ATTACK_SHA256, "beir_md5": BEIR_MD5,
                "excluded_logged_questions": len(logged),
                "training_questions": len(train_ids), "label_counts": dict(counts),
                "label_meaning": {"reference": "BEIR relevance-judged clean passage",
                                  "clean_other": "lexically hard BEIR clean passage; factual relation unknown",
                                  "poison": "published naive attack passage"},
                "train_sha256": digest(output, "sha256"),
                "training_query_ids_sha256": hashlib.sha256(
                    "\n".join(train_ids).encode()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
