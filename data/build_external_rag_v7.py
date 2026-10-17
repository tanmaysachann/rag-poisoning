"""Recover authentic BEIR passages for an independent archived RAG attack run.

The attack archive stores retrieved IDs but omits original passage text. This
builder joins those IDs to the checksum-verified BEIR NQ corpus. It never uses
answer strings to select references or passages.
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


FAMILIES = ("naive", "corruptrag_ak")


def read_jsonl(archive: ZipFile, name: str) -> dict[str, dict]:
    with archive.open(name) as stream:
        return {row["query_id"]: row for row in map(json.loads, stream)}


def split_of(query_id: str) -> str:
    bucket = int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16) % 4
    return "calibration" if bucket == 0 else "test"


def source_text(doc_id: str, passages: dict[str, dict],
                naive: dict[str, dict], corruptrag: dict[str, dict]) -> str:
    if doc_id.startswith("poisoned-naive-q:"):
        return naive[doc_id.split(":", 1)[1]]["poisoned_doc"]
    if doc_id.startswith("poisoned-corruptrag-ak-q:"):
        return corruptrag[doc_id.split(":", 1)[1]]["corruptrag_ak_text"]
    doc = passages[doc_id]
    return "\n".join(part for part in (doc["title"], doc["text"]) if part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-archive", type=Path,
                        default=Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip"))
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/external_rag_v7"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.attack_archive) as attack, ZipFile(args.beir_archive) as beir:
        naive = read_jsonl(attack, "data/nq-incorrect-answers-poisoned-docs.jsonl")
        corruptrag = read_jsonl(attack, "data/nq-corruptrag-ak-poisoned-docs.jsonl")
        names = set(attack.namelist())
        query_ids = sorted(
            path.rsplit("/", 1)[-1][:-5]
            for path in names if path.startswith("results/noise/test") and path.endswith(".json")
        )
        records: list[dict] = []
        needed_ids: set[str] = set()
        noise_counts: Counter[str] = Counter()
        for query_id in query_ids:
            noise = json.loads(attack.read(f"results/noise/{query_id}.json"))
            noise_counts[noise["noise_type"] if noise["is_noise"] else "none"] += 1
            if noise["noise_type"] == "full":
                continue
            logs = {
                family: json.loads(attack.read(
                    f"results/experiments/vanilla_{family}/{query_id}.json"
                )) for family in ("clean", *FAMILIES)
            }
            if any(log["question_id"] != query_id for log in logs.values()):
                raise ValueError(f"Mismatched question: {query_id}")
            for log in logs.values():
                needed_ids.update(x for x in log["retrieved_doc_ids"] if x.startswith("doc"))
            records.append({"query_id": query_id, "split": split_of(query_id),
                            "query": logs["clean"]["question_text"],
                            "correct_answer": logs["clean"]["correct_answer"],
                            "target_answer": logs["naive"]["target_answer"],
                            "noise_type": noise["noise_type"] if noise["is_noise"] else "none",
                            "logs": logs})

        relevant: dict[str, set[str]] = {}
        with io.TextIOWrapper(beir.open("nq/qrels/test.tsv"), encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if int(row["score"]) > 0:
                    relevant.setdefault(row["query-id"], set()).add(row["corpus-id"])
        queries = {row["_id"]: row["text"] for row in map(
            json.loads, beir.open("nq/queries.jsonl")
        )}
        for record in records:
            if record["query"] != queries[record["query_id"]]:
                raise ValueError(f"BEIR query text changed: {record['query_id']}")
        passages: dict[str, dict] = {}
        with beir.open("nq/corpus.jsonl") as stream:
            for line in stream:
                row = json.loads(line)
                if row["_id"] in needed_ids:
                    passages[row["_id"]] = row
        if missing := needed_ids - passages.keys():
            raise ValueError(f"Missing {len(missing)} BEIR passages: {sorted(missing)[:3]}")

        counts = Counter()
        outputs = {split: (args.output / f"{split}.jsonl").open("w", encoding="utf-8")
                   for split in ("calibration", "test")}
        try:
            for record in records:
                logs = record.pop("logs")
                query_id = record["query_id"]
                clean_ids = logs["clean"]["retrieved_doc_ids"]
                reference_id = next((doc_id for doc_id in clean_ids
                                     if doc_id in relevant.get(query_id, set())), None)
                if reference_id is None:
                    counts["no_reference"] += 1
                else:
                    counts["with_reference"] += 1
                record["reference_id"] = reference_id
                record["reference_text"] = (
                    source_text(reference_id, passages, naive, corruptrag)
                    if reference_id else None
                )
                record["retrievals"] = {}
                for family, log in logs.items():
                    ids = log["retrieved_doc_ids"]
                    own_id = (f"poisoned-naive-q:{query_id}" if family == "naive"
                              else f"poisoned-corruptrag-ak-q:{query_id}")
                    own_rank = ids.index(own_id) + 1 if own_id in ids else None
                    record["retrievals"][family] = {
                        "candidates": [{"id": doc_id, "text": source_text(
                            doc_id, passages, naive, corruptrag
                        )} for doc_id in ids],
                        "own_poison_retrieved": own_rank is not None,
                        "own_poison_rank": own_rank,
                        "archived_any_poison_retrieved": log["poison_retrieved"],
                    }
                    if family != "clean":
                        counts[f"{family}_own_poison_retrieved"] += own_rank is not None
                outputs[record["split"]].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[record["split"]] += 1
        finally:
            for handle in outputs.values():
                handle.close()
    manifest = {
        "source": "Architecture Matters archived retrieval logs + official BEIR NQ corpus",
        "beir_md5": hashlib.md5(args.beir_archive.read_bytes()).hexdigest(),
        "attack_zip_sha256": hashlib.sha256(args.attack_archive.read_bytes()).hexdigest(),
        "noise_counts": dict(noise_counts), "case_counts": dict(counts),
        "passages_recovered": len(passages),
        "split_rule": "SHA256(query_id) first 32 bits modulo 4; 0=calibration, else=test",
        "file_sha256": {f"{split}.jsonl": hashlib.sha256(
            (args.output / f"{split}.jsonl").read_bytes()
        ).hexdigest() for split in outputs},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
