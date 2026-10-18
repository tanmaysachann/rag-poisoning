"""Score independent PoisonArena generators on questions absent from v9 training.

The threshold is imported unchanged from the v9 Architecture Matters clean
calibration. PoisonArena qrels supply an oracle-like reference, not live provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path
from zipfile import ZipFile

import joblib
import torch

from data.build_reference_detector_v9 import ATTACK_SHA256, BEIR_MD5, digest
from detect.reference_conditioned_v9 import ReferenceConditionedScorer
from detect.research_detector import ResearchDetector


def verified_json(path: Path, expected_sha256: str):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"PoisonArena file changed: {path}")
    return json.loads(data)


def normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def build_cases(directory: Path, attack_archive: Path, beir_archive: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    retrievals = verified_json(directory / "nq_retrievals.json",
                               manifest["files"]["nq_retrievals"]["sha256"])
    arena = verified_json(directory / "nq_arena.json",
                          manifest["files"]["nq_arena"]["sha256"])
    queries_data = (directory / "nq_queries.json").read_bytes()
    qrels_data = (directory / "nq_qrels.json").read_bytes()
    if hashlib.sha256(queries_data).hexdigest() != manifest["files"]["nq_queries"]["sha256"]:
        raise ValueError("PoisonArena query file changed")
    if hashlib.sha256(qrels_data).hexdigest() != manifest["files"]["nq_qrels"]["sha256"]:
        raise ValueError("PoisonArena qrels changed")
    with ZipFile(beir_archive) as beir:
        prior_questions = {normalized_question(row["text"]) for row in map(
            json.loads, beir.open("nq/queries.jsonl"))}
    families = {}
    counts = {}
    for family, info in manifest["files"].items():
        if family in {"nq_retrievals", "nq_arena", "nq_queries", "nq_qrels"}:
            continue
        source = verified_json(directory / f"{family}.json", info["sha256"])
        novel = [row for row in source if normalized_question(row["question"])
                 not in prior_questions]
        counts[family] = {"published_questions": len(source),
                          "question_text_overlap_excluded": len(source) - len(novel),
                          "new_questions": len(novel)}
        families[family] = novel
    query_ids = {row["id"] for rows in families.values() for row in rows}
    with ZipFile(beir_archive) as beir:
        query_lookup = {row["_id"]: row["text"] for row in map(
            json.loads, queries_data.splitlines())
                        if row["_id"] in query_ids}
        if set(query_lookup) != query_ids:
            raise ValueError("Independent questions missing in PoisonArena queries")
        relevant = {qid: set() for qid in query_ids}
        with io.StringIO(qrels_data.decode("utf-8")) as source:
            for row in csv.DictReader(source, delimiter="\t"):
                if row["query-id"] in relevant and int(row["score"]) > 0:
                    relevant[row["query-id"]].add(row["corpus-id"])
        if any(not docs for docs in relevant.values()):
            raise ValueError("Missing qrel on independent question")
        selected = {qid: sorted(ids)[0] for qid, ids in relevant.items()}
        needed = set(selected.values())
        for qid in query_ids:
            if qid not in retrievals:
                raise ValueError(f"Missing PoisonArena retrieval: {qid}")
            needed.update(list(retrievals[qid])[:10])
        passages = {}
        with beir.open("nq/corpus.jsonl") as source:
            for line in source:
                row = json.loads(line)
                if row["_id"] in needed:
                    passages[row["_id"]] = "\n".join(
                        part for part in (row.get("title", ""), row["text"]) if part)
        if set(passages) != needed:
            raise ValueError(f"Missing {len(needed - passages.keys())} clean passages")
    cases = {}
    for family, rows in families.items():
        for row in rows:
            qid = row["id"]
            if normalized_question(row["question"]) != normalized_question(query_lookup[qid]):
                raise ValueError(f"Question changed in PoisonArena: {qid}")
            case = cases.setdefault(qid, {"query_id": qid, "query": query_lookup[qid],
                                          "reference_id": selected[qid],
                                          "reference": passages[selected[qid]],
                                          "clean": [{"id": doc_id, "text": passages[doc_id]}
                                                    for doc_id in list(retrievals[qid])[:10]],
                                          "attacks": {}})
            case["attacks"][family] = [str(attack["context"]) for attack in row["adv_texts"]]
            if any(not text.strip() for text in case["attacks"][family]):
                raise ValueError(f"Empty attack: {family}/{qid}")
    counts["unique_new_questions"] = len(cases)
    counts["published_answer_in_qrel"] = sum(
        normalized_question(str(arena[qid]["correct answer"])) in
        normalized_question(case["reference"])
        for qid, case in cases.items())
    train_doc_ids = {json.loads(line)["candidate_id"] for line in
                     Path("data/reference_detector_v9/train.jsonl").open(encoding="utf-8")}
    counts["reference_doc_overlap_with_train"] = sum(
        case["reference_id"] in train_doc_ids for case in cases.values())
    return list(cases.values()), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/poisonarena_v9"))
    parser.add_argument("--attack-archive", type=Path,
                        default=Path("artifacts/vendor/rag-poisoning-architecture-bench-data.zip"))
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/reference_detector_v9"))
    parser.add_argument("--calibration-report", type=Path,
                        default=Path("results/reference_detector_v9_replay.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/reference_detector_v9_poisonarena.json"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--heldout-v10", action="store_true")
    parser.add_argument("--compare-v9", action="store_true")
    args = parser.parse_args()
    if digest(args.attack_archive, "sha256") != ATTACK_SHA256:
        raise ValueError("Architecture Matters archive changed")
    if digest(args.beir_archive, "md5") != BEIR_MD5:
        raise ValueError("BEIR archive changed")
    cases, counts = build_cases(args.data, args.attack_archive, args.beir_archive)
    if args.heldout_v10:
        cases = [case for case in cases if int(hashlib.sha256(
            ("v10-pilot:" + normalized_question(case["query"])).encode()
        ).hexdigest()[:8], 16) % 10 >= 7]
    threshold = json.loads(args.calibration_report.read_text(encoding="utf-8"))["threshold"]
    torch.set_num_threads(min(8, torch.get_num_threads()))
    scorer = ReferenceConditionedScorer(
        args.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    old = joblib.load("artifacts/research_detector_v3_hashing.joblib")
    if not isinstance(old, ResearchDetector):
        raise TypeError("Unexpected v3 baseline model")
    rows: dict[tuple[str, str], dict] = {}
    for case in cases:
        qid = case["query_id"]
        for doc in case["clean"]:
            rows[qid, "clean:" + doc["id"]] = {"query": case["query"],
                                                "reference": case["reference"],
                                                "candidate": doc["text"]}
        rows[qid, "reference"] = {"query": case["query"],
                                  "reference": case["reference"],
                                  "candidate": case["reference"]}
        for family, texts in case["attacks"].items():
            for index, candidate in enumerate(texts):
                rows[qid, f"{family}:{index}"] = {"query": case["query"],
                                                   "reference": case["reference"],
                                                   "candidate": candidate}
    scores, baseline = {}, {}
    keys = list(rows)
    started = time.perf_counter()
    for start in range(0, len(keys), 256):
        part = keys[start:start + 256]
        values = scorer.score([rows[key] for key in part], batch_size=args.batch_size)
        scores.update(zip(part, map(float, values)))
        print(f"v9: {start + len(part)}/{len(keys)} pairs", flush=True)
    model_seconds = time.perf_counter() - started
    comparison = None
    if args.compare_v9:
        earlier = ReferenceConditionedScorer(
            Path("artifacts/reference_detector_v9"), scorer.device)
        earlier_scores = {}
        for start in range(0, len(keys), 256):
            part = keys[start:start + 256]
            values = earlier.score([rows[key] for key in part], batch_size=args.batch_size)
            earlier_scores.update(zip(part, map(float, values)))
        earlier_threshold = json.loads(Path("results/reference_detector_v9_replay.json")
                                       .read_text(encoding="utf-8"))["threshold"]
        comparison = (earlier_scores, earlier_threshold)
    started = time.perf_counter()
    for index, key in enumerate(keys, 1):
        row = rows[key]
        baseline[key] = old.score(row["query"], row["candidate"])["risk_score"]
        if index % 500 == 0:
            print(f"v3: {index}/{len(keys)} pairs", flush=True)
    old_seconds = time.perf_counter() - started

    def collect(values: dict, cutoff: float) -> dict:
        clean = [values[case["query_id"], "clean:" + doc["id"]]
                 for case in cases for doc in case["clean"]]
        references = [values[case["query_id"], "reference"] for case in cases]
        family_results = {}
        for family in counts:
            if not isinstance(counts[family], dict):
                continue
            docs = [values[case["query_id"], f"{family}:{index}"]
                    for case in cases for index, _ in enumerate(
                        case["attacks"].get(family, []))]
            questions = [case for case in cases if family in case["attacks"]]
            family_results[family] = {
                "questions": len(questions), "attack_documents": len(docs),
                "caught_documents": sum(score >= cutoff for score in docs),
                "questions_with_any_caught": sum(any(
                    values[case["query_id"], f"{family}:{index}"] >= cutoff
                    for index in range(len(case["attacks"][family])))
                    for case in questions)}
        return {"clean_candidates": len(clean),
                "clean_false_flags": sum(score >= cutoff for score in clean),
                "judged_references": len(references),
                "reference_false_flags": sum(score >= cutoff for score in references),
                "families": family_results}

    report = {"protocol": ("research_protocols/multifamily_detector_v10.md"
                            if args.heldout_v10 else "research_protocols/reference_detector_v9.md"),
              "model_weights_sha256": hashlib.sha256(
                  (args.model / "model.safetensors").read_bytes()).hexdigest(),
              "poisonarena_manifest_sha256": hashlib.sha256(
                  (args.data / "manifest.json").read_bytes()).hexdigest(),
              "selection": counts, "evaluated_questions": len(cases),
              "heldout_v10_questions": args.heldout_v10,
              "threshold_frozen_from_other_archive": threshold,
              "candidate_model": collect(scores, threshold),
              "frozen_v3": collect(baseline, old.threshold),
              "candidate_model_seconds": round(model_seconds, 2),
              "v3_seconds": round(old_seconds, 2),
              "unique_scored_pairs": len(keys),
              "limitation": "PoisonArena released qrel is an oracle-like proxy, not an independently authenticated live source."}
    if comparison is not None:
        report["prior_v9_same_questions"] = collect(*comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
