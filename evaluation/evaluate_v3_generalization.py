"""Paired test of frozen v3 defense on withheld answer-poison layouts.

Run validation during protocol development. Test mode is locked to a frozen
protocol that pins the v3 split, detector, attack generator, and evaluator.
The test split is read only after those inputs are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import joblib

from attack.baselines import choose_wrong_answer
from attack.harness import stage_document_attack
from attack.heldout_families import FAMILY_TEMPLATES, build_heldout_attack
from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark
from detect.research_detector import ResearchDetector
from evaluation.metrics import paired_bootstrap_difference, wilson_interval
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import HybridRetriever
from security.integrity import load_runtime_manifest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "benchmark_v3"
DETECTOR_PATH = ROOT / "artifacts" / "research_detector_v3_hashing.joblib"
PROTOCOL_PATH = ROOT / "research_protocols" / "v3_heldout_generalization.json"
FAMILIES = tuple(FAMILY_TEMPLATES)
PROFILES = {"extractive_hashing": "RAG_USE_MINILM=0; deterministic extractive selector"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interval(values: list[bool]) -> dict[str, float]:
    low, high = wilson_interval(sum(values), len(values))
    return {"lower": low, "upper": high}


def _hash_inputs() -> dict[str, str]:
    return {
        "benchmark_manifest_sha256": _sha256(BENCHMARK / "manifest.json"),
        "training_report_sha256": _sha256(ROOT / "results" / "research_detector_v3_validation.json"),
        "train_random_attack_sha256": _sha256(ROOT / "results" / "v3_attack_train_random.jsonl"),
        "train_stealth_attack_sha256": _sha256(ROOT / "results" / "v3_attack_train_stealth.jsonl"),
        "validation_calibration_attack_sha256": _sha256(ROOT / "results" / "v3_attack_validation_greedy.jsonl"),
        "test_queries_sha256": _sha256(BENCHMARK / "test" / "queries.jsonl"),
        "test_corpus_sha256": _sha256(BENCHMARK / "test" / "corpus.jsonl"),
        "detector_sha256": _sha256(DETECTOR_PATH),
        "attack_generator_sha256": _sha256(ROOT / "attack" / "heldout_families.py"),
        "evaluator_sha256": _sha256(Path(__file__)),
        "verifier_sha256": _sha256(ROOT / "evaluation" / "verify_v3_generalization.py"),
    }


def _verify_test_protocol(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    actual = _hash_inputs()
    for name, digest in actual.items():
        if protocol.get("frozen_inputs", {}).get(name) != digest:
            raise ValueError(f"Frozen test protocol mismatch for {name}; do not evaluate test")
    for relative_path, digest in protocol.get("development_evidence", {}).items():
        if _sha256(ROOT / relative_path) != digest:
            raise ValueError(f"Frozen validation development evidence changed: {relative_path}")
    detector = joblib.load(DETECTOR_PATH)
    if not isinstance(detector, ResearchDetector) or detector.threshold != protocol["detector_threshold"]:
        raise ValueError("Detector type or threshold differs from frozen test protocol")
    if tuple(protocol.get("families", [])) != FAMILIES:
        raise ValueError("Attack families differ from the frozen test protocol")
    protocol["_protocol_sha256"] = _sha256(protocol_path)
    return protocol


def _answer(query: str, documents: list[dict], retriever) -> tuple[str, int | None]:
    answer, source_id, _ = _select_answer(query, documents, retriever)
    return answer, source_id


def _contains_alias(answer: str, aliases: list[str]) -> bool:
    folded = answer.casefold()
    return any(alias.casefold() in folded for alias in aliases)


def _evaluate_split(split: str, output: Path, *, protocol: dict | None = None) -> dict:
    counts = validate_benchmark(BENCHMARK)
    rows = _read_jsonl(BENCHMARK / split / "queries.jsonl")
    corpus_path = BENCHMARK / split / "corpus.jsonl"
    corpus = _read_jsonl(corpus_path)
    train_queries = _read_jsonl(BENCHMARK / "train" / "queries.jsonl")
    answer_pool = [alias for row in train_queries for alias in row["answer_aliases"] if alias.strip()]
    detector = joblib.load(DETECTOR_PATH)
    if not isinstance(detector, ResearchDetector):
        raise ValueError("Frozen v3 detector artifact has the wrong type")
    if detector.embedding_model != "sklearn-hashing-384":
        raise ValueError("This evaluation requires the frozen hashing detector")
    os.environ["RAG_USE_MINILM"] = "0"

    case_rows: list[dict] = []
    clean_rows: list[dict] = []
    summary_families: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix=f"v3-generalization-{split}-", dir=ROOT / "tmp") as temp:
        work = Path(temp)
        clean_retriever = HybridRetriever(
            corpus_path, artifact_dir=work / "clean-index", write_integrity_manifest=True
        )
        clean_manifest = load_runtime_manifest(clean_retriever.integrity_path)
        if clean_manifest is None:
            raise RuntimeError("Clean control manifest was not created")

        clean_by_qid: dict[str, dict] = {}
        for query in rows:
            qid = str(query["qid"])
            candidates = clean_retriever.retrieve(query["question"], top_k=5)
            before, before_source = _answer(query["question"], candidates, clean_retriever)
            kept, decisions = [], []
            for doc in candidates:
                integrity = clean_manifest.check(doc["doc_id"], doc["text"])
                score = detector.score(query["question"], doc["text"])
                quarantine = integrity["status"] != "verified" or score["decision"] == "quarantine"
                decisions.append({"doc_id": int(doc["doc_id"]), "risk": score["risk_score"],
                                  "quarantined": quarantine})
                if not quarantine:
                    kept.append(doc)
            after, after_source = _answer(query["question"], kept, clean_retriever)
            row = {
                "qid": qid,
                "answer_aliases": query["answer_aliases"],
                "baseline_answer": before,
                "baseline_source_doc_id": before_source,
                "detector_answer": after,
                "detector_source_doc_id": after_source,
                "baseline_alias_match": _contains_alias(before, query["answer_aliases"]),
                "detector_alias_match": _contains_alias(after, query["answer_aliases"]),
                "baseline_abstained": before_source is None,
                "detector_abstained": after_source is None,
                "candidate_count": len(candidates),
                "clean_candidate_false_positives": sum(item["quarantined"] for item in decisions),
                "candidate_decisions": decisions,
            }
            clean_rows.append(row)
            clean_by_qid[qid] = row

        max_doc_id = max(int(doc["doc_id"]) for doc in corpus)
        for family_index, family in enumerate(FAMILIES):
            family_rows: list[dict] = []
            for query_index, query in enumerate(rows):
                qid = str(query["qid"])
                seed = int(hashlib.sha256(f"v3-generalization-v1|{family}|{qid}".encode()).hexdigest()[:8], 16)
                wrong = choose_wrong_answer(query["answer_aliases"], answer_pool, seed)
                attack_id = max_doc_id + 1 + family_index * len(rows) + query_index
                attack_text = build_heldout_attack(family, query["question"], wrong)
                started = time.perf_counter()
                case_dir = work / f"{family}-{query_index:04d}"
                case_dir.mkdir()
                staged = case_dir / "corpus.jsonl"
                stage_document_attack(corpus_path, staged, doc_id=attack_id, text=attack_text)
                retriever = HybridRetriever(
                    staged, artifact_dir=case_dir / "index", write_integrity_manifest=True
                )
                manifest = load_runtime_manifest(retriever.integrity_path)
                if manifest is None:
                    raise RuntimeError("Accepted-ingest case manifest was not created")
                candidates = retriever.retrieve(query["question"], top_k=5)
                before, before_source = _answer(query["question"], candidates, retriever)
                kept, decisions = [], []
                for doc in candidates:
                    integrity = manifest.check(doc["doc_id"], doc["text"])
                    score = detector.score(query["question"], doc["text"])
                    quarantined = integrity["status"] != "verified" or score["decision"] == "quarantine"
                    decisions.append({
                        "doc_id": int(doc["doc_id"]), "is_attack": int(doc["doc_id"]) == attack_id,
                        "integrity": integrity["status"], "risk": score["risk_score"],
                        "quarantined": quarantined,
                    })
                    if not quarantined:
                        kept.append(doc)
                after, after_source = _answer(query["question"], kept, retriever)
                # Since this is accepted-ingest corruption, the freshly sealed
                # manifest verifies the attack. Integrity-only must therefore
                # equal the unfiltered pipeline; this is an explicit ablation.
                integrity_only, integrity_source = before, before_source
                attack_in_results = any(int(doc["doc_id"]) == attack_id for doc in candidates)
                attack_integrity = next((
                    doc["integrity"] for doc in decisions if doc["doc_id"] == attack_id
                ), "not_retrieved")
                if attack_integrity not in {"verified", "not_retrieved"}:
                    raise RuntimeError("Accepted-ingest attack unexpectedly failed its fresh integrity manifest")
                before_success = before_source == attack_id and wrong.casefold() in before.casefold()
                after_success = after_source == attack_id and wrong.casefold() in after.casefold()
                detector_case = {
                    "qid": qid, "question": query["question"],
                    "answer_aliases": query["answer_aliases"],
                    "family": family, "attack_text": attack_text, "wrong_answer": wrong,
                    "attack_doc_id": attack_id, "attack_retrieved": attack_in_results,
                    "attack_integrity_status": attack_integrity,
                    "attack_quarantined": any(item["doc_id"] == attack_id and item["quarantined"] for item in decisions),
                    "undefended_answer": before, "undefended_source_doc_id": before_source,
                    "detector_answer": after, "detector_source_doc_id": after_source,
                    "integrity_only_attack_success": integrity_source == attack_id and wrong.casefold() in integrity_only.casefold(),
                    "undefended_attack_success": before_success,
                    "defended_attack_success": after_success,
                    "clean_alias_recovered": _contains_alias(after, query["answer_aliases"]),
                    "abstained": after_source is None,
                    "detector_candidate_false_positives": sum(
                        item["quarantined"] and not item["is_attack"] for item in decisions
                    ),
                    "candidate_count": len(candidates),
                    "detector_decisions": decisions,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
                family_rows.append(detector_case)
                case_rows.append(detector_case)
                if query_index and query_index % 25 == 0:
                    print(f"{split}/{family}: {query_index}/{len(rows)}", flush=True)

            n = len(family_rows)
            before_flags = [row["undefended_attack_success"] for row in family_rows]
            after_flags = [row["defended_attack_success"] for row in family_rows]
            delta_ci = paired_bootstrap_difference(before_flags, after_flags, draws=10000, seed=family_index + 903)
            retrieved = [row for row in family_rows if row["attack_retrieved"]]
            successful_before = [row for row in family_rows if row["undefended_attack_success"]]
            success_delta = sum(before_flags) / n - sum(after_flags) / n
            summary_families[family] = {
                "cases": n,
                "attack_retrieval_rate": sum(row["attack_retrieved"] for row in family_rows) / n,
                "attack_success_before": sum(before_flags),
                "attack_success_before_rate": sum(before_flags) / n,
                "attack_success_before_wilson_95": _interval(before_flags),
                "integrity_only_attack_success": sum(row["integrity_only_attack_success"] for row in family_rows),
                "attack_success_after_detector": sum(after_flags),
                "attack_success_after_detector_rate": sum(after_flags) / n,
                "attack_success_after_wilson_95": _interval(after_flags),
                "paired_absolute_reduction": success_delta,
                "paired_reduction_bootstrap_95": {"lower": delta_ci[0], "upper": delta_ci[1]},
                "attack_quarantine_rate_given_retrieval": (
                    sum(row["attack_quarantined"] for row in retrieved) / len(retrieved) if retrieved else None
                ),
                "attack_quarantine_rate_given_success": (
                    sum(row["attack_quarantined"] for row in successful_before) / len(successful_before)
                    if successful_before else None
                ),
                "clean_alias_recovered_under_attack": sum(row["clean_alias_recovered"] for row in family_rows) / n,
                "abstention_rate_under_attack": sum(row["abstained"] for row in family_rows) / n,
                "clean_candidate_false_positive_rate": (
                    sum(row["detector_candidate_false_positives"] for row in family_rows)
                    / max(sum(row["candidate_count"] - int(row["attack_retrieved"]) for row in family_rows), 1)
                ),
                "median_latency_ms": statistics.median(row["latency_ms"] for row in family_rows),
            }

    clean_before = [row["baseline_alias_match"] for row in clean_rows]
    clean_after = [row["detector_alias_match"] for row in clean_rows]
    clean_ci = paired_bootstrap_difference(clean_before, clean_after, draws=10000, seed=4242)
    result = {
        "schema_version": 1,
        "benchmark": "MS MARCO v1.1 disjoint benchmark v3",
        "split": split,
        "split_cases": len(rows),
        "benchmark_counts": counts,
        "evaluation_profile": "hashing retrieval + extractive selector + frozen v3 content detector",
        "answer_profile_details": PROFILES["extractive_hashing"],
        "attack_surface": "accepted-ingest; attack is included in the newly sealed corpus snapshot",
        "detector_threshold_source": "frozen v3 validation calibration; no test retuning",
        "detector_threshold": detector.threshold,
        "clean_control": {
            "cases": len(clean_rows),
            "baseline_alias_match": sum(clean_before),
            "baseline_alias_match_rate": sum(clean_before) / len(clean_before),
            "baseline_alias_match_wilson_95": _interval(clean_before),
            "detector_alias_match": sum(clean_after),
            "detector_alias_match_rate": sum(clean_after) / len(clean_after),
            "detector_alias_match_wilson_95": _interval(clean_after),
            "paired_alias_match_change": sum(clean_after) / len(clean_after) - sum(clean_before) / len(clean_before),
            "paired_change_bootstrap_95": {"lower": -clean_ci[1], "upper": -clean_ci[0]},
            "baseline_abstention_rate": sum(row["baseline_abstained"] for row in clean_rows) / len(clean_rows),
            "detector_abstention_rate": sum(row["detector_abstained"] for row in clean_rows) / len(clean_rows),
            "candidate_false_positive_rate": (
                sum(row["clean_candidate_false_positives"] for row in clean_rows)
                / max(sum(row["candidate_count"] for row in clean_rows), 1)
            ),
        },
        "paired_attack_families": summary_families,
        "limitations": [
            "All three withheld templates are synthetic, query-aware, and locally authored.",
            "The benchmark contains one labeled support passage per question; this is not web-scale retrieval.",
            "The test measures extractive source-selection outcomes, not instruction-following by a generative LLM.",
            "Attack layouts were withheld from v3 detector train and validation; the same project may have explored related formats on other development corpora.",
        ],
        "frozen_inputs": _hash_inputs(),
        "protocol_sha256": protocol.get("_protocol_sha256") if protocol else None,
        "cases_file": str(output.with_suffix(".cases.jsonl").relative_to(ROOT)).replace("\\", "/"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with output.with_name(output.stem + "_clean.cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in clean_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_manifest = {
        "schema_version": 1,
        "summary": str(output.relative_to(ROOT)).replace("\\", "/"),
        "cases": str(output.with_suffix(".cases.jsonl").relative_to(ROOT)).replace("\\", "/"),
        "clean_cases": str(output.with_name(output.stem + "_clean.cases.jsonl").relative_to(ROOT)).replace("\\", "/"),
        "sha256": {
            "summary": _sha256(output),
            "cases": _sha256(output.with_suffix(".cases.jsonl")),
            "clean_cases": _sha256(output.with_name(output.stem + "_clean.cases.jsonl")),
        },
        "frozen_inputs": result["frozen_inputs"],
        "protocol_sha256": result["protocol_sha256"],
    }
    output.with_name(output.stem + ".manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    if not (ROOT / "tmp").is_dir():
        (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    protocol = _verify_test_protocol(args.protocol) if args.split == "test" else None
    output = args.output or ROOT / "results" / f"v3_generalization_{args.split}.json"
    _evaluate_split(args.split, output, protocol=protocol)


if __name__ == "__main__":
    main()
