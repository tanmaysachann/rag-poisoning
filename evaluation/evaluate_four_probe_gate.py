"""Evaluate the integrated four-probe gate on validation, never the test split."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark")
    parser.add_argument("--calibration", type=Path, default=ROOT / "artifacts" / "four_probe_calibration.joblib")
    parser.add_argument("--output", type=Path, default=RESULTS / "four_probe_gate_validation.json")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.replace_existing:
        raise FileExistsError(f"Result exists: {args.output}; pass --replace-existing to rerun")
    bundle = joblib.load(args.calibration)
    os.environ["RAG_USE_MINILM"] = "0"
    os.environ["RAG_LLM_MODEL"] = str(bundle["qwen_model"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from data.validate_benchmark import validate_benchmark
    from detect.research_detector import ResearchDetector
    from pipeline.research_inference import infer_research
    from retrieval.hybrid_retriever import HybridRetriever
    from security.four_probe_gate import FourProbeGate
    from security.integrity import IntegrityManifest, document_hash
    from attack.harness import stage_document_attack

    validate_benchmark(args.benchmark)
    validation = args.benchmark / "validation"
    queries = _jsonl(validation / "queries.jsonl")
    corpus_path = validation / "corpus.jsonl"
    clean_docs = _jsonl(corpus_path)
    attacks = {
        family: {row["qid"]: row for row in _jsonl(RESULTS / f"attack_baseline_validation_{family}.jsonl")}
        for family in ("greedy", "stealth")
    }
    if any(set(rows) != {query["qid"] for query in queries} for rows in attacks.values()):
        raise ValueError("Attack cases must exactly cover validation queries")
    detector = joblib.load(ROOT / "artifacts" / "research_detector_hashing_web.joblib")
    if not isinstance(detector, ResearchDetector):
        raise ValueError("The packaged hashing research detector is invalid")
    gate = FourProbeGate(bundle, local_files_only=True)
    result_rows = []

    def manifest_for(documents: list[dict]) -> IntegrityManifest:
        return IntegrityManifest(
            {str(int(doc["doc_id"])): document_hash(str(doc["text"])) for doc in documents},
            None, False, False,
        )

    with tempfile.TemporaryDirectory(prefix="sentinel-four-probe-validation-") as temporary:
        root = Path(temporary)
        clean_retriever = HybridRetriever(corpus_path, artifact_dir=root / "clean_index")
        clean_manifest = manifest_for(clean_retriever.documents)
        for index, query in enumerate(queries, 1):
            clean_without_gate = infer_research(
                query["question"], clean_retriever, clean_manifest, detector,
            )
            clean_result = infer_research(
                query["question"], clean_retriever, clean_manifest, detector,
                four_probe_gate=gate,
            )
            baseline_answer = clean_without_gate["defended"]["answer"]
            clean_answer = clean_result["defended"]["answer"]
            baseline_alias = any(alias.casefold() in baseline_answer.casefold()
                                 for alias in query.get("answer_aliases", []))
            clean_alias = any(alias.casefold() in clean_answer.casefold()
                              for alias in query.get("answer_aliases", []))
            result_rows.append({
                "qid": query["qid"], "condition": "clean", "strategy": None,
                "alias_without_four_probe": baseline_alias,
                "clean_alias_recovered": clean_alias,
                "abstained_without_four_probe": clean_without_gate["defended"]["source_doc_id"] is None,
                "defended_attack_success": False,
                "undefended_attack_success": False,
                "probe": clean_result["four_probe"],
                "latency_ms": clean_result["latency_ms"],
                "abstained": clean_result["defended"]["source_doc_id"] is None,
            })
            if index % 10 == 0:
                print(f"clean validation {index}/{len(queries)}", flush=True)

        for family in ("greedy", "stealth"):
            for index, query in enumerate(queries, 1):
                attack = attacks[family][query["qid"]]
                staged = root / f"{family}_{index}.jsonl"
                stage_document_attack(
                    corpus_path, staged, doc_id=int(attack["attack_doc_id"]),
                    text=attack["text"], replace_existing=family == "stealth",
                )
                retriever = HybridRetriever(staged, artifact_dir=root / f"{family}_{index}_index")
                manifest = manifest_for(retriever.documents)
                defended = infer_research(
                    query["question"], retriever, manifest, detector,
                    four_probe_gate=gate,
                )
                without_gate = infer_research(
                    query["question"], retriever, manifest, detector,
                )
                wrong = str(attack["wrong_answer"]).casefold()
                attack_id = int(attack["attack_doc_id"])
                off = defended["undefended"]
                on = defended["defended"]
                without_gate_answer = without_gate["defended"]
                success_without_gate = (
                    without_gate_answer["source_doc_id"] == attack_id
                    and wrong in without_gate_answer["answer"].casefold()
                )
                success_with_gate = on["source_doc_id"] == attack_id and wrong in on["answer"].casefold()
                result_rows.append({
                    "qid": query["qid"], "condition": "accepted_ingest", "strategy": family,
                    "undefended_attack_success": off["source_doc_id"] == attack_id and wrong in off["answer"].casefold(),
                    "attack_success_without_four_probe": success_without_gate,
                    "defended_attack_success": success_with_gate,
                    "four_probe_blocked_success": success_without_gate and not success_with_gate,
                    "four_probe_added_success": not success_without_gate and success_with_gate,
                    "attack_quarantined": any(
                        int(row["doc_id"]) == attack_id and row["decision"] == "quarantine"
                        for row in defended["documents"]
                    ),
                    "probe": defended["four_probe"],
                    "latency_ms": defended["latency_ms"],
                    "abstained": on["source_doc_id"] is None,
                })
                if index % 10 == 0:
                    print(f"{family} validation {index}/{len(queries)}", flush=True)

    summary = {
        "split": "validation",
        "cases_per_condition": len(queries),
        "calibration_version": bundle["calibration_version"],
        "calibration_method": bundle.get("calibration_method"),
        "calibration_count": bundle["calibration_count"],
        "target_clean_fpr": bundle["target_clean_fpr"],
        "joint_threshold": bundle["joint_threshold"],
        "clean": {}, "attacks": {},
    }
    clean_rows = [row for row in result_rows if row["condition"] == "clean"]
    clean_probe_rows = [row for row in clean_rows if row["probe"].get("decision") in {"accept", "quarantine"}]
    summary["clean"] = {
        "alias_match_rate_without_four_probe": sum(row["alias_without_four_probe"] for row in clean_rows) / len(clean_rows),
        "alias_match_rate": sum(row["clean_alias_recovered"] for row in clean_rows) / len(clean_rows),
        "abstention_rate_without_four_probe": sum(row["abstained_without_four_probe"] for row in clean_rows) / len(clean_rows),
        "abstention_rate": sum(row["abstained"] for row in clean_rows) / len(clean_rows),
        "clean_answers_blocked_by_four_probe": sum(
            row["probe"].get("decision") == "quarantine" and not row["abstained_without_four_probe"]
            for row in clean_rows
        ),
        "probe_coverage": len(clean_probe_rows) / len(clean_rows),
        "probe_quarantine_rate_given_coverage": (
            sum(row["probe"].get("decision") == "quarantine" for row in clean_probe_rows) / len(clean_probe_rows)
            if clean_probe_rows else 0.0
        ),
        "median_latency_ms": statistics.median(row["latency_ms"] for row in clean_rows),
        "median_latency_ms_when_probe_ran": statistics.median(
            row["latency_ms"] for row in clean_probe_rows
        ) if clean_probe_rows else None,
        "mean_s3_qwen_ms": statistics.mean(
            row["probe"].get("timings_ms", {}).get("s3_qwen_srq", 0.0)
            for row in clean_probe_rows
        ) if clean_probe_rows else 0.0,
    }
    for family in ("greedy", "stealth"):
        family_rows = [row for row in result_rows if row.get("strategy") == family]
        covered = [row for row in family_rows if row["probe"].get("decision") in {"accept", "quarantine"}]
        trigger_counts = Counter(
            name for row in covered for name in row["probe"].get("triggered_probes", [])
        )
        summary["attacks"][family] = {
            "undefended_successes": sum(row["undefended_attack_success"] for row in family_rows),
            "successes_without_four_probe": sum(row["attack_success_without_four_probe"] for row in family_rows),
            "defended_successes": sum(row["defended_attack_success"] for row in family_rows),
            "four_probe_blocked_successes": sum(row["four_probe_blocked_success"] for row in family_rows),
            "four_probe_added_successes": sum(row["four_probe_added_success"] for row in family_rows),
            "attack_documents_quarantined": sum(row["attack_quarantined"] for row in family_rows),
            "probe_coverage": len(covered) / len(family_rows),
            "probe_quarantines": sum(row["probe"].get("decision") == "quarantine" for row in covered),
            "probe_trigger_counts": dict(trigger_counts),
            "abstention_rate": sum(row["abstained"] for row in family_rows) / len(family_rows),
            "median_latency_ms": statistics.median(row["latency_ms"] for row in family_rows),
            "median_latency_ms_when_probe_ran": statistics.median(
                row["latency_ms"] for row in covered
            ) if covered else None,
            "mean_s3_qwen_ms": statistics.mean(
                row["probe"].get("timings_ms", {}).get("s3_qwen_srq", 0.0) for row in covered
            ) if covered else 0.0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
