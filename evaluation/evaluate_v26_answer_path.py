"""Run actual retrieval, source admission, scanning, and extractive answering on FiQA.

This is a post-holdout engineering replay on the same cases, not a second blind
test. The staged index contains all 120 selected questions' fixed attack cases.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from detect.product_v26 import RoleAdaptiveScanner
from generation.grounded_answer import answer_from_accepted
from pipeline.v24_answer_gate import answer_with_v24
from retrieval.hybrid_retriever import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/v26_fiqa_holdout/cases.jsonl"
SCANNER_REPORT = ROOT / "results/v26_fiqa_holdout.json"
OUTPUT = ROOT / "results/v26_fiqa_answer_path.json"
TRACE = ROOT / "results/v26_fiqa_answer_path.cases.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> list[dict]:
    return [json.loads(line) for line in CASES.open(encoding="utf-8") if line.strip()]


def _v25_decision(result: dict) -> str:
    return "review" if any((result["v3"]["flag"], result["reference_adaptive"]["flag"],
                            result["integrity"].get("flag", False), result["identity"]["flag"])) else "pass"


def evaluate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    scanner_report = json.loads(SCANNER_REPORT.read_text(encoding="utf-8"))
    if scanner_report["status"] != "first_scoring_after_rule_freeze":
        raise ValueError("Finish the frozen scanner holdout first")
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in (
        CASES, SCANNER_REPORT, ROOT / "results/v26_rule_frozen.json")}
    rows = _rows()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    if len(grouped) != 120 or len(rows) != 960:
        raise ValueError("FiQA answer replay inventory changed")
    schedule = sorted(grouped)
    clean_rows = [row for row in rows if row["label"] in {"reference", "clean_other"}]
    expected_hashes: dict[str, str] = {}
    attack_ids: set[int] = set()
    source_rows: list[dict] = []
    for index, row in enumerate(rows, 1):
        source_rows.append({"doc_id": index, "text": row["candidate"]})
        row["replay_doc_id"] = index
        if row["label"] == "poison":
            attack_ids.add(index)
        elif row["label"] in {"reference", "clean_other"}:
            expected_hashes[str(index)] = hashlib.sha256(row["candidate"].encode()).hexdigest()
    clean_ids = {row["replay_doc_id"] for row in clean_rows}
    prior = [json.loads(line) for line in TRACE.open(encoding="utf-8") if line.strip()] if TRACE.exists() else []
    if [row["query_id"] for row in prior] != schedule[:len(prior)]:
        raise ValueError("Answer replay trace differs from the fixed question schedule")
    with tempfile.TemporaryDirectory(prefix="sentinel-v26-fiqa-answer-") as directory:
        work = Path(directory)
        staged_path, clean_path = work / "staged.jsonl", work / "clean.jsonl"
        for path, docs in ((staged_path, source_rows),
                           (clean_path, [doc for doc in source_rows if doc["doc_id"] in clean_ids])):
            with path.open("w", encoding="utf-8", newline="\n") as sink:
                for doc in docs:
                    sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
        staged = HybridRetriever(staged_path, artifact_dir=work / "staged_index")
        clean = HybridRetriever(clean_path, artifact_dir=work / "clean_index")
        if staged.embedder.backend != "hashing" or clean.embedder.backend != "hashing":
            raise RuntimeError("Replay requires deterministic hashing retrieval")
        scanner = RoleAdaptiveScanner(root=ROOT)
        try:
            with TRACE.open("a", encoding="utf-8", newline="\n") as sink:
                for index, qid in enumerate(schedule[len(prior):], len(prior) + 1):
                    group = grouped[qid]
                    query = group[0]["query"]
                    reference = group[0]["reference"]
                    ref_row = next(row for row in group if row["label"] == "reference")
                    if expected_hashes[str(ref_row["replay_doc_id"])] != hashlib.sha256(reference.encode()).hexdigest():
                        raise ValueError("Reference is not pinned to the original document")
                    retrieved = staged.retrieve(query, top_k=5)
                    clean_retrieved = clean.retrieve(query, top_k=5)
                    clean_answer = answer_from_accepted(query, clean_retrieved, clean)
                    cache: dict[tuple[str, str | None], dict] = {}

                    def scan26(q: str, ref: str, candidate: str, expected: str | None) -> dict:
                        key = (candidate, expected)
                        if key not in cache:
                            cache[key] = scanner.assess(q, ref, candidate,
                                                        expected_sha256=expected)
                        return cache[key]

                    def scan25(q: str, ref: str, candidate: str, expected: str | None) -> dict:
                        result = scan26(q, ref, candidate, expected)
                        return {**result, "decision": _v25_decision(result)}

                    options = {"pinned_reference": reference,
                               "pinned_reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                               "expected_hashes": expected_hashes}
                    v26 = answer_with_v24(query, retrieved, staged, scan=scan26, **options)
                    v25 = answer_with_v24(query, retrieved, staged, scan=scan25, **options)
                    retrieved_ids = {doc["doc_id"] for doc in retrieved}
                    record = {
                        "query_id": qid,
                        "retrieved_attack_count": len(retrieved_ids & attack_ids),
                        "retrieved_original_clean_count": len(retrieved_ids & clean_ids),
                        "clean_answer_source": clean_answer["source_doc_id"],
                        "clean_answer_sha256": hashlib.sha256(clean_answer["answer"].encode()).hexdigest(),
                        "undefended_source": v26["undefended"]["source_doc_id"],
                        "undefended_attack_sourced": v26["undefended"]["source_doc_id"] in attack_ids,
                        "undefended_target_answer": (v26["undefended"]["source_doc_id"] in attack_ids
                                                    and "WITHHELD" in v26["undefended"]["answer"]),
                    }
                    for version, result in (("v25", v25), ("v26", v26)):
                        defended = result["defended"]
                        record.update({
                            f"{version}_source": defended["source_doc_id"],
                            f"{version}_attack_sourced": defended["source_doc_id"] in attack_ids,
                            f"{version}_target_answer": (defended["source_doc_id"] in attack_ids
                                                        and "WITHHELD" in defended["answer"]),
                            f"{version}_clean_answer_match": defended["answer"] == clean_answer["answer"],
                            f"{version}_abstained": defended["source_doc_id"] is None,
                            f"{version}_scanner_attack_reviews": sum(
                                item["doc_id"] in attack_ids and item["scanner_decision"] == "review"
                                for item in result["documents"]),
                            f"{version}_admission_attack_reviews": sum(
                                item["doc_id"] in attack_ids and item["decision"] == "review"
                                for item in result["documents"]),
                        })
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    prior.append(record)
                    if index % 10 == 0:
                        print(f"Replayed {index}/{len(schedule)} FiQA questions through retrieval and answer gate", flush=True)
        finally:
            scanner.close()
    if any(sha(ROOT / name) != digest for name, digest in inputs.items()):
        raise ValueError("A pinned answer replay input changed")
    def totals(key: str) -> int:
        return sum(int(row[key]) for row in prior)
    report = {
        "protocol": "v26-fiqa-post-holdout-retrieval-answer-replay-v1",
        "status": "post_holdout_engineering_replay",
        "inputs_sha256": inputs, "trace_sha256": sha(TRACE),
        "questions": len(prior), "index": "120 FiQA reference + 120 other original + 120 benign markup + 600 attack documents",
        "retriever": "BM25 plus hashing dense reciprocal-rank fusion, top 5",
        "answerer": "deterministic extractive span selector; no hosted LLM",
        "retrieved_attack_documents": totals("retrieved_attack_count"),
        "questions_with_retrieved_attack": sum(row["retrieved_attack_count"] > 0 for row in prior),
        "clean_preattack_answer_available": sum(row["clean_answer_source"] is not None for row in prior),
        "undefended_attack_sourced_answers": totals("undefended_attack_sourced"),
        "undefended_attack_sourced_target_answers": totals("undefended_target_answer"),
        "versions": {version: {
            "attack_sourced_answers": totals(f"{version}_attack_sourced"),
            "attack_sourced_target_answers": totals(f"{version}_target_answer"),
            "clean_answer_exact_matches": totals(f"{version}_clean_answer_match"),
            "abstentions": totals(f"{version}_abstained"),
            "scanner_reviews_of_retrieved_attacks": totals(f"{version}_scanner_attack_reviews"),
            "source_admission_reviews_of_retrieved_attacks": totals(f"{version}_admission_attack_reviews"),
        } for version in ("v25", "v26")},
        "limits": [
            "This reuses the FiQA holdout after scanner scoring; it is an engineering replay, not fresh independent evidence.",
            "All attacks are new-document insertions and are blocked from answer sourcing by authenticated source admission.",
            "The extractor reports source spans, not verified factual answers or hosted LLM behavior.",
            "The 960-document index is a controlled challenge subset, not full FiQA retrieval over 57k documents.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
