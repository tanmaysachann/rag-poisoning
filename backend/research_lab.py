"""Bounded, live validation experiment for the deployed research console.

The browser may edit one attack document, but cannot select the test split,
change a question, retrain a model, or modify the checked-in corpus. Each run
builds an isolated temporary index and integrity manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import joblib

from attack.harness import stage_document_attack
from detect.research_detector import ResearchDetector
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import HybridRetriever
from security.integrity import load_runtime_manifest, write_manifest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data" / "benchmark" / "validation"
RESULTS = ROOT / "results"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=1)
def _inputs() -> tuple[dict, dict, dict, dict]:
    queries = {row["qid"]: row for row in _jsonl(VALIDATION / "queries.jsonl")}
    corpus = {int(row["doc_id"]): row for row in _jsonl(VALIDATION / "corpus.jsonl")}
    attacks = {
        strategy: {row["qid"]: row for row in _jsonl(RESULTS / f"attack_baseline_validation_{strategy}.jsonl")}
        for strategy in ("greedy", "stealth")
    }
    saved = {
        strategy: {row["qid"]: row for row in _jsonl(RESULTS / f"defense_validation_{strategy}_accepted_ingest.cases.jsonl")}
        for strategy in ("greedy", "stealth")
    }
    if set(queries) != set(attacks["greedy"]) or set(queries) != set(attacks["stealth"]):
        raise ValueError("Validation attack records do not match the query split")
    return queries, corpus, attacks, saved


@lru_cache(maxsize=1)
def _detector() -> ResearchDetector:
    detector = joblib.load(ROOT / "artifacts" / "research_detector_hashing.joblib")
    if not isinstance(detector, ResearchDetector) or detector.embedding_model != "sklearn-hashing-384":
        raise ValueError("The packaged research detector has the wrong type or backend")
    return detector


def case_catalog() -> dict:
    queries, corpus, attacks, saved = _inputs()
    cases = []
    for qid, query in queries.items():
        support_id = int(query["support_doc_ids"][0])
        cases.append({
            "qid": qid,
            "question": query["question"],
            "support_doc_id": support_id,
            "original_text": corpus[support_id]["text"],
            "greedy_text": attacks["greedy"][qid]["text"],
            "stealth_text": attacks["stealth"][qid]["text"],
            "saved_stealth_failure": bool(saved["stealth"][qid]["defended_attack_success"]),
            "saved_greedy_blocked": bool(saved["greedy"][qid]["attack_quarantined"]),
        })
    return {"split": "validation", "cases": cases, "default_qid": "msmarco-275"}


def run_case(qid: str, strategy: str, surface: str, attack_text: str | None = None) -> dict:
    queries, corpus, attacks, _ = _inputs()
    if qid not in queries or strategy not in attacks or surface not in {"accepted_ingest", "post_index_tamper"}:
        raise ValueError("Select a listed validation case, attack family, and trust surface")
    attack = attacks[strategy][qid]
    query = queries[qid]
    text = attack["text"] if attack_text is None else attack_text
    if not text.strip() or len(text) > 4000:
        raise ValueError("The attack document must contain 1 to 4000 characters")
    started = time.perf_counter()
    corpus_path = VALIDATION / "corpus.jsonl"
    with tempfile.TemporaryDirectory(prefix="sentinel-live-lab-") as directory:
        work = Path(directory)
        clean_retriever = HybridRetriever(corpus_path, artifact_dir=work / "clean_index")
        clean_candidates = clean_retriever.retrieve(query["question"], top_k=5)
        clean_answer, clean_source, _ = _select_answer(query["question"], clean_candidates, clean_retriever)
        staged_path = work / "staged.jsonl"
        stage_document_attack(
            corpus_path, staged_path, doc_id=int(attack["attack_doc_id"]), text=text,
            replace_existing=strategy == "stealth",
        )
        clean_manifest_path = work / "clean_manifest.json"
        manifest_key = os.getenv("RAG_MANIFEST_KEY")
        write_manifest(
            clean_manifest_path, corpus.values(), hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            signing_key=manifest_key.encode("utf-8") if manifest_key else None,
        )
        retriever = HybridRetriever(
            staged_path,
            artifact_dir=work / "index",
            integrity_path=clean_manifest_path if surface == "post_index_tamper" else None,
            write_integrity_manifest=surface == "accepted_ingest",
        )
        if retriever.embedder.backend != "hashing":
            raise RuntimeError("The web research lab requires deterministic hashing retrieval")
        candidates = retriever.retrieve(query["question"], top_k=5)
        manifest = load_runtime_manifest(retriever.integrity_path)
        if manifest is None:
            raise RuntimeError("The isolated experiment has no integrity manifest")
        off_answer, off_source, _ = _select_answer(query["question"], candidates, retriever)
        detector = _detector()
        kept = []
        rows = []
        for rank, doc in enumerate(candidates, 1):
            integrity = manifest.check(doc["doc_id"], doc["text"])
            score = detector.score(query["question"], doc["text"])
            quarantine = integrity["status"] != "verified" or score["decision"] == "quarantine"
            if not quarantine:
                kept.append(doc)
            rows.append({
                "rank": rank,
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "is_attack": doc["doc_id"] == attack["attack_doc_id"],
                "bm25_rank": doc["bm25_rank"],
                "dense_rank": doc["dense_rank"],
                "rrf_score": doc["score"],
                "integrity": integrity["status"],
                "sha256": integrity["actual_hash"],
                "risk": score["risk_score"],
                "threshold": score["threshold"],
                "features": score["features"],
                "decision": "quarantine" if quarantine else "accept",
            })
        on_answer, on_source, _ = _select_answer(query["question"], kept, retriever)
    wrong = attack["wrong_answer"].casefold()
    attack_id = int(attack["attack_doc_id"])
    return {
        "split": "validation",
        "execution": "live_hashing_extractive",
        "qid": qid,
        "question": query["question"],
        "strategy": strategy,
        "surface": surface,
        "attack_doc_id": attack_id,
        "attack_retrieved": any(row["is_attack"] for row in rows),
        "attack_quarantined": any(row["is_attack"] and row["decision"] == "quarantine" for row in rows),
        "defended_source_is_attack": on_source == attack_id,
        "defended_answer_matches_clean": on_answer.casefold() == clean_answer.casefold(),
        "clean_alias_recovered": any(alias.casefold() in on_answer.casefold() for alias in query["answer_aliases"]),
        "undefended_attack_success": off_source == attack_id and wrong in off_answer.casefold(),
        "defended_attack_success": on_source == attack_id and wrong in on_answer.casefold(),
        "undefended": {"answer": off_answer, "source_doc_id": off_source},
        "defended": {"answer": on_answer, "source_doc_id": on_source},
        "clean": {"answer": clean_answer, "source_doc_id": clean_source},
        "documents": rows,
        "detector_threshold": detector.threshold,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "note": "This is a live validation run with the hashing extractor. MiniLM, PPO and Qwen results are saved offline experiments.",
    }


def ppo_history() -> dict:
    undefended = json.loads((RESULTS / "ppo_validation_evaluation.json").read_text(encoding="utf-8"))
    defender_aware = json.loads((RESULTS / "ppo_validation_defender_evaluation.json").read_text(encoding="utf-8"))
    return {
        "split": "train",
        "runs": {
            "undefended": _jsonl(RESULTS / "ppo_train.jsonl"),
            "defender_aware": _jsonl(RESULTS / "ppo_train_defender.jsonl"),
        },
        "validation": {
            "undefended": {key: value for key, value in undefended["summary"].items() if key != "checkpoint"},
            "defender_aware": {key: value for key, value in defender_aware["summary"].items() if key != "checkpoint"},
            "fixed_insertion": {
                key: value for key, value in json.loads(
                    (RESULTS / "edit_rollout_validation_greedy_proxy.summary.json").read_text(encoding="utf-8")
                ).items() if key != "output"
            },
        },
    }


def ppo_case(qid: str) -> dict:
    queries, _, _, _ = _inputs()
    if qid not in queries:
        raise ValueError("Select a listed validation case")
    runs = {}
    for name, filename in (
        ("undefended", "ppo_validation_evaluation.json"),
        ("defender_aware", "ppo_validation_defender_evaluation.json"),
    ):
        payload = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
        case = next(row for row in payload["cases"] if row["qid"] == qid)
        runs[name] = {
            "trace": case["trace"],
            "final_document": case["final_document"],
            "retrieved": case["terminal"]["retrieved"],
            "attack_success": case["terminal"]["attack_success"],
            "defended_attack_success": case["terminal"]["defended_attack_success"],
            "attack_doc_detected": case["terminal"]["attack_doc_detected"],
        }
    return {"qid": qid, "split": "validation", "runs": runs}
