"""Local API and static web server for the Review-1 secure RAG demo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import POISONED_DOCS_PATH
from backend.research_lab import case_catalog, ppo_case, ppo_history, run_case
from backend.secure_service import SecureServiceConfigurationError, secure_query, secure_service_status
from pipeline.secure_rag import secure_rag_answer

app = FastAPI(title="Sentinel RAG", version="0.6.0")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    defense_enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    simulate_tamper_doc_id: int | None = None


class ResearchRunRequest(BaseModel):
    qid: str = Field(min_length=1, max_length=60)
    strategy: str = Field(pattern="^(greedy|stealth)$")
    surface: str = Field(pattern="^(accepted_ingest|post_index_tamper)$")
    attack_text: str | None = Field(default=None, max_length=4000)
    attack_budget: int = Field(default=1, ge=1, le=3)
    additional_attack_texts: list[str] | None = Field(default=None, max_length=2)


class SecureQueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/api/health")
def health() -> dict:
    return {"status": "operational", "mode": "research-prototype", "version": "0.6.0",
            "retrieval_scope": "closed-corpus-only"}


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return [{"id": doc["doc_id"], "query": doc["target_query"],
             "attack_type": doc["attack_type"], "operations": doc["operations_applied"],
             "injected_claim": doc["injected_claim"],
             "report_url": f"/reports/doc_{doc['doc_id']}_report.pdf"}
            for doc in _jsonl(POISONED_DOCS_PATH)]


@app.get("/api/research-summary")
def research_summary() -> dict:
    """Return allowlisted local validation summaries; never start training in a request."""
    names = {
        "clean": "clean_answers_validation.json",
        "clean_minilm": "clean_answers_validation_minilm.json",
        "clean_ranker": "clean_answers_validation_minilm_ranker_strict.json",
        "clean_ranker_support_removed": "clean_answers_validation_minilm_ranker_strict_no_support.json",
        "retrieval_hashing": "benchmark_retrieval.json",
        "retrieval_minilm": "benchmark_retrieval_minilm_validation.json",
        "retrieval_contriever": "benchmark_retrieval_contriever_validation.json",
        "detector_greedy": "research_detector_validation.json",
        "detector_stealth": "research_detector_validation_stealth.json",
        "encoder_probes": "model_probes_validation.json",
        "decoder_probes": "decoder_probes_validation.json",
        "slm_srq": "slm_srq_validation_50_75.json",
        "nq_challenge": "nq_challenge_validation_minilm_ranker.json",
        "local_generation": "local_generation_validation_10_raw.json",
        "defense_greedy": "defense_validation_greedy_accepted_ingest.json",
        "defense_stealth": "defense_validation_stealth_accepted_ingest.json",
        "defense_greedy_minilm": "defense_validation_greedy_accepted_ingest_minilm_ranker.json",
        "defense_stealth_minilm": "defense_validation_stealth_accepted_ingest_minilm_ranker.json",
        "ppo": "ppo_validation_evaluation.json",
        "ppo_defender": "ppo_validation_defender_evaluation.json",
        "ppo_multiseed": "ppo_multiseed_validation.json",
        "ppo_multiseed_undefended": "ppo_multiseed_undefended_validation.json",
        "ppo_detection_ablation": "ppo_detection_reward_ablation_validation.json",
        "ppo_proxy_ablation": "ppo_proxy_value_ablation_validation.json",
        "ppo_cache_ablation": "ppo_cache_ablation_validation.json",
        "ppo_head_ablation": "ppo_head_conditioning_ablation_validation.json",
        "qwen_span_repair": "local_generation_answer_repair_validation_20.json",
        "faiss_scale": "faiss_exact_scale_validation.json",
        "unseen_templates": "unseen_templates_validation.json",
        "poison_budget": "poison_budget_stealth_validation.json",
        "payload_families": "payload_families_validation.json",
        "fixed_baseline": "edit_rollout_validation_greedy_proxy.summary.json",
        "v2_clean": "multisupport_v2_validation.json",
        "v2_paired_gate": "paired_gate_v2_validation_full_context.json",
        "v2_pair_classifier": "pairwise_detector_v2_validation.json",
        "v2_corroboration": "answer_corroboration_v2_validation.json",
        "v2_corroboration_train": "answer_corroboration_v2_train_calibration.json",
        "v2_answer_key": "answer_key_corroboration_v2_validation.json",
        "v2_answer_key_train": "answer_key_corroboration_v2_train_calibration.json",
        "cited_extractive": "cited_extractive_validation_20.json",
        "perplexity": "perplexity_validation.json",
        "v3_retrieval": "v3_benchmark_retrieval.json",
        "v3_detector_greedy": "research_detector_v3_validation.json",
        "v3_detector_stealth": "research_detector_v3_validation_stealth.json",
        "v3_defense_greedy": "v3_defense_validation_greedy_accepted_ingest.json",
        "v3_defense_stealth": "v3_defense_validation_stealth_accepted_ingest.json",
        "v3_defense_greedy_minilm": "v3_defense_validation_greedy_accepted_ingest_minilm_ranker.json",
        "v3_defense_stealth_minilm": "v3_defense_validation_stealth_accepted_ingest_minilm_ranker.json",
        "v3_clean_minilm": "clean_answers_validation_minilm_ranker_strict_v3.json",
    }
    summaries = {}
    for key, filename in names.items():
        path = ROOT / "results" / filename
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary", payload)
            summaries[key] = {name: value for name, value in summary.items()
                              if name not in {"checkpoint", "output"}}
    return {"available": bool(summaries), "split": "validation", "test_split_evaluated": True,
            "summaries": summaries}


@app.get("/api/test-summary")
def test_summary() -> dict:
    """Return only sealed aggregate test evidence, never raw test cases."""
    if not (ROOT / "TEST_EVIDENCE_V1.json").is_file():
        return {"available": False, "split": "test", "protocol": "v1"}
    from evaluation.verify_test_v1 import verify

    try:
        verify(ROOT)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=409, detail="Frozen test evidence failed verification") from error
    names = {
        "clean": "clean_answers_test_minilm_ranker_strict.json",
        "greedy": "defense_test_greedy_accepted_ingest_minilm_ranker.json",
        "stealth": "defense_test_stealth_accepted_ingest_minilm_ranker.json",
        "intervals": "test_intervals_v1.json",
    }
    summaries = {key: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
                 for key, name in names.items()}
    return {"available": True, "split": "test", "protocol": "v1", "summaries": summaries}


@app.get("/api/research-cases")
def research_cases(strategy: str = "greedy", profile: str = "hashing") -> dict:
    """Read allowlisted validation traces for the local presentation console."""
    if strategy not in {"greedy", "stealth"} or profile not in {"hashing", "minilm_ranker"}:
        raise HTTPException(status_code=400, detail="Unknown research case selection")
    suffix = "_minilm_ranker" if profile == "minilm_ranker" else ""
    cases_path = ROOT / "results" / f"defense_validation_{strategy}_accepted_ingest{suffix}.cases.jsonl"
    queries_path = ROOT / "data" / "benchmark" / "validation" / "queries.jsonl"
    if not cases_path.is_file() or not queries_path.is_file():
        raise HTTPException(status_code=404, detail="Saved validation cases are unavailable")
    questions = {row["qid"]: row["question"] for row in _jsonl(queries_path)}
    fields = (
        "qid", "attack_doc_id", "attack_top5_rank", "attack_integrity_status",
        "attack_quarantined", "undefended_answer", "undefended_source",
        "defended_answer", "defended_source", "undefended_attack_success",
        "defended_attack_success", "clean_answer_recovered", "filtered_doc_ids",
        "loo_triggered", "loo_changed_answer", "latency_ms",
    )
    rows = [{**{key: row.get(key) for key in fields},
             "question": questions.get(row["qid"], "Unknown question")}
            for row in _jsonl(cases_path)]
    return {"split": "validation", "strategy": strategy, "profile": profile, "cases": rows}


@app.get("/api/lab/cases")
def lab_cases() -> dict:
    return case_catalog()


@app.post("/api/lab/run")
def lab_run(body: ResearchRunRequest) -> dict:
    try:
        return run_case(body.qid, body.strategy, body.surface, body.attack_text,
                        attack_budget=body.attack_budget,
                        additional_attack_texts=body.additional_attack_texts)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/lab/ppo")
def lab_ppo() -> dict:
    return ppo_history()


@app.get("/api/lab/ppo-case")
def lab_ppo_case(qid: str) -> dict:
    try:
        return ppo_case(qid)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/lab/defender-rl")
def lab_defender_rl() -> dict:
    """Expose the aggregate offline defender PPO comparison, including failure."""
    path = ROOT / "results" / "defender_bandit_v14.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Defender RL aggregate is unavailable")
    report = json.loads(path.read_text(encoding="utf-8"))
    models = {}
    for name, row in report["models"].items():
        models[name] = {
            "clean_flags": row["clean"]["false_flags"],
            "clean_documents": row["clean"]["documents"],
            "poisonedrag_caught": row["families"]["poisonedrag_black"]["caught"],
            "poisonedrag_documents": row["families"]["poisonedrag_black"]["documents"],
            "gaslite_caught": row["families"]["gaslite"]["caught"],
            "gaslite_documents": row["families"]["gaslite"]["documents"],
        }
    return {"split": "previously_inspected_development", "live": False,
            "training_rows": report["train_rows"], "models": models,
            "ppo_sampled_actions_per_seed": report["training_log"]["ppo_seed_11"]["sampled_actions"],
            "gate_passed": report["gate"]["passed"],
            "limitation": "Source-labelled offline passage screening, not a live SLM fine-tune or end-to-end defense."}


@app.get("/api/lab/snapshot-recovery")
def lab_snapshot_recovery() -> dict:
    """Expose aggregate v16 development outcomes without replaying during a request."""
    path = ROOT / "results" / "snapshot_recovery_v16.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot recovery aggregate is unavailable")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {"split": report["split"], "clean": report["clean"],
            "attack_replays": report["attack_replays"],
            "snapshot_corpus_sha256": report["snapshot_corpus_sha256"],
            "limitation": report["limitations"][0]}


@app.get("/api/lab/full-corpus")
def lab_full_corpus() -> dict:
    """Expose frozen offline scale results, including the failed clean gate."""
    paths = {
        name: ROOT / "results" / filename for name, filename in {
            "v17": "full_corpus_v17.json",
            "diagnostic": "full_corpus_v17_diagnostics.json",
            "v18": "answer_selector_v18.json",
            "v19": "answer_selector_v19.json",
            "v19_training": "answer_selector_v19_training.json",
            "v19_diagnostic": "answer_selector_v19_diagnostics.json",
        }.items()
    }
    if any(not path.is_file() for path in paths.values()):
        raise HTTPException(status_code=404, detail="Full-corpus aggregate is unavailable")
    reports = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in paths.items()}
    v17, v18, v19 = reports["v17"], reports["v18"], reports["v19"]
    return {
        "live": False,
        "scope": "public Natural Questions clean-answer evaluation; no poison replay",
        "corpus_passages": v17["beir_passages"],
        "questions_per_evaluation": 500,
        "v17": v17["evaluation"],
        "v17_dense_qrel_at_10": v17["retrieval"]["evaluation"]["dense"]["qrel_at_10"],
        "v17_reader_exact_candidate": reports["diagnostic"]["counts"][
            "reader_exact_candidate_in_dense_top_10"],
        "v18_baseline": v18["baseline_v17_frozen_policy"],
        "v18_learned": v18["learned_v18_selector"],
        "v17_gate_passed": v17["gate_passed"],
        "v18_gate_passed": v18["gate_passed"],
        "v19_baseline": v19["baseline_v17_frozen_policy"],
        "v19_trained": v19["trained_v19_selector"],
        "v19_training_questions": (reports["v19_training"]["nq_open_train_questions"]
                                   + reports["v19_training"]["spent_v17_questions"]),
        "v19_training_candidates": reports["v19_training"]["training_candidates"],
        "v19_reader_exact_candidate": v19["reader_exact_candidate_questions"],
        "v19_unthresholded_top_candidate_exact": reports["v19_diagnostic"][
            "unthresholded_model_top_candidate_exact"],
        "v19_gate_passed": v19["gate_passed"],
        "v19_unverified_emissions": v19["unverified_emissions"],
        "unverified_emissions": v18["unverified_emissions"],
        "limitation": "Public benchmark clean QA only; the live 150-passage guard has not been validated at this scale.",
    }


@app.post("/api/analyze")
def analyze(body: QueryRequest) -> dict:
    try:
        result = secure_rag_answer(
            body.query, body.defense_enabled, body.threshold, body.simulate_tamper_doc_id
        )
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    for doc in result["kept_docs"] + result["filtered_docs"]:
        report = ROOT / "output" / "pdf" / f"doc_{doc['doc_id']}_report.pdf"
        if report.exists(): doc["report_url"] = f"/reports/{report.name}"
    return {**result, "query": body.query, "defense_enabled": body.defense_enabled,
            "threshold": body.threshold,
            "stats": {"retrieved": len(result["kept_docs"]) + len(result["filtered_docs"]),
                      "kept": len(result["kept_docs"]), "filtered": len(result["filtered_docs"]),
                      "threats": sum(
                          detail["decision"] == "quarantine"
                          for detail in result["score_details"].values()
                      ),
                      "integrity_percent": round(
                          100 * sum(
                              doc["integrity"]["status"] == "verified"
                              for doc in result["kept_docs"] + result["filtered_docs"]
                          ) / max(len(result["kept_docs"]) + len(result["filtered_docs"]), 1)
                      )}}


@app.post("/api/secure-query")
def secure_analyze(body: SecureQueryRequest) -> dict:
    """Serve from an operator-signed corpus and require independent origins."""
    try:
        return secure_query(body.query, top_k=body.top_k)
    except SecureServiceConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/secure-query/status")
def secure_query_status() -> dict:
    """Return safe configuration flags for the secure-query browser view."""
    return secure_service_status()


@app.get("/api/generalization-summary")
def generalization_summary() -> dict:
    """Return the frozen v3 held-out aggregate only after artifact verification."""
    from evaluation.verify_v3_generalization import verify

    try:
        proof = verify()
        path = ROOT / "results" / "v3_generalization_test.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=409, detail="Frozen generalization evidence failed verification") from error
    return {"verification": proof, "summary": summary}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/{asset:path}", include_in_schema=False)
def assets(asset: str) -> FileResponse:
    def contained(base: Path, relative: str) -> Path | None:
        root = base.resolve()
        target = (root / relative).resolve()
        return target if target == root or root in target.parents else None

    if asset.startswith("reports/"):
        path = contained(ROOT / "output" / "pdf", asset.removeprefix("reports/"))
        if path is not None and path.is_file() and path.suffix.lower() == ".pdf":
            return FileResponse(path, media_type="application/pdf")
        raise HTTPException(status_code=404, detail="Report not found")
    path = contained(ROOT / "frontend", asset)
    if path is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path) if path.is_file() else FileResponse(ROOT / "frontend" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=False)
