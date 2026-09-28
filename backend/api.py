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
        "ppo_detection_ablation": "ppo_detection_reward_ablation_validation.json",
        "ppo_proxy_ablation": "ppo_proxy_value_ablation_validation.json",
        "ppo_cache_ablation": "ppo_cache_ablation_validation.json",
        "ppo_head_ablation": "ppo_head_conditioning_ablation_validation.json",
        "qwen_span_repair": "local_generation_answer_repair_validation_20.json",
        "faiss_scale": "faiss_exact_scale_validation.json",
        "unseen_templates": "unseen_templates_validation.json",
        "poison_budget": "poison_budget_stealth_validation.json",
        "fixed_baseline": "edit_rollout_validation_greedy_proxy.summary.json",
        "v2_clean": "multisupport_v2_validation.json",
        "v2_paired_gate": "paired_gate_v2_validation_full_context.json",
        "v2_pair_classifier": "pairwise_detector_v2_validation.json",
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
