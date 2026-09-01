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

from config import BASE_CORPUS_PATH, POISONED_DOCS_PATH, RESULTS_DIR
from pipeline.secure_rag import secure_rag_answer

app = FastAPI(title="Sentinel RAG", version="0.3.0")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    defense_enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    simulate_tamper: bool = False
    live_retrieval: bool = True


@app.get("/api/health")
def health() -> dict:
    return {"status": "operational", "mode": "review-1-mvp", "version": "0.3.0"}


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return [{"id": doc["doc_id"], "query": doc["target_query"],
             "attack_type": doc["attack_type"], "operations": doc["operations_applied"],
             "injected_claim": doc["injected_claim"],
             "report_url": f"/reports/doc_{doc['doc_id']}_report.pdf"}
            for doc in _jsonl(POISONED_DOCS_PATH)]


@app.get("/api/evaluation")
def evaluation() -> dict:
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {"metrics": metrics, "corpus": {"clean": len(_jsonl(BASE_CORPUS_PATH)),
            "poisoned": len(_jsonl(POISONED_DOCS_PATH))},
            "artifacts": {"roc_curve": "/results/roc_curve.png",
                          "confusion_matrix": "/results/confusion_matrix.png",
                          "dossier": "/reports/sentinel_rag_poisoned_corpus_dossier.pdf"}}


@app.post("/api/analyze")
def analyze(body: QueryRequest) -> dict:
    try:
        result = secure_rag_answer(body.query, body.defense_enabled, body.threshold,
                                   body.simulate_tamper, body.live_retrieval)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    for doc in result["kept_docs"] + result["filtered_docs"]:
        report = ROOT / "output" / "pdf" / f"doc_{doc['doc_id']}_report.pdf"
        if report.exists(): doc["report_url"] = f"/reports/{report.name}"
    tampered = sum(result["tamper_flags"].values())
    return {**result, "query": body.query, "defense_enabled": body.defense_enabled,
            "threshold": body.threshold,
            "stats": {"retrieved": len(result["kept_docs"]) + len(result["filtered_docs"]),
                      "kept": len(result["kept_docs"]), "filtered": len(result["filtered_docs"]),
                      "threats": sum(value >= body.threshold for value in result["scores"].values()),
                      "tampered": tampered, "integrity_percent": 100 - tampered * 20,
                      "live_sources": sum(bool(doc.get("live_source")) for doc in
                                          result["kept_docs"] + result["filtered_docs"])}}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/{asset:path}", include_in_schema=False)
def assets(asset: str) -> FileResponse:
    if asset.startswith("reports/"):
        path = ROOT / "output" / "pdf" / asset.removeprefix("reports/")
        if path.exists() and path.is_file(): return FileResponse(path, media_type="application/pdf")
    if asset.startswith("results/"):
        path = RESULTS_DIR / asset.removeprefix("results/")
        if path.exists() and path.is_file(): return FileResponse(path)
    path = ROOT / "frontend" / asset
    return FileResponse(path) if path.exists() and path.is_file() else FileResponse(ROOT / "frontend" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=False)
