"""Dedicated V24 research scanner service for a large-function container."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import v24_service, v25_service

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Sentinel RAG V24 scanner", version="24.0")


class ScanRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    reference: str = Field(min_length=1, max_length=6000)
    candidate: str = Field(min_length=1, max_length=6000)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")


@app.get("/api/health")
def health() -> dict:
    return {"status": "operational", "version": "v24", "mode": "research_scanner"}


@app.get("/api/v24/status")
def status() -> dict:
    return v24_service.status()


@app.get("/api/v24/evidence")
def evidence() -> dict:
    try:
        return v24_service.evidence()
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="V24 evaluation reports are unavailable") from error


@app.get("/api/v24/sample")
def sample() -> dict:
    try:
        return v24_service.sample()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="Pinned V24 example is unavailable") from error


@app.post("/api/v24/scan")
def scan(body: ScanRequest) -> dict:
    try:
        result = v24_service.scan(body.query, body.reference, body.candidate,
                                  body.expected_sha256)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        if "required" in str(error):
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=503, detail="V24 scanner integrity check failed") from error
    return {"version": "v24", "split": "research_inference", **result}


@app.get("/api/v25/status")
def v25_status() -> dict:
    return v25_service.status()


@app.get("/api/v25/evidence")
def v25_evidence() -> dict:
    try:
        return v25_service.evidence()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="V25 evidence is unavailable") from error


@app.post("/api/v25/scan")
def v25_scan(body: ScanRequest) -> dict:
    try:
        result = v25_service.scan(body.query, body.reference, body.candidate,
                                  body.expected_sha256)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        if "required" in str(error):
            raise HTTPException(status_code=400, detail=str(error)) from error
        raise HTTPException(status_code=503, detail="V25 scanner integrity check failed") from error
    return {"split": "research_inference", **result}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "v24_standalone.html")


@app.get("/{asset:path}", include_in_schema=False)
def static_asset(asset: str) -> FileResponse:
    if asset not in {"styles.css", "research.css", "v24.css", "v24.js"}:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(ROOT / "frontend" / asset)
