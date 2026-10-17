"""Fail-closed serving path for operator-attested RAG corpora.

The research workbench is intentionally permissive so it can compare attacks.
This stricter path answers only from a corpus whose integrity manifest and
source-origin map are signed and bound to the exact corpus snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from detect.research_detector import ResearchDetector
from pipeline.research_inference import infer_research
from retrieval.hybrid_retriever import HybridRetriever
from security.four_probe_gate import FourProbeGate, load_bundle
from security.integrity import IntegrityManifest, load_manifest
from security.provenance import SourceAttestations, load_source_attestations


class SecureServiceConfigurationError(RuntimeError):
    """The operator has not supplied a complete, valid trust configuration."""


def secure_service_status() -> dict[str, Any]:
    """Report whether strict serving is configured without loading models."""
    required_files = {
        "corpus": os.environ.get("RAG_SECURE_CORPUS", "").strip(),
        "integrity_manifest": os.environ.get("RAG_SECURE_MANIFEST", "").strip(),
        "source_attestations": os.environ.get("RAG_SOURCE_ATTESTATIONS", "").strip(),
    }
    files_ready = {
        name: bool(value) and Path(value).expanduser().is_file()
        for name, value in required_files.items()
    }
    calibration_value = os.environ.get("RAG_FOUR_PROBE_CALIBRATION", "").strip()
    calibration_path = (
        Path(calibration_value).expanduser().resolve() if calibration_value
        else Path(__file__).resolve().parents[1] / "artifacts" / "four_probe_calibration.joblib"
    )
    calibration_ready = calibration_path.is_file()
    model_value = os.environ.get("RAG_LLM_MODEL", "").strip()
    qwen_weight_ready = bool(model_value) and (Path(model_value).expanduser() / "model.safetensors").is_file()
    model_matches_calibration = False
    if calibration_ready and model_value:
        summary_path = Path(__file__).resolve().parents[1] / "results" / "four_probe_calibration.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            model_matches_calibration = (
                Path(model_value).expanduser().resolve()
                == Path(summary["qwen_model"]).expanduser().resolve()
            )
        except (OSError, ValueError, KeyError, TypeError):
            model_matches_calibration = False
    checks = {
        "operator_signing_key": bool(os.environ.get("RAG_MANIFEST_KEY", "").strip()),
        **{f"{name}_file": ready for name, ready in files_ready.items()},
        "four_probe_calibration": calibration_ready,
        "hashing_retrieval": os.environ.get("RAG_USE_MINILM", "").strip().lower()
        in {"0", "false", "no"},
        "local_qwen_weights": qwen_weight_ready,
        "qwen_matches_calibration": model_matches_calibration,
    }
    missing = [name for name, ready in checks.items() if not ready]
    return {
        "ready": not missing,
        "mode": "strict-secure-query",
        "checks": checks,
        "missing": missing,
        "models_loaded": False,
    }


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SecureServiceConfigurationError(f"{name} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SecureServiceConfigurationError(f"{name} must point to an existing file")
    return path


def _optional_calibration_path() -> Path:
    value = os.environ.get("RAG_FOUR_PROBE_CALIBRATION", "").strip()
    path = Path(value).expanduser().resolve() if value else (
        Path(__file__).resolve().parents[1] / "artifacts" / "four_probe_calibration.joblib"
    )
    if not path.is_file():
        raise SecureServiceConfigurationError(
            "Four-probe calibration is missing; run python -m evaluation.build_four_probe_calibration"
        )
    return path


@lru_cache(maxsize=2)
def _load_service(corpus_path_text: str, manifest_path_text: str,
                  origins_path_text: str, key: str, corpus_file_digest: str,
                  manifest_file_digest: str, origins_file_digest: str,
                  calibration_path_text: str, calibration_file_digest: str
                  ) -> tuple[HybridRetriever, IntegrityManifest, ResearchDetector, SourceAttestations, FourProbeGate]:
    corpus_path = Path(corpus_path_text)
    manifest_path = Path(manifest_path_text)
    origins_path = Path(origins_path_text)
    calibration_path = Path(calibration_path_text)
    corpus_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if corpus_digest != corpus_file_digest:
        raise SecureServiceConfigurationError("Corpus changed while secure serving was loading")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_file_digest:
        raise SecureServiceConfigurationError("Integrity manifest changed while secure serving was loading")
    if hashlib.sha256(origins_path.read_bytes()).hexdigest() != origins_file_digest:
        raise SecureServiceConfigurationError("Source attestations changed while secure serving was loading")
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != calibration_file_digest:
        raise SecureServiceConfigurationError("Four-probe calibration changed while secure serving was loading")
    signing_key = key.encode("utf-8")

    try:
        manifest = load_manifest(manifest_path, signing_key=signing_key, require_signature=True)
        attestations = load_source_attestations(origins_path, signing_key=signing_key)
    except (OSError, ValueError) as error:
        raise SecureServiceConfigurationError("A signed trust manifest is invalid or unreadable") from error
    if not manifest.signature_verified or manifest.legacy:
        raise SecureServiceConfigurationError("A valid signed integrity manifest is required")
    if manifest.corpus_sha256 != corpus_digest:
        raise SecureServiceConfigurationError("Integrity manifest does not match the configured corpus")

    if attestations.corpus_sha256 != corpus_digest:
        raise SecureServiceConfigurationError("Source attestations do not match the configured corpus")

    retriever = HybridRetriever(corpus_path, integrity_path=manifest_path)
    if retriever.corpus_digest != corpus_digest:
        raise SecureServiceConfigurationError("Corpus changed while the secure index was loading")
    if retriever.embedder.backend != "hashing":
        raise SecureServiceConfigurationError(
            "The packaged secure detector currently requires hashing retrieval; unset RAG_USE_MINILM"
        )

    detector_path = Path(__file__).resolve().parents[1] / "artifacts" / "research_detector_hashing_web.joblib"
    if not detector_path.is_file():
        raise SecureServiceConfigurationError("The packaged hashing detector is unavailable")
    detector = joblib.load(detector_path)
    if not isinstance(detector, ResearchDetector) or detector.embedding_model != "sklearn-hashing-384":
        raise SecureServiceConfigurationError("The packaged detector has an incompatible embedding backend")

    known_ids = {str(int(doc["doc_id"])) for doc in retriever.documents}
    if set(manifest.hashes) != known_ids or set(attestations.documents) != known_ids:
        raise SecureServiceConfigurationError("Trust manifests must cover every corpus document exactly")
    try:
        probe_bundle = load_bundle(calibration_path)
        probe_gate = FourProbeGate(probe_bundle, local_files_only=True)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        raise SecureServiceConfigurationError(f"Four-probe gate could not load: {error}") from error
    if (
        hashlib.sha256(corpus_path.read_bytes()).hexdigest() != corpus_file_digest
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_file_digest
        or hashlib.sha256(origins_path.read_bytes()).hexdigest() != origins_file_digest
        or hashlib.sha256(calibration_path.read_bytes()).hexdigest() != calibration_file_digest
    ):
        raise SecureServiceConfigurationError("Corpus or trust manifests changed during initialization")
    return retriever, manifest, detector, attestations, probe_gate


def secure_query(query: str, *, top_k: int = 5) -> dict[str, Any]:
    """Answer only when integrity and signed independent-origin policy pass."""
    request_started = time.perf_counter()
    if not query.strip():
        raise ValueError("Query must not be empty")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    key = os.environ.get("RAG_MANIFEST_KEY", "")
    if not key:
        raise SecureServiceConfigurationError("RAG_MANIFEST_KEY is required for secure serving")

    corpus_path = _required_path("RAG_SECURE_CORPUS")
    manifest_path = _required_path("RAG_SECURE_MANIFEST")
    origins_path = _required_path("RAG_SOURCE_ATTESTATIONS")
    calibration_path = _optional_calibration_path()
    file_digests = tuple(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (corpus_path, manifest_path, origins_path, calibration_path)
    )
    cache_hits_before = _load_service.cache_info().hits
    initialization_started = time.perf_counter()
    retriever, manifest, detector, attestations, probe_gate = _load_service(
        str(corpus_path), str(manifest_path), str(origins_path), key,
        *file_digests[:3], str(calibration_path), file_digests[3],
    )
    initialization_ms = round((time.perf_counter() - initialization_started) * 1000, 2)
    was_cached = _load_service.cache_info().hits > cache_hits_before
    result = infer_research(
        query, retriever, manifest, detector, top_k=top_k,
        source_attestations=attestations, require_independent_origins=True,
        four_probe_gate=probe_gate,
    )
    # Do not expose a counterfactual undefended answer from a production route.
    return {
        "query": query,
        "answer": result["defended"],
        "provenance": result["provenance"],
        "four_probe": result["four_probe"],
        "documents": [
            {key: value for key, value in row.items() if key != "text"}
            for row in result["documents"]
        ],
        "counterfactuals": [
            {"removed_doc_id": row["removed_doc_id"],
             "answer_changed": row["answer_changed"]}
            for row in result["counterfactuals"]
        ],
        "timings": result["timings"],
        "latency_ms": round((time.perf_counter() - request_started) * 1000, 2),
        "inference_latency_ms": result["latency_ms"],
        "service_initialization_ms": initialization_ms,
        "service_was_cached": was_cached,
        "retrieval_backend": result["retrieval_backend"],
        "audit": result["audit"],
        "policy": {
            "integrity_manifest_signed": manifest.signature_verified,
            "source_attestations_signed": True,
            "independent_origins_required": 2,
            "four_probe_gate_enabled": True,
            "four_probe_gate_blocks_selected_answer_source": True,
        },
    }
