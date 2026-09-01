# Sentinel RAG - Review-1 MVP

Sentinel RAG is a controlled, CPU-first demonstration of poisoning detection and zero-trust context processing for Retrieval-Augmented Generation. Review-1 intentionally excludes PPO/RL training while implementing the retrieval, integrity, detection, filtering, evaluation, and presentation layers end to end.

## Implemented scope

- 25 clean reference documents and 5 detailed poisoned PDF reports
- BM25 + dense retrieval fused with Reciprocal Rank Fusion
- SHA-256 integrity manifest with post-index tamper simulation
- Seven explainable signals: Mahalanobis distance, Isolation Forest, relevance spike, instruction pattern, URL pattern, authority cue, and leave-one-out influence
- Standardized logistic-regression fusion classifier
- Leave-one-out cross-validation with saved ROC and confusion-matrix figures
- Zero-trust prompt isolation and deterministic extractive answer backend
- FastAPI security console with defended/undefended comparison
- Dynamic Wikipedia retrieval for free-text questions, with source links and evidence-based abstention

## Setup

```powershell
cd "C:\Users\tanma\OneDrive\Desktop\Major Project\rag-poisoning"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Build all Review-1 artifacts

```powershell
python scripts/build_index.py --reuse-data
python scripts/inject_poison.py
python -m detect.train_fusion_classifier
python scripts/build_pdf_reports.py
```

`scripts/inject_poison.py` prints the top-5 retrieval rank of every poisoned report. Classifier training writes `results/metrics.json`, `results/roc_curve.png`, `results/confusion_matrix.png`, and `artifacts/fusion_classifier.joblib`.

## Run the console

```powershell
python backend/api.py
```

Open `http://127.0.0.1:8000`. If that port is already in use, stop the older process with `Ctrl+C` before restarting.

## Vercel deployment

The repository includes `api/index.py` and `vercel.json` for Vercel's Python serverless runtime. `requirements.txt` contains only runtime dependencies; corpus-building and PDF-generation packages live in `requirements-dev.txt`.

```powershell
vercel.cmd
vercel.cmd --prod
```

## Presentation flow

1. Select one of the five attack scenarios.
2. Run the pipeline with Defense ON and inspect the quarantined document, signal bars, reason codes, integrity hash, and trusted evidence sentence.
3. Compare the result against Defense OFF in the counterfactual panel.
4. Enable post-index tampering to demonstrate SHA-256 mismatch detection.
5. Open Evaluation to show LOOCV metrics and saved plots.
6. Open Architecture to explain the implemented zero-trust stages.

For controlled attack scenarios, live retrieval is disabled automatically so the experiment remains reproducible. Editing the query enables live Wikipedia retrieval; unsupported questions return an insufficient-evidence response instead of an unrelated corpus answer.

## Offline use

The default embedding backend is deterministic hashing, so the prepared demo does not need a model download. Set `RAG_USE_MINILM=1` only when `sentence-transformers/all-MiniLM-L6-v2` is already cached or internet access is available, then rebuild the indexes and retrain the classifier with that same backend.

## Honest limitation

The reported detector metrics come from a small controlled dataset and should not be presented as production generalization. RL-based attack generation, large-scale evaluation, an SLM-backed SRQ implementation, and multiple LLM leave-one-out generations are Phase-2 work.
