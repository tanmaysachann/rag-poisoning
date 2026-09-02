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
- Strict closed-corpus retrieval: no web search, external API, or fallback knowledge source
- Explicit S1 Mahalanobis geometry probe and S4 leave-one-out stability probe

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

Run the dynamic-pipeline regression suite with:

```powershell
python -m unittest discover -s tests -v
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
3. Compare the result against Defense OFF in the counterfactual panel. Both runs retrieve the same five candidates from the same 30-document corpus and use the same sentence extractor. Defense ON applies the S1/S4 probe gate and removes documents whose fused risk crosses the threshold.
4. Enable post-index tampering to demonstrate SHA-256 mismatch detection.
5. Open Evaluation to show LOOCV metrics and saved plots.
6. Open Architecture to explain the implemented zero-trust stages.

The five scenarios are query shortcuts, not answer templates. Every question searches only the 25 clean reports plus five poisoned reports in `data/demo_corpus.jsonl`. Nothing is fetched from Wikipedia or any other external source. A question unsupported by those 30 reports returns an insufficient-evidence response.

S1 is the Review-1 hidden-state proxy described in the presentation: MiniLM document embeddings are compared with the covariance-aware clean distribution using Mahalanobis distance. S4 performs an actual baseline plus leave-one-document-out extraction pass and measures whether removing one retrieved report changes the answer. Their calibrated risks participate directly in the quarantine decision; the remaining statistical and behavioural features support the fusion classifier.

## Offline use

The hosted Vercel build uses deterministic 384-dimensional hashing embeddings because the complete PyTorch/MiniLM runtime is too large for a small serverless function. The local review build supports the real `sentence-transformers/all-MiniLM-L6-v2` encoder and reports the active backend directly under the **RETRIEVED** metric.

After installing `requirements-dev.txt`, build and run the MiniLM version with:

```powershell
$env:RAG_USE_MINILM="1"
python scripts/build_index.py --reuse-data
python scripts/inject_poison.py
python -m detect.train_fusion_classifier
python backend/api.py
```

The first command that loads MiniLM may download the model once. After it is cached, add `$env:HF_HUB_OFFLINE="1"` and `$env:TRANSFORMERS_OFFLINE="1"` for a fully offline demonstration. Keep `RAG_USE_MINILM=1` set in the terminal used to start the API so the index, detector, and query vectors all use the same embedding space.

For the zero-download fallback, remove that environment variable and rerun the three artifact-building commands. Separate backend-specific classifier artifacts prevent a hashing classifier from being mixed with MiniLM features.

## Honest limitation

The reported detector metrics come from a small controlled dataset and should not be presented as production generalization. RL-based attack generation, large-scale evaluation, an SLM-backed SRQ implementation, and multiple LLM leave-one-out generations are Phase-2 work.
