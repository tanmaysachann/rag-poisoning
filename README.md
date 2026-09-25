# Sentinel RAG

Sentinel RAG is a controlled, CPU-first research project on poisoned retrieval context. The original five-case Review-1 demo remains available; a separate benchmark and document-edit PPO experiment are under development. See `PROJECT_MEMORY.md` for the current implementation status and `IMPLEMENTATION_PLAN.md` for the full plan.
The ordered work for the next session is saved in `NEXT_SESSION.md`.
`V2_BENCHMARK_PROTOCOL.md` documents the disjoint two-passage development
benchmark and why its first consistency gate was not adopted.

`REPORT_EVIDENCE.md` maps measured claims to raw results and names the slide commitments that still need evidence. `LITERATURE_AUDIT.md` explains citation corrections, `PRESENTATION_REFERENCES_READY.md` provides 24 replacement entries, and `TEST_PROTOCOL_V1.md` records the frozen first test run.
`DEMO_SCRIPT.md` gives a seven-minute walkthrough of the live console and
separate research evidence.

Verify the frozen 75-question test snapshot with `python -m evaluation.verify_test_v1`.
Its hashes and exact case counts are recorded in `TEST_EVIDENCE_V1.json`.
`RUN_ENVIRONMENT.md` records the local CPU/software/model versions used for the
reported measurements.
The optional real-model SRQ study is `python -m evaluation.evaluate_slm_srq --help`;
it requires locally downloaded Qwen 0.5B weights and does not change the demo.
`python scripts/download_qwen_weight.py` fetches the pinned 988 MB weight and
matching tokenizer/config files from the official model repository, resumes
interrupted weight transfers, and checks the weight's SHA-256. They stay under
the ignored `artifacts/models/qwen2_5_0_5b/` directory.

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
python scripts/doctor.py
python -m unittest discover -s tests -v
```

To verify that the offline demo can rebuild from its checked-in data without
overwriting the current artifacts, run `python scripts/smoke_rebuild.py`. It builds
and tests a temporary project copy, then removes that copy.

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
7. Open Research to compare saved validation metrics and inspect per-question greedy/stealth defended and undefended traces for both answer profiles.

The five scenarios are query shortcuts, not answer templates. Every question searches only the 25 clean reports plus five poisoned reports in `data/demo_corpus.jsonl`. Nothing is fetched from Wikipedia or any other external source. A question unsupported by those 30 reports returns an insufficient-evidence response.

## Trusted ingest

An operator can preview document additions, changes, deletions, and the exact
corpus SHA-256 without modifying the protected manifest:

```powershell
python scripts/trusted_ingest.py data/demo_corpus.jsonl artifacts/demo_corpus_hashes.json
```

After reviewing the printed digest and content, seal that exact byte snapshot
with `--seal-reviewed-sha256 <digest>`. Replacing an existing manifest also
requires `--replace-existing-manifest`. Set `RAG_MANIFEST_KEY` outside the
repository to HMAC-sign it, and `RAG_REQUIRE_SIGNED_MANIFEST=1` in serving
when signed manifests are required. The attack staging harness never seals a
manifest. Sealing attests to a reviewed snapshot; it does not prove factual
truth or make accepted-ingest poisoning impossible.

Every predefined scenario is constructed to demonstrate both sides of the experiment: the poisoned payload wins extraction with Defense OFF, while Defense ON quarantines that same report and recovers the clean answer. This is enforced by regression tests for all five attacks. The relevance gate also checks the query's least-common corpus concept, preventing generic overlap such as "largest" and "Earth" from using an ocean report to answer a country question.

S1 is the Review-1 hidden-state proxy described in the presentation: MiniLM document embeddings are compared with the covariance-aware clean distribution using Mahalanobis distance. S4 performs an actual baseline plus leave-one-document-out extraction pass and measures whether removing one retrieved report changes the answer. Their calibrated risks participate directly in the quarantine decision; the remaining statistical and behavioural features support the fusion classifier.

## Research benchmark build

The five-case corpus remains the offline demo. To build a separate research
benchmark from revision-pinned MS MARCO v1.1 and NQ-Open data, run:

```powershell
python -m data.build_benchmark --pairs 500 --nq-queries 50 --output data/benchmark
python -m data.validate_benchmark data/benchmark
python scripts/index_benchmark.py
python -m evaluation.run_attack_baselines --split validation --strategy random --limit 20
python -m evaluation.run_attack_baselines --split validation --strategy greedy --limit 20
python -m evaluation.run_attack_baselines --split train --strategy random --limit 350
python -m evaluation.run_attack_baselines --split validation --strategy greedy --limit 75
python -m detect.train_research_detector
python -m evaluation.run_attack_baselines --split validation --strategy stealth --limit 75
python -m evaluation.evaluate_research_detector --split validation --strategy stealth
python -m evaluation.run_defense --split validation --strategy greedy --surface accepted_ingest
python -m evaluation.run_defense --split validation --strategy stealth --surface post_index_tamper
python -m evaluation.evaluate_clean_answers --split validation
python -m poison.evaluate_rollouts --split validation --strategy greedy_proxy --limit 75
python -m poison.train --batches 10 --episodes-per-batch 20
python -m poison.evaluate_policy --split validation --limit 75
python -m poison.train --defender-aware --replace-existing --batches 10 --episodes-per-batch 20
python -m poison.evaluate_policy --split validation --limit 75 --checkpoint artifacts/poison_policy_ppo_defender.pt
python scripts/index_benchmark.py --minilm --offline --splits validation --artifact-root artifacts/minilm_benchmark --output results/benchmark_retrieval_minilm_validation.json
python -m generation.train_sentence_ranker --minilm --offline --strict
python -m evaluation.evaluate_clean_answers --split validation --sentence-ranker --minilm --offline --strict
python -m evaluation.evaluate_clean_answers --split validation --sentence-ranker --minilm --offline --strict --remove-support
python -m evaluation.evaluate_model_probes
python -m evaluation.summarize_validation
```

The builder resolves and records the exact dataset revisions, writes disjoint
train/validation/test corpora and queries, and refuses to overwrite an existing
benchmark. Each MS MARCO query has a selected passage containing a normalized
answer alias. NQ-Open questions are a separate challenge set: an alias appearing
in the test corpus is recorded, but this alone is not treated as proof that the
question is answerable. The dataset download requires network access on the
first run.

The 200-episode CPU PPO run uses a 774-value hashing state and masked operation,
position, and payload heads. It is a research prototype, not the slide's
1,556-value state. On all 75 validation queries, PPO and a fixed one-insertion
baseline each caused the extractive answerer to return the wrong answer in
67 cases (89.3%); random edit rollouts succeeded in 27 cases (36%). PPO did not
improve on the simple baseline. These are undefended, same-corpus results.
The defender-aware PPO run freezes the hashing detector, edits the original
passage before accepted ingest, and rewards success after filtering. It
succeeded in 11/75 defended validation cases (14.7%), matching fixed
single-span substitution. The policy mainly chose `PARAPHRASE`.

The validation clean answerer matched an answer alias in 17/75 cases and
abstained in 36/75. The research detector catches the query-matched insertion
template but misses most subtle answer substitutions accepted at ingest. See
`results/defense_validation_*.json` and `results/ppo_validation_evaluation.json`
for the measured tradeoffs and per-case traces. A separate frozen test v1 run
is recorded in `TEST_PROTOCOL_V1.md` and `TEST_EVIDENCE_V1.json`.
`results/validation_intervals.json` adds Wilson 95% intervals and query-paired
bootstrap intervals for before/after reductions. They describe this development
validation set, not an unbiased final test.

With a locally cached MiniLM encoder, validation support recall@5 improved from
86.7% to 98.7% at higher query latency. A train-only learned sentence ranker
with MiniLM matched an answer alias in 41/75 validation cases (54.7%), versus
21/75 (28.0%) for the conservative extractor with the same retriever. After
the labeled support passage was withheld, that ranker still answered 7/75
cases; the conservative extractor answered 1/75. Other passages may still
contain answers, so this is a support-removal abstention stress test. The
ranker is experimental and is not wired into the defended answer path.

The same 75 greedy and 75 stealth attack documents were replayed with MiniLM
and the learned ranker. Greedy attack success was 41.3% undefended and 0%
defended, with 52.0% clean alias recovery. Stealth substitution succeeded in
28.0% of cases both before and after defense; only 5.3% of those attack
documents were quarantined. These results are in
`results/defense_validation_*_minilm_ranker.json`. The learned ranker remains
outside the live defense path because its subtle-attack behavior is unresolved.

The cached MiniLM encoder also supplies actual layer-3 hidden states and
attention tensors for a separate research probe. On validation, the
train-calibrated layer-distance score caught 0% of greedy and 6.7% of stealth
edits; attention deviation caught 0% and 4%. These are encoder signals, not
hidden states from the answer-generating model. They are not used to quarantine
documents. See `results/model_probes_validation.json`.

The local Qwen 0.5B model was run for SRQ and decoder-internal research probes.
With a threshold calibrated on 50 train-clean pairs, SRQ detected 0/75 greedy
and 4/75 stealth validation attacks while flagging 5/75 clean passages. A
separate decoder prefill probe detected at most 7/75 stealth attacks, with
6/75 clean passages flagged by that attention score. Neither signal is used
by the live gate. The local Qwen answer generator produced no valid cited
answer in a ten-question clean diagnostic. See `REPORT_EVIDENCE.md` for the
complete methods, limitations, and raw result paths.

The sealed v1 test matrix used MiniLM retrieval and the experimental sentence
ranker on 75 queries. Greedy attack success changed from 37/75 before defense
to 0/75 after, while stealth success changed from 23/75 to 22/75. Clean
alias match was 39/75. Run `python -m evaluation.verify_test_v1` to check
the stored hashes, paired case coverage, and summary counts. The test set is
spent for this v1 system; future experiments on it are exploratory.

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

The demo detector metrics come from a small controlled dataset and are not production generalization results. Both PPO implementations are CPU prototypes with one seed and tie their matched fixed-edit baselines. Real local-SLM SRQ, MiniLM encoder, and Qwen decoder prefill probes were measured but did not detect these attacks reliably. Validated local-LLM answer quality, multiple-model leave-one-out, broad attack-family coverage, and multi-seed policy comparisons remain outstanding.
