# Slide corrections: Presentation.pdf (38 slides) vs measured evidence

Source of truth: `REPORT_EVIDENCE.md`, `TEST_EVIDENCE_V1.json`,
`V2_BENCHMARK_PROTOCOL.md`. Every row tells the presenter what to keep,
correct, or remove before submission. Nothing here evaluates the v2 test split.

## Slides 1-5 (problem, attack/defense objectives) — KEEP

Threat model, attacker control, defender view, and success conditions are
implemented as stated and frozen in `TEST_PROTOCOL_V1.md`. Attack success is
reported per goal and split into unconditional vs conditional-on-retrieval.

## Slide 6 (SHA-256 integrity, N+1 leave-one-out) — CORRECT WORDING

- KEEP: versioned hash manifest, mismatch quarantine, reviewed-digest trusted
  ingest (`scripts/trusted_ingest.py`), extractive leave-one-out trace with
  logged N+1 cost (`pipeline/research_inference.py`, `pipeline/secure_rag.py`).
- CORRECT: hashing is tamper evidence against a protected manifest only; it
  does not detect accepted-ingest poisoning. LOO is selective/diagnostic, not
  full N+1 LLM inference on every query, and influence alone never quarantines.
- Evidence: `results/defense_validation_*`, sealed v1 test (greedy 37/75 → 0/75).

## Slide 7 (SLM Semantic Regurgitation Quotient) — CORRECT TO NEGATIVE RESULT

- The formula, pinned Qwen2.5-0.5B implementation, 50-case train calibration,
  and 75-case validation runs are complete. Result: 0/75 greedy recall, 4/75
  stealth recall at 6.7% clean FPR (`results/slm_srq_validation_50_75.json`).
- Slides must present SRQ as measured-but-ineffective, never as a serving
  gate. Greedy insertion lowered mean SRQ (0.481 → 0.338), contradicting the
  high-score-risk assumption.

## Slide 8 (SDGs 4, 9, 16, 17) — KEEP AS DISCUSSION ONLY

No engineering claim. Keep as impact discussion.

## Slides 9-14, 35-37 (literature) — REPLACE REFERENCES

Do not present the printed reference list. Use the 24 verified replacement
entries in `PRESENTATION_REFERENCES_READY.md`; see `LITERATURE_AUDIT.md` for
the swapped-arXiv-ID chain and the three missing slide-14 sources. The
Information Fusion article is about knowledge-graph RAG specifically.

## Slides 16-18 (MS MARCO + NQ-Open) — KEEP WITH NUMBERS

- v1: 350 train / 75 validation / 75 sealed test MS MARCO pairs plus 50 NQ-Open
  challenge questions (`data/benchmark/manifest.json`).
- v2 (disjoint, two-passage): 140 / 30 / 30 with test untouched
  (`data/benchmark_v2/manifest.json`).
- NQ-Open caveat stays: 44/50 challenge aliases are absent from the 75-passage
  validation corpus, so alias absence is a lexical stress condition, not proof
  of unsupported answers.

## Slides 19, 21, 23-27 (MDP, policy, cache, reward, PPO) — CORRECT NUMBERS

- State is **774-dimensional** (`STATE_VERSION=edit-mdp-v3-hashing-774`), not
  the unexplained 1,556-dimensional slide target. Correct the slide or show
  the checked schema that sums to 774.
- Actions `INSERT/PARAPHRASE/SYNONYM/DELETE/STOP`, masked factored policy with
  two value heads, terminal retrieval/answer reward, and inspectable edit
  cache are implemented (`poison/`).
- Results must be reported honestly: undefended PPO ties greedy insertion on
  all three seeds (197/225 vs 197/225,
  `results/ppo_multiseed_undefended_validation.json`); defender-aware PPO ties
  fixed substitution (31/225 vs 34/225, `results/ppo_multiseed_validation.json`).
  No PPO advantage is claimed. Measured ablations: detector-risk shaping
  (+8.9pp when present), proxy-value loss (no effect), edit-effect cache
  (no effect), action-head conditioning (no effect).
- PPO loss sign in slide 27 needs correction for a minimization optimizer
  (implemented correctly in `poison/ppo.py`).

## Slides 20, 22, 28-30 (secure pipeline, serving, UI) — CORRECT SCOPE

- KEEP: staged pipeline (retrieval → integrity → cheap signals → calibrated
  classifier → selective LOO → accepted-only generation → citation validation →
  audit trace), trusted-ingest CLI, live validation workbench, sealed-test
  aggregate console.
- CORRECT: the deployed lab runs the **hashing/extractive validation profile**;
  MiniLM, Qwen, and PPO figures are offline artifacts, not live backends.
  MiniLM encoder and Qwen decoder-prefill probes plus model-backed SRQ were
  measured but failed as standalone detectors (0-9.3% recall range) and are not
  live gates. The MiniLM sentence ranker is experimental: it raises clean
  utility (v1 test 39/75) but does not stop stealth substitution (23/75 → 22/75).
- Overt non-factual families (instruction/URL/refusal) are quarantined 75/75
  with 71/70/72 → 0 defended successes
  (`results/payload_families_validation.json`); subtle accepted-ingest
  substitution remains the open problem (stealth 22/75 defended successes).

## Slides 31-34 (models, retrieval, storage, compute) — CORRECT MODEL NAMES

- Measured: MiniLM-L6-v2 (cached), hashing-384 fallback, BM25+RRF, NumPy exact
  search; Contriever-msmarco (pinned revision, CPU 25.69 ms/query) and FAISS
  FlatIP comparisons are offline-only; serving stays MiniLM/NumPy.
- **Llama-3-8B NF4 has never run** on this CPU-only host; the measured local
  generator is Qwen2.5-0.5B-Instruct (pinned SHA-256 in `RUN_ENVIRONMENT.md`),
  and it stays outside serving (20Q diagnostic: 5/20 span-grounded, 1/20 alias
  match). Do not label extractive outputs as LLM generation, and do not promise
  a specific Kaggle GPU.

## Slide 38 (final demo) — KEEP WITH SEPARATION

`DEMO_SCRIPT.md` seven-minute walkthrough: five-case demo, validation
experiments, and sealed test figures kept distinct. Offline rebuild via
`python scripts/smoke_rebuild.py`; full check via
`python scripts/verify_release.py`.
