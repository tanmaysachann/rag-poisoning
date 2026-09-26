# Sentinel RAG research evidence (frozen test v1 checkpoint)

This evidence map supports the 38-slide `Presentation.pdf`. It includes a
first frozen 75-question held-out test run for the MiniLM/ranker/hashed-detector
pipeline, plus separate validation-stage development results. The five-case
Review-1 demo, MS MARCO/NQ-Open benchmark, and PPO attack experiment are
separate profiles. The source audit is in `LITERATURE_AUDIT.md`.

## Dataset and experimental boundary

- The frozen benchmark has 350 training, 75 validation, and 75 test MS MARCO
  query/passage pairs. Each pair has one selected passage with a normalized
  answer-token span. The separate NQ-Open challenge has 50 questions; lexical
  alias presence does not establish answerability.
- Dataset revisions, row counts, hashes, and selection method are in
  `data/benchmark/manifest.json`. `python -m data.validate_benchmark
  data/benchmark` checks split disjointness and file hashes.
- The validation set was used for model comparison and limitation finding.
  Its numbers are descriptive. `TEST_PROTOCOL_V1.md` froze code, data, artifact
  hashes, threat conditions, and primary outcomes before the first test run.
  The test result now describes that frozen v1 system; any subsequent change
  evaluated on the same test data is exploratory, not a new untouched test.
- The attack boundary is a self-owned local corpus. `accepted_ingest` means
  malicious text is sealed as a new trusted snapshot, so a hash does not prove
  factual truth. `post_index_tamper` means content changes after the protected
  clean snapshot; integrity should catch it when retrieved.

## Reproducible findings

### Frozen v1 test result

| Measure | Test result (75 questions per condition) | Raw evidence |
|---|---|---|
| Clean MiniLM retrieval | Support recall@1 73/75 (97.3%); @5 75/75 (100%) | `results/benchmark_retrieval_minilm_test.json` |
| Clean strict ranker | Known answer-alias match 39/75 (52.0%); selected labeled support 75/75; abstained 0/75 | `results/clean_answers_test_minilm_ranker_strict.json` and `.cases.jsonl` |
| Greedy accepted-ingest attack | 37/75 (49.3%) attack success before defense, 0/75 after; 75/75 attack documents quarantined; 34/75 (45.3%) clean aliases recovered | `results/defense_test_greedy_accepted_ingest_minilm_ranker.json` and `.cases.jsonl` |
| Stealth answer substitution | 23/75 (30.7%) before defense, 22/75 (29.3%) after; 3/75 (4.0%) attack documents quarantined; 1/75 (1.3%) clean aliases recovered | `results/defense_test_stealth_accepted_ingest_minilm_ranker.json` and `.cases.jsonl` |
| Interval summary | Greedy paired success reduction 49.3 percentage points, bootstrap 95% interval 38.7–61.3; stealth reduction 1.3 points, interval 0–4.0 | `results/test_intervals_v1.json` |

These are local 75-document tests with one train-pool wrong answer per query,
not estimates for arbitrary web corpora or attacks. MiniLM uses a cached encoder
and the answerer is an extractive sentence ranker, not the slide's 8B generator.
Median per-case time was about 0.90 seconds and includes rebuilding an isolated
attack index, so it is not serving latency. A correct alias may be absent even
when a selected support sentence is relevant; conversely an alias match alone
does not prove the answer is truthful.

`TEST_EVIDENCE_V1.json` records input and output SHA-256 hashes and exact counts.
`python -m evaluation.verify_test_v1` verified the snapshot, paired query coverage,
and agreement between per-case records, summaries, and reported counts.
`python -m scripts.build_research_figures` generated the visually checked
`output/research_figures/test_security_v1.png` and vector PDF from that sealed
snapshot.

### Validation development results

| Claim | Validation evidence | Raw files / command |
|---|---|---|
| Hashing hybrid retrieval | Support recall@1 66.7%; @5 86.7%; mean query 0.47 ms on 75 passages | `results/benchmark_retrieval.json`; `python scripts/index_benchmark.py` |
| Cached MiniLM hybrid retrieval | Support recall@1 97.3%; @5 98.7%; mean query 11.28 ms on the same 75 passages | `results/benchmark_retrieval_minilm_validation.json`; `python scripts/index_benchmark.py --minilm --offline --splits validation --artifact-root artifacts/minilm_benchmark --output results/benchmark_retrieval_minilm_validation.json` |
| Conservative extractive answerer, hashing | Exact alias match 22.7%; abstention 48.0% | `results/clean_answers_validation.json` and `.cases.jsonl` |
| Conservative extractive answerer, MiniLM | Exact alias match 28.0%; abstention 41.3% | `results/clean_answers_validation_minilm.json` |
| Train-only MiniLM sentence ranker | Exact alias match 54.7%; labeled support selected in 98.7%; abstention 1.3% | `results/sentence_ranker_train_minilm_strict.json`, `results/clean_answers_validation_minilm_ranker_strict.json`; `python -m generation.train_sentence_ranker --minilm --offline --strict` |
| Support-removal stress test | Ranker answers 7/75 (9.3%) after labeled support passage removed; conservative extractor answers 1/75 (1.3%) | `results/clean_answers_validation_minilm_ranker_strict_no_support.json`, `results/clean_answers_validation_minilm_no_support.json` |
| NQ-Open challenge against validation corpus | 44/50 questions have no exact answer alias in the 75-passage corpus; strict ranker abstains on 20/44 (45.5%) of those and 21/50 (42%) overall. Alias absence is not proof of no semantic support. | `results/nq_challenge_validation_minilm_ranker.json` and `.cases.jsonl`; `python -m evaluation.evaluate_nq_challenge` |
| Overt greedy poisoning, hashing answerer | 65/75 undefended attack success; 0/75 defended; 15/75 clean alias recovery | `results/defense_validation_greedy_accepted_ingest.json` and `.cases.jsonl` |
| Stealth answer substitution, hashing answerer | 12/75 undefended; 11/75 defended; 4/75 attack documents quarantined | `results/defense_validation_stealth_accepted_ingest.json` and `.cases.jsonl` |
| Greedy transfer to MiniLM + ranker | 31/75 undefended; 0/75 defended; 39/75 clean alias recovery; median case latency ~1.03 s | `results/defense_validation_greedy_accepted_ingest_minilm_ranker.json` and `.cases.jsonl` |
| Stealth transfer to MiniLM + ranker | 21/75 undefended and defended; 4/75 attack documents quarantined; median case latency ~0.96 s | `results/defense_validation_stealth_accepted_ingest_minilm_ranker.json` and `.cases.jsonl` |
| Same-template detector | Greedy recall 100%, clean FPR 4% at validation threshold 0.6902 | `results/research_detector_validation.json` |
| Different stealth family detector | Recall 5.3%, ROC-AUC 0.513 at the same frozen threshold | `results/research_detector_validation_stealth.json` |
| Experimental train-only mixed-family detector | Greedy recall 98.7%; stealth recall 4.0%; clean FPR 4.0%; did not replace the frozen detector | `results/research_detector_mixed_validation_greedy.json`, `results/research_detector_mixed_validation_stealth.json` |
| Real local Qwen 0.5B SRQ | Threshold calibrated on 50 train-clean cases; on 75 validation clean cases FPR 5/75 (6.7%), greedy recall 0/75, stealth recall 4/75 (5.3%); ROC-AUC 0.319 greedy, 0.435 stealth. Kept out of quarantine. | `results/slm_srq_validation_50_75.json` and `.cases.jsonl`; `python -m evaluation.evaluate_slm_srq --calibration-limit 50 --validation-limit 75 --max-new-tokens 48` |
| Local Qwen 0.5B cited-answer diagnostic | On 10 clean validation questions, the original prompt's raw text matched a known alias in 6/10 but supplied 0/10 valid citations; strict validation abstained in all ten. Prompt variants improved citation formatting only by producing empty marker-only outputs, now rejected by a regression test. This tiny prompt-development sample is not a final generator benchmark. | `results/local_generation_validation_10_raw.json` and `.cases.jsonl`; `results/local_generation_validation_10_guided_v2.json`; `results/local_generation_validation_10_two_line.json` |
| Real MiniLM encoder probes | Layer-3 Mahalanobis recall 0% greedy / 6.7% stealth, validation clean FPR 9.3%; last-layer attention deviation recall 0% / 4%, clean FPR 0%; both calibrated on separate training clean pairs | `results/model_probes_validation.json` and `.cases.jsonl` |
| Real Qwen decoder prefill probes | Middle-layer final-token Mahalanobis recall 1/75 greedy and 0/75 stealth, with 0/75 clean FPR; final-token attention-deviation recall 3/75 greedy and 7/75 stealth, with 6/75 (8%) clean FPR. Fifty train-clean references and 25 separate train-clean calibration pairs; no prompt was truncated. These are prefill internals, not generated-response activations or a serving gate. | `results/decoder_probes_validation.json` and `.cases.jsonl`; `python -m evaluation.evaluate_decoder_probes --reference-count 50 --calibration-count 25 --limit 75 --max-length 512` |
| Undefended PPO edit policy | After 200 training episodes with seed 42, 67/75 validation attack success; one-insertion and two-insertion fixed baselines also 67/75, random 27/75 | `results/ppo_train.jsonl`, `results/ppo_validation_evaluation.json`, `results/edit_rollout_validation_*.summary.json` |
| Defender-aware PPO edit policy | Frozen detector reward, accepted-ingest replacement: 11/75 defended attack success, tying fixed answer substitution; 72/75 trajectories used `PARAPHRASE` | `results/ppo_train_defender.jsonl`, `results/ppo_validation_defender_evaluation.json` |
| Offline Review-1 demo | An isolated rebuild produced 25 clean docs, 5 poisoned docs, 6 PDFs, and passed 39 unit tests in the copied project | `python scripts/smoke_rebuild.py` |

Exact answer-alias match is a narrow automatic metric. A sentence can be
factually correct while using different words, and some source answers are
long prose rather than short entities. `source_is_support` and support recall
help diagnose the selection path, but neither proves factual correctness.
The support-removal condition may leave another answer-bearing passage, so
its answer rate is not automatically a false-answer rate.

Per-case latency in `run_defense` includes building an isolated attack index;
it is not a steady-state serving latency. The MiniLM comparison uses a cached
encoder on the local CPU, not an 8B quantized generator. PPO is a single-seed
prototype and does not outperform either matched fixed-edit baseline.
`results/validation_intervals.json` reports Wilson 95% intervals and paired
query-bootstrap reductions for the four saved defense profiles; these are
descriptive because the validation set informed development choices.

The real-SLM SRQ result uses a pinned Qwen 0.5B model and MiniLM word vectors.
Its slide-defined denominator is the unique document vocabulary size. Greedy
insertion raised mean vocabulary size from 46.6 to 64.2 while mean raw SRQ fell
from 0.481 on clean passages to 0.338 on greedy passages. A post-hoc check of
high/low directions and query/response components is in
`results/slm_srq_direction_diagnostic.json`; none gave useful recall at the
train-calibrated 5% FPR target. Those six comparisons are exploratory and were
not promoted to a new defended test claim.

## Slide commitments still requiring evidence

| Slides | State at this checkpoint |
|---|---|
| 6 | Integrity, reviewed-digest trusted ingest, and extractive leave-one-out implemented; N+1 LLM inference and consensus-based quarantine are not validated. |
| 7 | SRQ formula, pinned local Qwen 0.5B response generation, calibration, and full 75-case validation measurement completed. Its observed recall is poor and clean FPR exceeds the nominal training target, so it remains diagnostic only. |
| 19, 21, 23-27 | Bounded edit MDP, edit cache, and masked factored PPO run; 774-value state replaces the unexplained 1,556-value slide target; no multi-seed advantage or ablation study. |
| 20, 22, 28-30 | Demo shield and read-only validation plus sealed-test aggregate console exist. Real MiniLM encoder and Qwen decoder prefill probes, plus model-backed SRQ, were measured but failed as standalone detectors. The local Qwen answerer was model-run, but strict citations caused abstention in its 10-case diagnostic. Generated-response probes and full research serving path remain open. |
| 31-34 | MiniLM comparison measured; Contriever, FAISS scale study, and Llama-3-8B NF4 run not completed. |
| 35-38 | Primary-source bibliography audit and frozen v1 test matrix/intervals are complete; wider reproducibility package and corrected presentation export remain open. |

## Current deployment choice

The live five-case console continues to use the conservative extractor and
the original demo detector. The learned ranker is experimental because it
answers more often when labeled support is removed and does not stop stealth
accepted-ingest poisoning. Hash verification remains a hard gate for unknown
or changed documents relative to a protected manifest.

## Two-passage development study (2026-09-26)

`results/stealth_failure_analysis_validation.json` shows that all 21 successful
stealth validation cases in the v1 MiniLM/ranker profile answered from the
altered document, and all 21 altered documents passed integrity after accepted
ingest. Exact whole-answer text from none of the 74 nonabstaining clean
answers appeared in another document in the entire 75-passage validation
corpus. An exact second-passage gate was therefore unsuitable for v1.

A separate, disjoint v2 benchmark has 140/30/30 train/validation/test query
groups with two labeled answer-bearing passages each. On v2 validation,
MiniLM retrieved both supports for 29/30 questions, while the frozen sentence
ranker matched an alias in 16/30 clean cases. Replacing the answer span in one
passage produced 2/30 attacker successes before an extra consistency gate.
A doc-only calibrated sentence-cosine gate reduced that to 1/30 but also
reduced clean alias matches to 13/30. Full-context calibration could not meet
the intended 5% clean-rejection target because 18/140 train cases lacked a
usable paired score. Its best attainable threshold left 2/30 attacks
successful and still reduced clean alias matches to 13/30. The gate is not a
serving defense. A seven-feature pairwise classifier, trained on 100 clean and
100 attacked pairs and calibrated on 40 other clean pairs, flagged 3/30
altered validation pairs and 1/30 clean pairs, but neither of the two cases
where the attacker changed the selected answer. It also remains diagnostic.
The v2 test split remains unevaluated. See
`V2_BENCHMARK_PROTOCOL.md` for construction, raw files, and limits.

## Exact-span citation study (2026-09-26)

The earlier local Qwen diagnostic accepted a document ID as a citation without
checking that the answer occurred in that document. The generator contract now
asks Qwen to copy an exact source phrase or sentence. A cited answer is accepted
only when its full token sequence occurs contiguously in every cited retrieved
document. If Qwen omits the ID, a deterministic resolver may attach one only
when the full uncited answer is an exact source span. Invalid citations and
unsupported paraphrases abstain. This verifies span provenance; it does not
establish factual truth or protect against an already accepted false source.

With pinned Qwen2.5-0.5B-Instruct weights and MiniLM retrieval, a 20-question
clean validation diagnostic retrieved labeled support in 20/20 cases. Five
outputs became span-grounded cited answers, all via exact-span citation repair;
the model itself produced zero valid `[DOC id]` citations. Only one of the 20
accepted answers contained a benchmark answer alias, and 15/20 abstained.
Some short grounded outputs repeated a query term instead of answering it.
The local generator therefore remains outside live serving; improving answer
relevance and citation formatting is still open. The summary and raw outputs
are in `results/local_generation_grounded_validation_20.json` and its paired
case file. These validation examples are development data, not a test claim.
