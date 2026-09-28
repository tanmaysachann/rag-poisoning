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
| Pinned Contriever-msmarco retrieval | The official 438 MB weight at revision `abe8c1493371369031bcb1e02acb754cf4e162fa` was SHA-256 verified. Masked mean pooling and normalized 768-dimensional vectors gave dense support recall@1 and @5 of 75/75; BM25+RRF gave 73/75 and 74/75, matching the saved MiniLM hybrid recalls. Median CPU query encoding plus search was 25.69 ms, with a separate 5.76 s corpus encode. The MiniLM saved mean query time was 11.28 ms, measured in a different run. No answer or attack evaluation was run for Contriever. | `results/benchmark_retrieval_contriever_validation.json` and `.cases.jsonl`; `python scripts/download_contriever_weight.py`; `python -m evaluation.evaluate_contriever`; [official Meta model card](https://huggingface.co/facebook/contriever-msmarco) |
| FAISS FlatIP dense-search scale study | On 425 v1 train+validation passage vectors, median search was 0.030 ms with current NumPy full sort and 0.015 ms with FAISS; on 10,200 perturbed replicas, 1.287 ms versus 0.757 ms. Top-five scores agreed for all 30 queries; one 425-vector query had a tied ID order. CPU single-thread timing excludes encoding and BM25, and the larger corpus is synthetic. The current 75-passage serving profile stays on NumPy. | `results/faiss_exact_scale_validation.json`; `python -m evaluation.benchmark_faiss_exact` (optional `faiss-cpu==1.8.0.post1`) |
| Conservative extractive answerer, hashing | Exact alias match 22.7%; abstention 48.0% | `results/clean_answers_validation.json` and `.cases.jsonl` |
| Conservative extractive answerer, MiniLM | Exact alias match 28.0%; abstention 41.3% | `results/clean_answers_validation_minilm.json` |
| Train-only MiniLM sentence ranker | Exact alias match 54.7%; labeled support selected in 98.7%; abstention 1.3% | `results/sentence_ranker_train_minilm_strict.json`, `results/clean_answers_validation_minilm_ranker_strict.json`; `python -m generation.train_sentence_ranker --minilm --offline --strict` |
| Support-removal stress test | Ranker answers 7/75 (9.3%) after labeled support passage removed; conservative extractor answers 1/75 (1.3%) | `results/clean_answers_validation_minilm_ranker_strict_no_support.json`, `results/clean_answers_validation_minilm_no_support.json` |
| NQ-Open challenge against validation corpus | 44/50 questions have no exact answer alias in the 75-passage corpus; strict ranker abstains on 20/44 (45.5%) of those and 21/50 (42%) overall. Alias absence is not proof of no semantic support. | `results/nq_challenge_validation_minilm_ranker.json` and `.cases.jsonl`; `python -m evaluation.evaluate_nq_challenge` |
| Overt greedy poisoning, hashing answerer | 65/75 undefended attack success; 0/75 defended; 15/75 clean alias recovery | `results/defense_validation_greedy_accepted_ingest.json` and `.cases.jsonl` |
| Three answer-layout development probes | `Question/Answer` header, answer-key entry, and postscript layouts used the same 75 validation questions and saved greedy wrong-answer schedule. All 225 altered documents were retrieved and quarantined, with 0 defended attack successes; undefended successes were 65/75, 65/75, and 66/75. Each layout recovered 15/75 clean aliases after filtering. These overt layouts are development probes, not evidence of generalization to independent attack sources. | `results/unseen_templates_validation.json` and `.cases.jsonl`; `python -m evaluation.evaluate_unseen_templates`; `python -m evaluation.verify_unseen_templates` |
| Stealth answer substitution, hashing answerer | 12/75 undefended; 11/75 defended; 4/75 attack documents quarantined | `results/defense_validation_stealth_accepted_ingest.json` and `.cases.jsonl` |
| Accepted stealth copy budget, hashing answerer | Replacing one labeled support passage with the saved stealth edit gave 11/75 defended wrong answers. Adding one or two identical accepted copies gave 9/75 and 9/75; poisoned passages retrieved across the 75 cases rose from 60 to 119 and 174. The extra copies competed for rank and did not increase attack success in this narrow setup. All copies share one attacker origin, so this does not test independent corroboration or varied poisoning. | `results/poison_budget_stealth_validation.json` and `.cases.jsonl`; `python -m evaluation.evaluate_poison_budget`; `python -m evaluation.verify_poison_budget` |
| V2 peer-answer failover probe | On the corrected 30-question two-passage validation traces, replacing the selected answer with its paired answer when cosine fell below 0.60 raised clean alias matches from 16 to 20 and removed both original wrong-answer successes, but created two new wrong-answer successes by choosing poisoned peers. A post-hoc 0.50 threshold produced 17 clean aliases and 1 attacker success, but it was selected after inspecting validation. Neither result supports a deployed threshold or independent-origin claim. | `results/v2_peer_failover_validation_probe.json`; `python -m evaluation.probe_v2_peer_failover` |
| Greedy transfer to MiniLM + ranker | 31/75 undefended; 0/75 defended; 39/75 clean alias recovery; median case latency ~1.03 s | `results/defense_validation_greedy_accepted_ingest_minilm_ranker.json` and `.cases.jsonl` |
| Stealth transfer to MiniLM + ranker | 21/75 undefended and defended; 4/75 attack documents quarantined; median case latency ~0.96 s | `results/defense_validation_stealth_accepted_ingest_minilm_ranker.json` and `.cases.jsonl` |
| Same-template detector | Greedy recall 100%, clean FPR 4% at validation threshold 0.6902 | `results/research_detector_validation.json` |
| Different stealth family detector | Recall 5.3%, ROC-AUC 0.513 at the same frozen threshold | `results/research_detector_validation_stealth.json` |
| Experimental train-only mixed-family detector | Greedy recall 98.7%; stealth recall 4.0%; clean FPR 4.0%; did not replace the frozen detector | `results/research_detector_mixed_validation_greedy.json`, `results/research_detector_mixed_validation_stealth.json` |
| Real local Qwen 0.5B SRQ | Threshold calibrated on 50 train-clean cases; on 75 validation clean cases FPR 5/75 (6.7%), greedy recall 0/75, stealth recall 4/75 (5.3%); ROC-AUC 0.319 greedy, 0.435 stealth. Kept out of quarantine. | `results/slm_srq_validation_50_75.json` and `.cases.jsonl`; `python -m evaluation.evaluate_slm_srq --calibration-limit 50 --validation-limit 75 --max-new-tokens 48` |
| Local Qwen 0.5B cited-answer diagnostic | On 10 clean validation questions, the original prompt's raw text matched a known alias in 6/10 but supplied 0/10 valid citations; strict validation abstained in all ten. Prompt variants improved citation formatting only by producing empty marker-only outputs, now rejected by a regression test. This tiny prompt-development sample is not a final generator benchmark. | `results/local_generation_validation_10_raw.json` and `.cases.jsonl`; `results/local_generation_validation_10_guided_v2.json`; `results/local_generation_validation_10_two_line.json` |
| Experimental Qwen span repair | Replaying 20 saved clean validation outputs, strict exact repair cited the labeled support in 5/20 and matched a known alias in 1/20. A question-aware longest contiguous source phrase cited support in 7/20 and matched an alias in 3/20, with 13/20 abstentions. On the same questions the cached MiniLM sentence ranker matched 9/20. The repair remains offline because lexical alignment does not establish answer relevance or attack robustness. | `results/local_generation_answer_repair_validation_20.json` and `.cases.jsonl`; `python -m evaluation.evaluate_answer_repair` |
| Real MiniLM encoder probes | Layer-3 Mahalanobis recall 0% greedy / 6.7% stealth, validation clean FPR 9.3%; last-layer attention deviation recall 0% / 4%, clean FPR 0%; both calibrated on separate training clean pairs | `results/model_probes_validation.json` and `.cases.jsonl` |
| Real Qwen decoder prefill probes | Middle-layer final-token Mahalanobis recall 1/75 greedy and 0/75 stealth, with 0/75 clean FPR; final-token attention-deviation recall 3/75 greedy and 7/75 stealth, with 6/75 (8%) clean FPR. Fifty train-clean references and 25 separate train-clean calibration pairs; no prompt was truncated. These are prefill internals, not generated-response activations or a serving gate. | `results/decoder_probes_validation.json` and `.cases.jsonl`; `python -m evaluation.evaluate_decoder_probes --reference-count 50 --calibration-count 25 --limit 75 --max-length 512` |
| Undefended PPO edit policy | After 200 training episodes with seed 42, 67/75 validation attack success; one-insertion and two-insertion fixed baselines also 67/75, random 27/75 | `results/ppo_train.jsonl`, `results/ppo_validation_evaluation.json`, `results/edit_rollout_validation_*.summary.json` |
| Defender-aware PPO edit policy | Frozen detector reward, accepted-ingest replacement: 11/75 defended attack success, tying fixed answer substitution; 72/75 trajectories used `PARAPHRASE` | `results/ppo_train_defender.jsonl`, `results/ppo_validation_defender_evaluation.json` |
| Defender-aware PPO across three seeds | Seeds 42/43/44 each trained for 200 episodes and evaluated on the same 75 validation questions with seed-matched fixed substitution and random edits. PPO defended successes: 11, 8, 12; fixed substitution: 11, 11, 12; random edits: 2, 1, 3. PPO minus fixed: -3/225 (-1.3 percentage points), query-cluster bootstrap 95% interval -3.1 to 0.0 points. PPO minus random: +25/225 (+11.1 points), interval 5.3 to 17.3 points. PPO beat random edits but not fixed substitution. | `results/ppo_multiseed_validation.json`, per-seed policy and baseline case files; `python -m evaluation.summarize_ppo_seeds` |
| PPO detector-risk reward ablation | Three matched 200-episode defender-aware training seeds and 75 validation questions per seed. The original detector-risk step penalty yielded 31/225 defended successes; setting only that weight to zero yielded 11/225. Paired difference +8.9 percentage points (query-cluster bootstrap 95% interval +4.4 to +13.8). The terminal defended-success reward stayed active. Seed 42 tied, while seeds 43/44 selected insertion edits that the detector caught. This is a development ablation of one component, not proof of PPO superiority over fixed substitution. | `results/ppo_detection_reward_ablation_validation.json`, six checkpoints and case files; `python -m evaluation.summarize_ppo_detection_ablation` |
| PPO auxiliary critic ablation | Three matched 200-episode defender-aware seeds were retrained with the proxy-value loss weight changed from 0.1 to 0 while all reward terms were retained. Both versions achieved 31/225 defended successes on the same 75 validation questions per seed, with zero changed deterministic case outcomes. This head did not improve this measured setup. | `results/ppo_proxy_value_ablation_validation.json`, three new checkpoints and case files; `python -m evaluation.summarize_ppo_proxy_ablation` |
| PPO edit-effect cache ablation | An optional exact-neighbor cache used prior state/action edit effects to shape step rewards during three matched 200-episode defender-aware runs; it was absent in the original PPO runs. With cache weight 0.5, the cache held 333, 339, and 320 edit rows by seed. Both conditions achieved 31/225 defended validation successes, with zero changed deterministic case outcomes. This cache predicts proxy step reward, not terminal attack success, and provides no measured advantage here. | `results/ppo_cache_ablation_validation.json`, three new checkpoints, training histories, and validation case files; `python -m evaluation.summarize_ppo_cache_ablation` |
| PPO action-head conditioning ablation | Three matched 200-episode seeds were retrained after removing the chosen-operation input from the position head and the chosen-operation/position inputs from the payload head. Legal-action masks remained active. Both versions had 31/225 defended successes and identical deterministic validation outcomes. This limited ablation does not show a benefit from the conditioning inputs in this setup. | `results/ppo_head_conditioning_ablation_validation.json`, three new checkpoints and case files; `python -m evaluation.summarize_ppo_head_ablation` |
| Offline Review-1 demo | An isolated rebuild produced 25 clean docs, 5 poisoned docs, and 6 PDFs; the isolated copy runs demo and grounding-contract tests. The full research suite runs separately in the main checkout. | `python scripts/smoke_rebuild.py`; `python -m unittest discover -s tests -q` |

Exact answer-alias match is a narrow automatic metric. A sentence can be
factually correct while using different words, and some source answers are
long prose rather than short entities. `source_is_support` and support recall
help diagnose the selection path, but neither proves factual correctness.
The support-removal condition may leave another answer-bearing passage, so
its answer rate is not automatically a false-answer rate.

Per-case latency in `run_defense` includes building an isolated attack index;
it is not a steady-state serving latency. The MiniLM comparison uses a cached
encoder on the local CPU, not an 8B quantized generator. Defender-aware PPO
has now been compared across three seeds and does not outperform fixed answer
substitution; the undefended policy still has a single-seed comparison.
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
| 19, 21, 23-27 | Bounded edit MDP, optional edit-effect cache training signal, and masked factored PPO run; 774-value state replaces the unexplained 1,556-value slide target. Three defender-aware seeds show no advantage over fixed substitution. Detector-risk reward, auxiliary critic, edit-cache, and action-head conditioning ablations are measured; other reward and architecture ablations remain open. |
| 20, 22, 28-30 | Demo shield, live validation inference trace, and sealed-test aggregate console exist. The live lab now logs retrieval, integrity, detector, exact-span answer, and leave-one-out results. Real MiniLM encoder and Qwen decoder prefill probes, plus model-backed SRQ, were measured but failed as standalone detectors. Local Qwen cited-answer quality remains inadequate for serving; generated-response probes and broad research serving remain open. |
| 31-34 | MiniLM, pinned Contriever-msmarco, and FAISS FlatIP comparisons measured; Llama-3-8B NF4 has not run on the CPU-only host. |
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
reduced clean alias matches to 14/30. Full-context calibration could not meet
the intended 5% clean-rejection target because 17/140 train cases lacked a
usable paired score. Its best attainable threshold left 2/30 attacks
successful and still reduced clean alias matches to 14/30. The gate is not a
serving defense. A seven-feature pairwise classifier, trained on 100 clean and
100 attacked pairs and calibrated on 40 other clean pairs, flagged 3/30
altered validation pairs and 1/30 clean pairs, but neither of the two cases
where the attacker changed the selected answer. It also remains diagnostic.
These pair results were recomputed after fixing the evaluator's metadata lookup: the old
evaluator could mistake an unrelated passage for the paired support. The
corrected peer matched a labeled support in 29/30 clean and 28/30 attacked
validation runs.
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

## Signed source-origin contract (2026-09-27)

`security/provenance.py` binds operator-reviewed origin labels to document
hashes in a signed manifest. Its optional strict inference mode accepts an
exact cited span only when at least two accepted documents from distinct
attested origins contain that span. Tests cover repeated labels, altered
documents, forged manifest labels, and abstention when the attestation is
missing. This is a functional contract, not an efficacy result: the current
MS MARCO passages have no verified independent publication origins, so the
policy is disabled in the live lab. A future corpus must measure clean-answer
loss and poisoning success before this can be promoted as a defense.
