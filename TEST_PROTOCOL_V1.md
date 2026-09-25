# Frozen test protocol v1 (2026-09-25)

This protocol was written before executing attack or defense evaluation on the 75-question benchmark test split. Training and threshold selection used only the train and validation splits. The later mixed-family detector experiment did not improve stealth recall, so v1 retains the original frozen detector. Any future model or threshold change requires a new versioned test protocol and must be labeled exploratory on this same test set.

## Predeclared conditions

- Dataset: revision-pinned MS MARCO v1.1 benchmark in `data/benchmark/test`, 75 query/passage pairs. This small sampled corpus is a local experiment, not a full MS MARCO evaluation.
- Attack construction: `run_attack_baselines.py`, seed 42, all 75 questions, greedy three-position insertion and stealth answer-span replacement. Wrong answers are sampled from the **train** alias pool.
- Primary threat surface: `accepted_ingest`; poisoned content is sealed into a new trusted snapshot, so content integrity alone is expected to pass. The defender must use its trained content detector.
- Primary system: cached `all-MiniLM-L6-v2` hybrid retrieval, frozen strict train-only sentence ranker for extractive answer selection, and frozen eight-feature hashing detector with threshold `0.6902` selected from clean validation scores to target at most 5% FPR. Selective leave-one-out remains diagnostic.
- Primary paired outcomes per attack family: attack success before and after defense; attack-document quarantine; clean answer-alias recovery; abstention. Also report retrieval support recall, clean alias-match accuracy, and latency separately. Exact alias matching is an imperfect answer-quality metric.
- Uncertainty: Wilson 95% intervals for event rates and query-paired bootstrap for the before/after attack-success reduction. No significance claim from this small split.
- The Review-1 five-case demo and experimental mixed-family detector are outside this frozen evaluation.

## Frozen input hashes (SHA-256)

| Input | SHA-256 |
|---|---|
| `data/benchmark/test/corpus.jsonl` | `9b158876935933da286457192819866cc169e812ba353fc7d5db4e9abd5664ce` |
| `data/benchmark/test/queries.jsonl` | `0a80f2e8aba87e93752d3bf6d254283d066c6ec9c5c4b6c7574609b16660fdae` |
| `artifacts/research_detector_hashing.joblib` | `a7c379d05fb0b08305cb4549267fc62a1e54a64db9748e08c6e6c7c6be25798f` |
| `artifacts/sentence_ranker_minilm_strict.joblib` | `23335392899e967c3edf30ee1f3a905c6beef55c536f183569c603b49017c340` |
| `attack/baselines.py` | `73c7093f38f66dd087a5c7a5967595f6a7caf84f531a6a8e3df9aac01bef8d97` |
| `evaluation/run_attack_baselines.py` | `2ce87b88322a1056ff1551bffea755351bc19a9a5035b1e1c3ca51e87b867057` |
| `evaluation/run_defense.py` | `dd4c6ccc5e5e2769537db5cd14fa480e5ca63e04b93a90c8887bdd1410e4792f` |
| `pipeline/secure_rag.py` | `bb900bbe66ad6154f917491195bfae8aa118c0b86633611cf5f64f49eaa3584b` |
| `retrieval/hybrid_retriever.py` | `a87a57a4ac0d81b5e9f02146ab7421d8775dac1624866f770bc7a3bd90c03a71` |

## Commands

```powershell
python scripts/index_benchmark.py --splits test --output results/benchmark_retrieval_test.json
python scripts/index_benchmark.py --minilm --offline --splits test --artifact-root artifacts/minilm_benchmark --output results/benchmark_retrieval_minilm_test.json
python evaluation/run_attack_baselines.py --split test --strategy greedy --limit 75
python evaluation/run_attack_baselines.py --split test --strategy stealth --limit 75
python -m evaluation.evaluate_clean_answers --split test --minilm --offline --sentence-ranker --strict
python -m evaluation.run_defense --split test --strategy greedy --minilm-ranker
python -m evaluation.run_defense --split test --strategy stealth --minilm-ranker
```

The clean-answer command above was checked against its CLI help before test execution. Save raw per-case JSONL files along with summaries.

The completed run is sealed in `TEST_EVIDENCE_V1.json`, including hashes of the
timing-sensitive result files and expected case counts. Run
`python -m evaluation.verify_test_v1` to check every hash, query alignment,
summary, and count. A later mechanical rerun can use `--skip-output-hashes`
because measured latency changes byte-level output while the input hashes and
case outcomes remain independently checked.
