# Two-passage benchmark v2: development protocol

Created 2026-09-26 from the revision-pinned MS MARCO v1.1 source recorded in
`data/benchmark/manifest.json`. The generated file hashes and selection rules
are in `data/benchmark_v2/manifest.json`.

## Construction and separation

- Scanned 4,042 source rows to select 200 query groups with two distinct
  *selected* passages that each contain an exact normalized answer alias.
- Excluded all 500 source query IDs, passage texts, and exact question strings
  from the original v1 benchmark. Query groups were split with seed 59 into
  140 train, 30 validation, and 30 test questions; each has two passages.
- `python -m data.validate_multisupport_benchmark` checks file hashes, group
  and passage separation, two support labels, and alias presence. Validation
  reads test files only for structural and hash checks. No model selection or
  attack result has been computed on v2 test.
- Both passages come from the same MS MARCO query row. Their independent
  publication origin and factual truth are **not established**. This benchmark
  measures resilience when one of two answer-bearing passages is altered; it
  does not prove a general provenance solution.

## Development experiments completed

On the 30-question v2 validation split, cached MiniLM retrieved at least one
support in 30/30 cases and both supports in 29/30. The frozen v1 sentence
ranker matched an answer alias in 16/30 clean cases. See
`results/multisupport_v2_validation.json` and its case records.

The one-passage replacement experiment changes the first support document's
answer span to a wrong answer drawn from v2 training queries. The second
support document remains clean. The frozen sentence ranker was attacked in
2/30 validation cases before an extra paired-passage gate. The initial gate,
calibrated on doc-only training sentences, lowered success to 1/30 but also
lowered clean alias matches from 16/30 to 14/30. Recalibration on full-context
training decisions found that 17/140 clean training cases lacked a usable
paired score, so a 5% clean rejection target was unattainable. The best
attainable threshold kept attack success at 2/30 and lowered clean alias
matches to 14/30. The gate has **not** been added to the live defense. See
`results/paired_gate_v2_validation_doc_only.json` (corrected doc-only run),
`results/paired_gate_v2_train_calibration.json`, and
`results/paired_gate_v2_validation_full_context.json`.

The superseded `results/paired_gate_v2_validation.json` is an invalid early
run and must not be used for claims. The original v2 pair evaluator expected
`source_query_id` in compact retrieved rows and could choose an unrelated
passage as the peer. The evaluator now resolves that field from the corpus
metadata by retrieved document ID and requires a non-null matching
group, and all train/validation pair results above were recomputed. The
corrected validation traces select a labeled support peer in 29/30 clean and
28/30 attacked cases. A query group still does not establish independent
publication origin.

These are small development samples, and v1 versus v2 attack rates are not a
controlled comparison: their corpora, retrieval competition, and attack
conditions differ. Sentence cosine does not prove factual agreement.

A second diagnostic trained a seven-feature lexical/MiniLM pair classifier on
100 clean and 100 substituted v2 training pairs, then calibrated risk on 40
separate clean training pairs. On v2 validation it flagged 3/30 altered pairs
and 1/30 clean pairs (ROC-AUC 0.793). It flagged neither of the two pairs that
actually changed the selected answer. This evaluator was given the labeled
passage pair and could not identify which passage was false; its reported
recall is an optimistic diagnostic, not an end-to-end defense. See
`results/pairwise_detector_v2_validation.json` and raw case records. The
classifier was **not** added to the live gate.

A post-hoc peer-answer failover probe used the corrected validation traces to
choose the other passage's answer when the pair cosine fell below a candidate
threshold. At 0.60, clean alias matches rose from 16/30 to 20/30 and the two
original attack successes disappeared, but two new successes appeared when
the poisoned passage was the peer. At 0.50, the probe had 17/30 clean aliases
and 1/30 attack success, but that threshold was inspected on validation. The
saved table and affected query IDs are in
`results/v2_peer_failover_validation_probe.json`. This shows that failover
cannot assume the peer is clean; it is not a serving defense. The v2 test
split remains untouched.

## Next decision

Before v2 test evaluation, define a better answer-level agreement or trusted
source policy on train/validation and freeze its threshold and success/utility
metrics. The two tested pairwise signals did not beat the no-gate baseline.
Until a better rule does, retain that baseline and leave v2 test untouched. A
broader final test requires genuinely distinct source provenance, more attack
families, and a larger query sample.
