# Next session: stealth poisoning and evidence quality

Saved 2026-09-25. The working project and its measured limitations are
documented in `PROJECT_MEMORY.md` and `REPORT_EVIDENCE.md`. The v1 test set has
already been evaluated and must not be reused as an untouched test for new
defenses.

2026-09-27 checkpoint: steps 2 and part of step 6 are complete. The live
research lab has a unified inference and leave-one-out trace. Three
defender-aware PPO seeds (42/43/44) were evaluated against matched fixed
substitution and random edits: 31/225 versus 34/225 and 6/225 defended
successes. PPO beat random edits but not fixed substitution. An optional signed
origin map and exact-span two-origin abstention policy now exist, but the
current benchmark cannot validate that policy as a
stealth defense because source independence is unverified. The v2 test remains
untouched. Next, obtain genuinely independent source origins and measure both
clean utility and attack success; neither v2 pair gate should be promoted. An
initial detector-risk reward ablation is measured (31/225 versus 11/225), while
other architecture ablations and broader attack families still remain.
The auxiliary proxy-value loss was also ablated across three matched seeds;
all 225 deterministic validation outcomes stayed the same. An optional
edit-effect cache training signal was integrated and separately compared on
those seeds; it also changed zero deterministic validation outcomes. Neither
component improved this setup.
An action-head conditioning ablation on the same seeds likewise changed zero
deterministic validation outcomes; legal-action masks stayed active.
Three overt answer-layout development probes have since been measured on the
same 75 validation questions. The detector quarantined all 225 altered
documents, but this does not address subtle accepted-ingest substitutions or
genuinely new source families. `python -m evaluation.verify_unseen_templates`
checks the saved case files and aggregate.

2026-09-26 checkpoint: steps 2 and the first part of step 4 are complete.
`V2_BENCHMARK_PROTOCOL.md` records the new disjoint two-passage benchmark and
an unsuccessful paired-cosine gate. The v2 test split is still unevaluated.
Do not promote the cosine gate: its full-path training calibration missed the
5% clean-rejection target, and validation showed no attack reduction with
lower clean answer utility.
The later seven-feature pairwise classifier also failed to flag either
answer-changing validation attack. Both diagnostics remain outside the live
defense. The next implementation should address answer-level agreement or
trusted origin rather than add more pair-level similarity thresholds.

## Start here

1. Run `python scripts/doctor.py`, `python -m unittest discover -s tests -v`,
   and `python -m evaluation.verify_test_v1` to confirm the saved checkpoint.
2. Inspect the 75 validation stealth case records in
   `results/defense_validation_stealth_accepted_ingest_minilm_ranker.cases.jsonl`.
   Group failures by whether the altered source was retrieved, selected for
   the answer, quarantined, or replaced by a clean source. Record examples
   without changing the sealed v1 test outputs.
3. Design an answer-level agreement or trusted provenance rule for
   accepted-ingest factual replacements. Use v2 train/validation for
   development, and measure the clean-answer cost against the no-gate
   baseline. The first sentence-cosine gate failed this standard. Two MS
   MARCO passages from one query do not prove independent origin or truth.
4. Build a new, disjoint held-out source/query set and freeze its protocol,
   model artifacts, code hashes, and intended metrics before evaluating any
   new defense. Include unseen attack templates and source families. The
   existing 75-question v1 test remains historical evidence for v1.
5. Improve the local generator's citation contract and answer quality on
   training/validation data. Keep the extractive demo path until the generator
   passes cited-answer and abstention checks.
6. Run multiple PPO seeds and matched fixed/random baselines; add ablations
   only after the defense and evaluation contracts are stable.
7. Package reproduction commands and
   failure examples, and update the final slides using
   `PRESENTATION_REFERENCES_READY.md` and measured results only.

## Current decision rule

The live gate remains the original conservative demo. Real SRQ and model
internal probes were measured but did not justify deployment. The MiniLM
sentence ranker is experimental: v1 test clean alias match was 39/75; greedy
attack success fell from 37/75 to 0/75 under defense, while stealth success
changed from 23/75 to 22/75. PPO has not beaten its matched fixed baselines.
