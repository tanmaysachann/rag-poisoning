# Next session: stealth poisoning and evidence quality

Saved 2026-09-25. The working project and its measured limitations are
documented in `PROJECT_MEMORY.md` and `REPORT_EVIDENCE.md`. The v1 test set has
already been evaluated and must not be reused as an untouched test for new
defenses.

## Start here

1. Run `python scripts/doctor.py`, `python -m unittest discover -s tests -v`,
   and `python -m evaluation.verify_test_v1` to confirm the saved checkpoint.
2. Inspect the 75 validation stealth case records in
   `results/defense_validation_stealth_accepted_ingest_minilm_ranker.cases.jsonl`.
   Group failures by whether the altered source was retrieved, selected for
   the answer, quarantined, or replaced by a clean source. Record examples
   without changing the sealed v1 test outputs.
3. Design and implement a provenance/corroboration defense for accepted-ingest
   factual replacements. Train and calibrate only on training data; use the
   existing validation set for development. Treat unsupported single-source
   answers as a reason to abstain, and measure the clean-answer cost.
4. Build a new, disjoint held-out source/query set and freeze its protocol,
   model artifacts, code hashes, and intended metrics before evaluating any
   new defense. Include unseen attack templates and source families. The
   existing 75-question v1 test remains historical evidence for v1.
5. Improve the local generator's citation contract and answer quality on
   training/validation data. Keep the extractive demo path until the generator
   passes cited-answer and abstention checks.
6. Run multiple PPO seeds and matched fixed/random baselines; add ablations
   only after the defense and evaluation contracts are stable.
7. Complete retrieval/storage comparisons, package reproduction commands and
   failure examples, and update the final slides using
   `PRESENTATION_REFERENCES_READY.md` and measured results only.

## Current decision rule

The live gate remains the original conservative demo. Real SRQ and model
internal probes were measured but did not justify deployment. The MiniLM
sentence ranker is experimental: v1 test clean alias match was 39/75; greedy
attack success fell from 37/75 to 0/75 under defense, while stealth success
changed from 23/75 to 22/75. PPO has not beaten its matched fixed baselines.
