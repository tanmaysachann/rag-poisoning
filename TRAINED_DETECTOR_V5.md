# Trained detector experiment

## What runs

`detect/train_cross_encoder_pilot.py` fine-tunes MiniLM to score a question and
candidate passage. It reads only the 1,200-question MS MARCO training split and
150-question validation split in `data/detector_pilot_v4`. The model uses no
hand-written attack signal at inference. It is combined with the frozen v3
eight-feature detector using a fixed OR gate: either detector can quarantine a
document. `detect/combined_gate.py` is the local inference implementation;
`scripts/run_combined_query.py` runs that policy on a supplied corpus and query.

The confirmation data in `data/detector_confirmation_v5` has 150 new MS MARCO
questions and 100 new SQuAD contexts. The model, thresholds, and OR policy were
frozen before this confirmation. Each question was attacked with four locally
authored layouts; retrieval and the extractive answerer were held identical
between gates. The aggregate paired interval resamples questions, keeping all
four attacks on each question together.

## Confirmed outcomes

| Split | Wrong-answer successes, old | Wrong-answer successes, combined | Clean alias matches, old | Clean alias matches, combined |
| --- | ---: | ---: | ---: | ---: |
| MS MARCO | 99/600 | 41/600 | 22/150 | 21/150 |
| SQuAD | 120/400 | 47/400 | 21/100 | 21/100 |

The MS MARCO paired reduction is 9.7 percentage points, with a question-clustered
95% bootstrap interval of 6.7 to 12.8 points. The combined policy passed the
predeclared end-to-end adoption rule. Median detector scoring time was about
16 ms for v3 and 34 ms for the new model alone on a laptop RTX 3050; combined
scoring requires both. Model load time is excluded.

## Important limits

This is a **local research policy**, not a solved RAG system. The attack layouts
are synthetic, not adaptive or independently authored. On new MS MARCO clean
questions the underlying answerer matched only 24/150 aliases before any gate.
The combined gate flagged 7.6% of clean retrieved documents on MS MARCO and
15.8% on SQuAD, versus 2.4% and 0.4% for v3. Unchanged SQuAD answer accuracy
does not make that false-flag increase safe for a better answerer. The model also
cannot verify a plausible false claim without an independent trusted source.

The public web deployment still uses the smaller v3 detector. The trained
weights are a local 90 MB artifact under `artifacts/detector_pilot_v4` and are
excluded from Git; the public runtime does not install PyTorch/Transformers.
`research_protocols/combined_gate_v5.json` records hashes and causes the local
runner to fail if the model or evidence is missing or changed. There is no
silent fallback to v3 while claiming to run the combined policy.

## Reproduce locally

Install `requirements-dev.txt`, then run from the repository root:

```powershell
python -m data.build_detector_pilot
python -m detect.train_cross_encoder_pilot
python -m evaluation.evaluate_detector_pilot
python -m data.build_detector_pilot --output data/detector_confirmation_v5 --train 300 --validation 50 --test 150 --domain-test 100 --seed 1518 --exclude data/detector_pilot_v4
python -m evaluation.evaluate_detector_pilot --data data/detector_confirmation_v5 --output results/detector_confirmation_v5_test.json
python -m evaluation.finalize_detector_confirmation results/detector_confirmation_v5_test.json
python -m scripts.freeze_combined_gate
python -m evaluation.verify_detector_evidence
python -m scripts.run_combined_query --corpus data/benchmark_v3/validation/corpus.jsonl --query "average daily urine output for pediatric association"
```

The builder refuses to overwrite existing frozen data, and the freezer refuses
to overwrite an existing gate manifest. Remove or select new output paths only
when intentionally starting a new experiment.
