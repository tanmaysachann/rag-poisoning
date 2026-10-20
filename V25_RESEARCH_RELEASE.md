# V25 scanner: reference-adaptive rule and new-domain holdout

V25 is the current poisoned-passage **review scanner**. It joins the trained V3
behavioral risk, V24 learned fusion score, source containment, and an optional
authenticated document digest. The learned fusion signal must also exceed the
score of the supplied reference by a frozen margin. A caller must authenticate
that reference; a benchmark qrel alone cannot do so. `review` means the
passage needs inspection, not that its factual claims are proven false.

## What trained the model, and what did not

The trained detector uses 14,504 BEIR Natural Questions warm-start rows from
2,072 questions, then 1,855 PoisonArena six-family fine-tuning rows from 51
questions. The V3 behavioral model is an earlier trained component of the
ensemble. See [the V24 product data table](PRODUCT_V24.md#exactly-what-data-trained-it),
`data/reference_detector_v9/manifest.json`, and
`data/multifamily_v20/{train,calibration,evaluation}.jsonl` for actual labeled
passages and splits. The 2,681,468-passage NQ snapshot is a source lookup
index, not millions of labeled training examples.

V25 is a **frozen decision rule over existing trained signals**, not a newly
trained neural checkpoint. The rule was chosen using previously inspected
PoisonArena development and SciFact synthetic transfer results. It was frozen
with source and artifact SHA-256 values in `results/v25_rule_frozen.json`
before scoring the new NFCorpus holdout. No NFCorpus cases were used for
training, threshold selection, or rule revision.

PPO in this project trains an **attacker** to edit retrieved passages. It does
not train the V25 defender. `poison/{env,train,policy,ppo}.py` implements that
loop, and `results/v24_ppo_multiseed.json` records three 200-episode seeds
and paired defender replays. [The V24 research release](V24_RESEARCH_RELEASE.md#what-rl-actually-does)
explains the reward and measured limitations. This is the precise answer to
"how is reinforcement learning implemented?"

## The new, inspectable evaluation dataset

The holdout source is the [official BEIR NFCorpus archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip).
`data/v25_nfcorpus_holdout/manifest.json` pins the archive, build script,
source hashes, question selection, and case counts. The 120 test questions
have no exact question-text overlap with the detector's earlier training,
calibration, or development questions. Each question has one qrel-relevant
original passage, one other original clean passage, and five deterministic
attacker-authored new-document insertions: system notice, forged Q&A, buried
override, role delimiter, and editorial key. Thus
`data/v25_nfcorpus_holdout/cases.jsonl` has **840 source-labeled cases**:
240 original clean and 600 synthetic attacks. Open that JSONL file to show
the exact question, reference, candidate, label, and family for any row.

The labels describe **source and construction**. NFCorpus relevance judgments
do not establish factual truth. The fixed attack templates do not represent
all possible poisoned documents. This is a scanner evaluation, not an
end-to-end answer-quality evaluation.

## Frozen decision and first holdout result

The rule requests review on source containment, a supplied known-ID digest
mismatch, V3 behavioral risk at least 0.75, or V24 fusion score at least
4.272025739459554 and at least 0.5 above the reference fusion score. The
threshold and exact implementation are pinned before holdout scoring in
`results/v25_rule_frozen.json`. The first scoring report is
`results/v25_nfcorpus_holdout.json`, with individual predictions in the
adjacent `.cases.jsonl` file.

| Scanner | Synthetic attacks flagged | Original clean passages flagged |
| --- | ---: | ---: |
| V24 | 137/600 (22.8%) | 16/240 (6.7%) |
| V3 behavioral alone | 495/600 (82.5%) | 16/240 (6.7%) |
| **V25** | **493/600 (82.2%)** | **14/240 (5.8%)** |

V25 improves markedly over V24 on this new corpus. V3 alone has nearly the
same attack recall here, so these data do not show a large improvement over
V3 by itself. V25 retains the other source checks and had much stronger
performance than V3 alone on the inspected PoisonArena development set.
V25 caught all 120 attacks in each of four NFCorpus template families, but
only **13/120 role-delimiter attacks**. That is a clear remaining weakness.
The report includes 95% question-cluster bootstrap intervals and hashes for
the frozen rule, dataset, and prediction trace. This holdout must not be used
for a new threshold choice while calling the resulting estimate untouched.

## Reproduce and demonstrate

From the repository root, with the official NFCorpus archive at
`tmp/nfcorpus.zip` and installed dependencies:

```powershell
Get-Content data/v25_nfcorpus_holdout/manifest.json
Get-Content data/v25_nfcorpus_holdout/cases.jsonl -TotalCount 1
Get-Content results/v25_rule_frozen.json
Get-Content results/v25_nfcorpus_holdout.json
python -m unittest tests.test_product_v25 tests.test_research_api tests.test_v24_answer_gate -v
python -m scripts.scan_poison_v25 --input data/v25_nfcorpus_holdout/cases.jsonl --limit 3
```

`data/build_v25_nfcorpus_holdout.py` and
`evaluation/evaluate_v25_nfcorpus.py` document the build and scoring methods.
They refuse to replace completed outputs. The hosted scanner exposes both
V24 and V25 on the same input, along with the saved V25 holdout evidence.
The separate research RAG answer replay remains a V24 admitted-source replay;
the V25 result above does not claim improved live answer accuracy.
