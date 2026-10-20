# V26 role-boundary scanner and FiQA evaluation

V26 adds a supervised structural role-boundary classifier to the frozen V25
review decision. It still requires an independently authenticated reference
for source comparisons. A `review` output is a source-risk signal, not a
claim that a document's factual content is false. The separate RAG answer
gate admits only documents whose bytes match a trusted pre-attack snapshot.

## What was trained

The existing V24/V25 ensemble uses the published project training data listed
in `PRODUCT_V24.md` and `V25_RESEARCH_RELEASE.md`: 14,504 BEIR Natural
Questions warm-start rows and 1,855 PoisonArena six-family fine-tuning rows,
plus the earlier trained V3 behavioral detector. The 2.68-million-passage NQ
index is a trusted-source lookup, **not** a labeled training set.

V26's added component is logistic regression over role markers, imperative
language, answer redirection, and quotation cues. It trains on 80 of the
previously inspected NFCorpus questions with synthetic role attacks and
benign markup examples (880 rows). Forty grouped questions (440 rows) set
the 0.5 operating threshold. The development report records 200/200
synthetic role examples flagged and 0/240 clean examples flagged on that
calibration split. The fixed source, feature code, model artifact, threshold,
and hashes are in `results/v26_role_development.json` and
`results/v26_rule_frozen.json`.

PPO in this project trains an **attacker** to edit retrieved passages, as
implemented in `poison/{env,train,policy,ppo}.py` and reported in
`results/v24_ppo_multiseed.json`. PPO does **not** train V26's defender. The
V26 role component is supervised learning. This distinction should be stated
explicitly in a mentor demonstration.

## Data that can be shown

The new cross-domain source is the [official BEIR FiQA archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip),
verified against its published MD5 and pinned SHA-256. The selection script
`data/build_v26_fiqa_holdout.py` chose 120 FiQA test questions with no exact
normalized-text overlap with the earlier NQ, PoisonArena, SciFact, or
NFCorpus schedules. Its frozen `data/v26_fiqa_holdout/cases.jsonl` contains
960 inspectable rows: 120 qrel-relevant reference passages, 120 other
original passages, 120 synthetic benign markup cases, and 600 synthetic
attacker-authored insertions across five fixed families. The adjacent
`manifest.json` records archive, builder, case hashes, selection, and labels.
BEIR relevance labels do not verify factual truth.

The V26 rule and FiQA cases were hash-pinned before the first score. The
first report `results/v26_fiqa_holdout.json` and per-case trace are retained
without revising the frozen classifier after inspection.

## First cross-domain result

| Scanner | Fixed synthetic attacks flagged | Clean and benign-markup cases flagged |
| --- | ---: | ---: |
| V25 | 600/600 | 10/360 |
| V26 | 600/600 | 10/360 |

The FiQA challenge shows **no incremental V26 gain** because V25 already
flags every fixed attack in this challenge. Both flag 0/240 original clean
passages and 10/120 synthetic benign markup passages. The trained role signal
itself flags all 360 role-family attacks and none of the 360 clean cases.
These figures describe a fixed synthetic source-label challenge, not general
poisoned-document accuracy or factual answer accuracy.

On the previously inspected NFCorpus development cases, V26 flags 120/120
role-delimiter attacks versus V25's 13/120 while retaining the same 14/240
flags on original clean passages. The 80 training questions and 40 threshold
calibration questions are shown separately in
`results/v26_nfcorpus_development_replay.json`; this is **development**
evidence, not a fresh holdout.

## Targeted follow-up after the first result

After the first FiQA aggregate result showed V25 saturation, a separate
role-delimiter stress schedule was built from 120 *other* FiQA test questions
and pinned before scoring in
`results/v26_fiqa_role_stress_frozen.json`. It reuses the known NFCorpus
role-delimiter pattern, so it is an explicitly **post-hoc targeted stress
test**, not a second blind discovery test. The cases and source manifest are
in `data/v26_fiqa_role_stress/`; the scored report and per-case trace are in
`results/v26_fiqa_role_stress.json` and the adjacent JSONL.

| Targeted case | V25 flagged | V26 flagged |
| --- | ---: | ---: |
| Role-delimiter synthetic attacks | 0/120 | **120/120** |
| Original references | 0/120 | 0/120 |
| Quoted-markup clean controls | 3/120 | 3/120 |

This establishes the effect of the added role signal for this known pattern
on distinct questions. It does not establish robustness to arbitrary new
attack styles. The service reports the first FiQA result and this follow-up
with their different evidence status.

## End-to-end retrieval and answer replay

`evaluation/evaluate_v26_answer_path.py` built a controlled 960-document
index from the *same* FiQA challenge cases after scanner scoring. It ran
BM25 plus hashing-dense retrieval, the V25 and V26 scanners, authenticated
source admission, and the existing deterministic extractive answerer for
120 questions. The saved report and per-question trace are in
`results/v26_fiqa_answer_path.json` and the adjacent JSONL. This is an
engineering replay, not a second independent holdout.

An attack document reached the top five for every question (586 retrieved
attack documents in total). The undefended path sourced 113 answers from
attack documents and emitted the exact attacker target in 109 cases. With
source admission, both V25 and V26 sourced **0** answers from attack
documents. Every inserted attack lacks a known source hash, so this zero is
primarily the admission policy's effect; it does not demonstrate that the
scanner alone would stop the downstream answer.

Answer utility is poor in this replay. Even before attacks, the extractive
answerer produced a sourced answer on only **7/120** questions in the
240-document clean subset. With the gate, V26 abstained on **119/120**.
The report's 114 exact clean-answer matches include matching abstentions
and must not be presented as answer accuracy. No hosted LLM or verified
FiQA answer quality was measured. This is a concrete remaining product
limitation, despite the strong source-screening figures.

## Reproduce

With the official archives in `tmp/` and the pinned dependencies installed:

```powershell
Get-Content data/v26_fiqa_holdout/manifest.json
Get-Content data/v26_fiqa_holdout/cases.jsonl -TotalCount 1
Get-Content results/v26_rule_frozen.json
Get-Content results/v26_fiqa_holdout.json
python -m training.train_v26_role  # only in a clean checkout without the artifact/report
python -m evaluation.evaluate_v26_fiqa  # only before the report exists
```

The training and evaluation commands refuse to replace their pinned outputs.
The sample commands above show the exact questions, passages, labels,
predictions, and hashes used. The deployed `/api/v26/evidence` endpoint serves
the saved reports; `/api/v26/scan` performs live inference. The release also
retains V25 and V24 for paired scans.
