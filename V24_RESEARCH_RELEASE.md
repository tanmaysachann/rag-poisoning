# V24 research release: dataset, RL, transfer, and answer path

This release keeps the detector, PPO attacker, retrieval gate, and evaluation
artifacts separate. The public scanner's `pass` or `review` is a passage risk
signal. The live research answer replay also requires authenticated source
admission and can recover from a separately pinned clean snapshot. Source
identity does not prove factual truth.

## Show the dataset to a mentor

The trained detector uses 14,504 BEIR Natural Questions warm-start training
rows from 2,072 questions, then 1,855 fine-tuning rows from 51 PoisonArena
training questions. The other PoisonArena questions are split into calibration
(20) and previously inspected development evaluation (29). Exact row counts,
labels, family names, upstream archives, and hashes are in
`data/reference_detector_v9/manifest.json`,
`data/multifamily_v20/manifest.json`, and `PRODUCT_V24.md`. The 2,681,468
Natural Questions snapshot passages are an unlabeled source lookup index, not
millions of detector training examples.

For cross-domain transfer, we downloaded official BEIR SciFact test data from
`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`
and pinned its SHA-256 to
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
`data/build_v24_blind_scifact.py` made 1,500 source-labeled cases over 300
questions: 300 qrel-relevant original abstracts, 300 other original abstracts,
and 900 deterministic attacker-authored new-document insertions in three
families. It checked exact question-text nonoverlap with the earlier detector
training, calibration, and evaluation questions. Show
`data/v24_blind_scifact/cases.jsonl` and its `manifest.json` for the actual
rows and their provenance. SciFact qrels measure relevance, not whether a
scientific claim is factually true. These synthetic attack labels come from
how the passages were made, not independent factual adjudication.

## What RL actually does

PPO trains an **attacker policy** in `poison/train.py`, `poison/env.py`,
`poison/policy.py`, and `poison/ppo.py`. It edits a retrieved passage with
masked operation, position, and payload actions; receives step shaping for
validity, relevance, and detector risk; and receives terminal reward from
actual local retrieval and extractive RAG answering. The defender is frozen
within each training run. The V24 detector itself is supervised plus source
integrity; claiming that PPO trained the classifier would be inaccurate.

We trained three 200-episode seeds against V23 containment on 350 MS MARCO
training questions. On the same 75 previously inspected validation questions,
their defended attack successes were **55, 60, and 62**. Fixed substitutions
scored **5, 9, and 4**; random edits scored **20, 20, and 22**. Replaying each
identical PPO attack under the known-document pre-attack ID/hash rule yielded
**0/75 for each seed**. That rule has no expected digest for a newly inserted
document. `results/v24_ppo_multiseed.json` binds the checkpoint, training
trace, baseline, and paired replay hashes; three seeds show variability, not
a statistical superiority result.

## Frozen transfer and the exposed failure

The frozen V24 scanner flagged **488/900 (54.2%)** SciFact synthetic inserted
attacks and **122/600 (20.3%)** original clean abstracts. Its previously
inspected PoisonArena development result of 534/755 attacks and 9/311 clean
passages does not transfer to this new domain at the same error rates. The
legacy V3 classifier flagged 899/900 attacks and 4/600 clean passages on
these fixed templates; that comparison must not be generalized to arbitrary
poisoning. The SciFact generator, model, and threshold were frozen before
scoring. Once these outcomes were read, this set ceased to be an untouched
test for subsequent design choices. See `results/v24_blind_scifact.json`.

The first V24-only answer integration allowed **100/300** attack-sourced
target answers versus **19/300** under V3 on 150 previously inspected MS MARCO
validation questions with two attack strategies. The failure came primarily
from new inserted documents that lacked an expected document hash; the scanner
reviewed only 24/150 inserted attack passages. This initial trace and report
remain in `results/v24_answer_path_validation.*`.

The revised `pipeline/v24_answer_gate.py` scans every retrieved passage,
checks its known ID against a pre-attack hash, sends unadmitted or changed
bytes to review, answers only from passing admitted passages, and can recover
from the independently pinned clean corpus. New legitimate documents require
operator admission to that corpus before they can supply an answer. On the
same 300 inspected cases, attack-sourced target answers were **0/300**,
clean-answer alias recovery **56/300**, snapshot recovery **77/300**, and
abstention **160/300**. V3 clean alias recovery was **28/300**. This is an
engineering replay after seeing the first failure, not an independent blind
performance claim. See `results/v24_answer_path_validation_admitted.json`.

## Reproduce and inspect

From the repository root with the pinned source archive available:

```powershell
python -m data.build_v24_blind_scifact --archive tmp/scifact.zip
python -m evaluation.evaluate_v24_blind_scifact
python -m evaluation.run_v24_ppo_multiseed
python -m evaluation.evaluate_v24_answer_path
python -m unittest tests.test_research_api tests.test_v24_answer_gate -v
```

The builders and report commands refuse to replace a completed output. The
paired answer replay stores a case trace and can resume an interrupted run.
The original answer report is kept separately; the current default output is
the admitted-source revision. Inspect the relevant JSON reports and case
traces rather than rerunning long experiments during a presentation.

The public site is a research console. Its live answer replay uses a benchmark
qrel-selected reference and a hashing extractive answerer on previously
inspected validation cases. A public RAG deployment needs an operational
process to authenticate reference selection and admit document updates. No
result here establishes hosted LLM answer quality or broad factual accuracy.
