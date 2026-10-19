# Full-corpus text selector experiment (v19)

## Decision

**The larger supervised training run failed the clean-answer gate.** The
trained selector is not deployed in the live defense. No v19 poisoning attack
replay was run, and this experiment does not establish real-world poisoning
resistance.

The 2,681,468 passages were the **retrieval/evaluation corpus**, not the
training set. The model trained on 37,703 extracted answer candidates from
3,700 questions: 3,000 previously unused official NQ Open training questions
plus 700 spent v17 development questions. There were 2,464 exact-alias
positive candidates and 35,239 negatives. The base was pinned MiniLM with a
new binary classification head, fine-tuned for two epochs on an RTX 3050.

## Frozen setup

The question-only v19 split was frozen before retrieval at SHA-256
`1a6fc5013a8d48a516485a7828d083eeebd4bf4b46e9883df63cce6b4df199c9`.
The [protocol](research_protocols/answer_selector_v19.md) records the later
base-weight and loss details specified before reader predictions, training
labels, model weights, or fresh evaluation predictions existed. Training
questions were selected from official NQ Open train, excluding NQ Open dev
question texts and prior external attack texts. The 500 fresh evaluation
questions were selected from NQ Open dev, excluding v17, v18, and prior attack
or challenge questions. The frozen v18 questions supplied 200 spent
calibration questions only.

Question-only retrieval used the pinned 2.68-million-passage FTS index. New
training questions used lexical top ten; spent v17 questions reused their
verified lexical/dense/fusion candidates. The pinned RoBERTa reader generated
proposed answers. Each retained span had to match the exact indexed passage
and its hash. Gold aliases labeled training candidates and calibrated the
abstention cutoff; they were not model inputs or inference features. Fresh
reader and selector predictions were saved before scoring with the fresh
aliases. The trained weights SHA-256 was
`755cd87eb439482e22b8a68473719fd25e0541317dfbacb61bd4a3fff10b3ac9`.

## Measured results

| Split and policy | Exact correct | Nonmatching emitted | Abstained |
| --- | ---: | ---: | ---: |
| Spent 200-question calibration, frozen policy | 27 | 22 | 151 |
| Spent 200-question calibration, trained selector | 20 | 20 | 160 |
| Fresh 500-question evaluation, frozen policy | 30 | 64 | 406 |
| Fresh 500-question evaluation, trained selector | 23 | 64 | 413 |

The predeclared v19 pass rule was at least 150/500 exact correct, at most
50/500 nonmatching emitted answers, zero unverified emissions, and more correct
answers than the paired frozen policy. The trained selector failed three of
these four requirements. There were zero unverified emissions and zero reader
source-span verification failures. The v19 baseline is lower than v18's
60/500, and the question splits differ; only the two v19 rows are a
same-question comparison.

In the fresh 500 questions, the reader produced an exact official answer
candidate somewhere in its verified union for 205 questions. Without the
calibrated abstention cutoff, the text selector ranked a correct candidate
first for 75. Its cutoff reduced correct emissions to 23 while allowing 64
nonmatching emissions. Among the 205 questions where an exact candidate
existed, the selector abstained on 156 and emitted a nonmatching answer on 26.
These are **post-evaluation diagnostics on a spent split**, not attainable
accuracy claims or a new tuning set. They indicate that answer ranking and
confidence remain the observed bottleneck. More training examples of this
kind did not fix it.

The fresh 500-question inference took 583 seconds for lexical retrieval,
747 seconds for Contriever reranking, 174 seconds for 8,841 reader pairs, and
17 seconds for the trained selector. These are batch timings on the local
machine, not interactive latency measurements.

Frozen aggregate results and hashes are in
[`results/answer_selector_v19_training.json`](results/answer_selector_v19_training.json),
[`results/answer_selector_v19.json`](results/answer_selector_v19.json), and
[`results/answer_selector_v19_diagnostics.json`](results/answer_selector_v19_diagnostics.json).
The training/evaluation caches and model weights remain local and gitignored;
reproduction requires the pinned official NQ Open files, BEIR corpus snapshot,
and model checkpoints.

## Claim boundary

Official answer aliases can miss semantically correct paraphrases, and an
alias match is not independent passage-truth review. The BEIR passage snapshot
is hash-pinned but not an operator-reviewed factual authority. Base models
may have encountered related Wikipedia material during pretraining. V19 is a
public clean-QA benchmark, not a poisoning detection result. The live
150-passage snapshot guard and its bounded attack replay remain separate.
