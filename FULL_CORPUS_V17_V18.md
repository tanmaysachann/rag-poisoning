# Full-corpus clean-answer evaluation (v17–v18)

## Decision

**The scale gate failed. Neither the v17 reader policy nor the v18 learned
selector is deployed as a real-world poisoning defense.** Both experiments
searched the pinned 2,681,468-passage BEIR Natural Questions corpus, not the
150-passage live Research Lab corpus. Their 500-question evaluation splits are
disjoint from each other and from questions used in earlier external attack
studies. They measure clean answer utility, not attack resistance.

The split protocols and hashes are in
[`research_protocols/full_corpus_v17.md`](research_protocols/full_corpus_v17.md),
[`research_protocols/full_corpus_v17_matched_split.json`](research_protocols/full_corpus_v17_matched_split.json),
[`research_protocols/answer_selector_v18.md`](research_protocols/answer_selector_v18.md),
and [`research_protocols/answer_selector_v18_split.json`](research_protocols/answer_selector_v18_split.json).
Official NQ-Open answers were used only for training/calibration labels and
final scoring; they did not enter retrieval, passage reranking, reader input,
or selector features. The original v17 question split matched only 456/700
questions to NQ-Open dev, so a question-text-only join was frozen before any
predictions. The original split was retained for audit.

## Measured results

| Evaluation | Strict correct | Nonmatching emitted | Abstained | Decision |
| --- | ---: | ---: | ---: | --- |
| V17, 500 new questions | 32/500 | 58/500 | 410/500 | Failed clean gate |
| V18, frozen v17 policy on a second 500 new questions | 60/500 | 46/500 | 394/500 | Paired baseline |
| V18, learned selector on the same second split | 50/500 | 28/500 | 422/500 | Failed clean gate and lost to baseline |

V17 selected dense reranking, two-source consensus, and QA score cutoff
`0.8076929450035095` on its separate 200-question calibration split. Its
predeclared evaluation gate required at least 200/500 strict correct answers,
at most 50/500 nonmatching emissions, and no unverified-source emission. It
missed both answer-quality thresholds. V18 fitted a regularized gradient
boosting selector on 8,839 question/passage candidates from 500 spent v17
questions and calibrated on another 200. It used nine answer-blind features:
QA confidence, three retrieval ranks, answer length, agreement across
passages, agreement confidence, and relative QA confidence. It passed a small
calibration screening comparison, 16 versus 14 exact answers on the same
200 questions, then lost 50 versus 60 on the new 500-question split. It
reduced nonmatching emissions by abstaining more. Its predeclared gate required
at least 150/500 strict correct answers, no more than 50 nonmatching emissions,
and more correct answers than the frozen policy. It failed.

The v17 post-evaluation error funnel explains the bottleneck. An official
answer alias appeared in 444/500 lexical top-100 pools and 370/500 dense
top-ten passage sets. The extractive reader produced at least one exact alias
candidate in 228/500 questions, but the calibrated final policy answered only
32 correctly. These are oracle-style presence counts, **not achievable system
accuracy**; selecting the right candidate without the answer key remains hard.
The relevance-judged passage reached the dense top ten in 269/500 v17 and
256/500 v18 questions. No emitted v17 or v18 answer lacked a verified
indexed source span.

The v18 prediction run took about 1,417 seconds for 500 questions on a local
RTX 3050, including 487 seconds for FTS retrieval, 756 for Contriever
reranking, and 175 for reader extraction. This is batch throughput, not
interactive p95 latency. The first v17 run used resumable caches, so its
final-run timing does not represent a single uncached pass.

Exact aggregate results and frozen output hashes are in
[`results/full_corpus_v17.json`](results/full_corpus_v17.json),
[`results/full_corpus_v17_diagnostics.json`](results/full_corpus_v17_diagnostics.json),
[`results/answer_selector_v18_training.json`](results/answer_selector_v18_training.json),
and [`results/answer_selector_v18.json`](results/answer_selector_v18.json).
Raw BEIR index, downloaded official answers, embedding caches, model artifact,
and per-question predictions remain local and gitignored. Reproduction
requires the published archives and pinned local model checkpoints.

## Claim boundary

This is a public NQ benchmark study. The snapshot is not an operator-reviewed
factual authority, exact alias scoring may miss valid paraphrases, and a hash
does not certify truth. The model may have encountered related Wikipedia
material during pretraining. V17 and v18 did not run an attack replay because
their clean answer paths failed the prerequisite gate. The live 150-passage
snapshot guard remains a bounded source-integrity demonstration; these
experiments provide **no evidence of real-world poisoning robustness**.

The next defensible implementation target is the answer-selection bottleneck:
choose the correct supported span among competing passages without using an
answer key, then validate on new questions and an independently reviewed
document collection before running fresh attacks. Further quarantine-model
or RL training on source labels would not repair the observed clean-answer
failure.
