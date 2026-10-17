# Detector v5 confirmation protocol

Written after reviewing v4 test results and before evaluating v5.

The v4 model alone failed its preregistered MS MARCO attack improvement gate.
The v3 detector caught a narrative layout missed by the model, while the model
caught more FAQ and correction layouts. The **fixed combined policy** quarantines
a candidate when either detector reaches its already frozen threshold. No
weights or thresholds are refit for this confirmation.

Fresh MS MARCO test questions and passages exclude v1–v4 sources, questions,
and passages. The SQuAD domain test excludes all contexts from the v4 SQuAD
test. The v5 builder writes train and validation files only as an answer pool
for reproducible wrong-answer selection; neither is used for model fitting or
threshold tuning. The attack layouts are unchanged from v4, so v5 tests new
questions and source documents, **not** unseen attack formats.

The primary comparison is the frozen v3 detector against the combined OR gate
on the same top-5 candidates and extractive answerer. Ship the combined gate
only if all conditions hold:

For aggregate attack intervals, bootstrap by question so the four attacks on
one question remain together. Family and clean intervals resample questions.

1. On v5 MS MARCO, aggregate attack-source wrong-answer success is lower and
   the paired bootstrap 95% interval for old minus combined excludes zero.
2. MS MARCO clean alias accuracy is no more than 2 points below old, and the
   paired interval lower bound exceeds -5 points.
3. On v5 SQuAD, aggregate attack success does not increase and clean alias
   accuracy is no more than 2 points below old.

Even a pass supports only this bounded synthetic benchmark. A candidate-only
model has no independent factual reference; a plausible false fact in an
approved source remains a separate trust problem.
