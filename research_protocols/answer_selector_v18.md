# V18: learned clean-answer selection on a new full-corpus question split

Frozen after v17's 500-question evaluation was scored and diagnosed. Those
v17 questions are spent development data. They may train a new selector, but
cannot supply a fresh result for it.

## Data and model

Use all 700 v17 questions and their answer-blind, source-verified predictions
as development data. Assign 500 to selector training and 200 to threshold
calibration by SHA-256 of `v18-train:` plus the BEIR query ID. Extract one
candidate per unique verified source passage in the union of v17's lexical,
dense, and fusion top tens. The model may see the reader confidence, the three
retrieval ranks, answer length, and exact-answer agreement across different
passages. It must not see a gold answer, qrel, document ID, question ID, attack
label, or source hash as a feature. Use official NQ-Open aliases solely to
label training candidates and to calibrate the final abstention cutoff.

Fit a regularized gradient-boosted binary candidate scorer with a fixed random
seed and fixed hyperparameters. Pick the highest-scored candidate per question.
Choose the cutoff on the 200 calibration questions that maximizes strict exact
matches while allowing at most 10% nonmatching emitted answers. No tuning on
the new 500-question evaluation split.

Pre-evaluation resource gate: compare the learned selector to the frozen v17
dense/consensus policy on the *same* 200 calibration questions. If the learned
selector has no more strict exact matches, or exceeds 20 nonmatching emitted
answers, stop before producing predictions for the new 500 questions. Retain
that split for a future stronger candidate. This gate was added after training
v18 but before looking at the frozen v17 policy's outcome on this particular
200-question calibration set or producing any v18 evaluation predictions.

## Fresh split and comparison

From BEIR questions with exactly one normalized question-text match in
NQ-Open original dev, exclude all prior Architecture Matters, PoisonArena,
v3-challenge, and v17 question IDs/texts. Sort remaining IDs by SHA-256 of
`v18-full-corpus:` plus ID; take the first 500. Freeze IDs and input hashes
before producing predictions or looking at their answer outcomes. Reuse the
same pinned 2,681,468-passage FTS index, Contriever reranker, and RoBERTa
extractive reader. Save predictions before opening answer labels.

On the same 500 questions, score both v17's frozen dense/consensus policy and
the new learned selector. Report exact matches, nonmatching emissions,
abstentions, source/span verification failures, retrieval qrel recall, and
timing. The learned selector passes only if it reaches at least 150/500 strict
exact matches, at most 50/500 nonmatching emitted answers, zero unverified
emissions, and more exact matches than v17's policy on these same questions.
If it fails, do not promote it to the live defense. This remains a public
benchmark clean-answer study, not proof of poisoning robustness.
