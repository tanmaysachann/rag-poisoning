# V19: text based answer selector on a fresh full corpus split

The question-only split (SHA-256
`1a6fc5013a8d48a516485a7828d083eeebd4bf4b46e9883df63cce6b4df199c9`)
was frozen before creating v19 retrievals. The base-weight pin, context window,
and class-loss weighting were specified while training-question retrieval was
running, before reader predictions, training labels, model weights, or fresh
evaluation predictions existed. This is a clean Natural Questions benchmark
experiment. It does not measure poisoning resistance.

## Data boundary

Use the official NQ Open train file (87,925 rows, SHA-256
`0c93aaf13b12f855628489665e4ef85ffb8573ad444646f11f4fabdc57a0aabf`)
from `google-research-datasets/natural-questions/nq_open/NQ-open.train.jsonl`
and the previously pinned official NQ Open dev file (SHA-256
`f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616`).
Select 3,000 unique train questions by SHA-256 of `v19-train:` plus normalized
question text. Exclude all dev question texts, ambiguous duplicate train texts,
and prior external-attack question texts. Use all 700 spent v17 questions as
additional training data. Do not train on any v18 question.

Of the spent v18 questions, select 200 by SHA-256 of `v19-cal:` plus question
ID for threshold calibration. The other 300 may be used only for diagnosis.
For the untouched evaluation, select 500 unique NQ Open dev questions by
SHA-256 of `v19-eval:` plus normalized question text. Exclude all v17 and v18
question texts and all prior external-attack/challenge question texts. Freeze
the question-only split and input hashes before v19 predictions or labels are
opened. No gold answer may enter retrieval, reranking, reader input, model
features, or inference.

## Candidate and model path

Reuse the pinned 2,681,468-passage BEIR NQ FTS index, Contriever checkpoint,
and RoBERTa extractive reader. For each new NQ Open training question, retrieve
the lexical top 10 and run the reader on each passage. Save the exact indexed
source hash and only retain answers verified as exact source spans. For the
spent v17 questions, reuse the frozen v17 union of lexical, dense, and fusion
top ten candidates and the corresponding verified indexed passages. Label a
candidate positive if its normalized answer exactly matches any official
answer alias. Retain all positives and the highest reader-score wrong answers,
at most ten negative candidates per question. These labels are noisy proxies
for answer correctness, not passage truth annotations.

Fine-tune the locally cached `sentence-transformers/all-MiniLM-L6-v2` as a
binary cross encoder (revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, model weights SHA-256
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`).
The encoder starts from that checkpoint; its binary classification head is
initialized for this task and the full model is fine-tuned. Its inputs are
question plus proposed answer and a
source-passage window centered on that answer. Fix seed 19, max length 256,
batch size 8, two epochs, AdamW learning rate 2e-5, weight decay 0.01, and
linear warmup/decay. The passage window is 400 characters on each side of the
first exact answer occurrence. Cap the positive class loss weight at 10, using
the observed negative-to-positive training ratio. Use the final epoch's model.
At inference, score each verified candidate from the
same union of the three top-ten rankings and choose the highest model score;
use reader score only to resolve exact ties. Calibrate one abstention threshold
on the 200 spent v18 questions, maximizing strict matches subject to at most
20 nonmatching emissions. Do not change model, data selection, or threshold
after seeing the untouched evaluation outcomes.

## Comparison and gate

On the same fresh 500 questions, compare the trained selector to the frozen
v17 dense/consensus10 policy and its frozen cutoff. Report strict matches,
nonmatching emissions, abstentions, candidate oracle presence, source/span
failures, timings, and the model/data hashes. The v19 selector passes only
with at least 150/500 strict matches, at most 50/500 nonmatching emissions,
zero unverified emissions, and more correct answers than the paired frozen
baseline. If it fails, do not promote it to the live defense or run a scaled
poisoning claim. Even a pass requires fresh attack-family replay and review of
the document authority assumptions before any real-world claim.
