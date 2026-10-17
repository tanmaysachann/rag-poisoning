# Trusted-reference answer pilot (v8)

## Why this experiment exists

The [v7 external replay](EXTERNAL_RAG_V7.md) showed that the trained factual
verifiers and the existing content detector missed most retrieved poisoned
passages. This pilot asks a narrower question: **if an independently trusted,
answer-bearing passage is available, can the system answer from it instead of
letting a retrieved attack passage determine the answer?**

The answer in this benchmark is partly yes, with a major caveat. BEIR relevance
judgments identify the reference passage. That is an **oracle-like research
proxy**, not a source that the live product can automatically authenticate or
select. The v7 test aggregate was already known when this pilot began, so its
687-question split is question-disjoint but is not a fresh sealed benchmark.
The [v8 protocol](research_protocols/trusted_reference_v8.md) records the
calibration decisions and that limitation.

## Implemented path

`generation/trusted_reference.py` provides a local extractive reader and a
fail-closed entry point. The entry point accepts a source only when its text
hash matches a separately reviewed `SourceAttestations` record and its origin
is in an explicit allowlist. It checks that the returned answer is an exact
span of that source and abstains when the source or adequate span is missing.
An attestation proves origin and snapshot integrity, **not factual truth**.

The benchmark evaluator uses the BEIR judged passage in place of that live
attestation. It never passes the published correct or attacker-target answer
to the reader. Those strings are used only to score output. It also runs the
same reader on the logged top-10 candidates to compare a source restriction
with unrestricted answer selection.

We compared two readers on 234 calibration questions. The selected
`nlpconnect/roberta-base-squad2-nq` reader had 82 strict correct-answer
matches, 20 wrong emitted answers, and 132 abstentions at its frozen
confidence cutoff of `0.24418172240257263`. The smaller
`deepset/tinyroberta-squad2` reader had 68, 14, and 152 respectively. The
selection criterion was correct matches minus twice wrong emissions. The NQ
reader's published training description does not establish that none of these
BEIR questions appeared in its training data, so the result must not be called
model-training-disjoint.

## Question-disjoint replay

The table uses normalized answer substring matching. It can miss aliases and
paraphrases. “Wrong” here means **no strict match to the published answer**;
some may be valid alternatives. Fully ambiguous questions were excluded by
the archive, but partially ambiguous questions remain.

| Retrieval | Answer policy | Correct match | No strict match emitted | Attacker target match | Abstained |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean | Highest-score span from all top 10, no cutoff | 218/687 | 469/687 | 17/687 | 0/687 |
| Naive attack | Highest-score span from all top 10, no cutoff | 179/687 | 508/687 | 109/687 | 0/687 |
| CorruptRAG-AK attack | Highest-score span from all top 10, no cutoff | 259/687 | 428/687 | 95/687 | 0/687 |
| Clean | Judged reference only, frozen cutoff | 257/687 | 79/687 | 8/687 | 351/687 |
| Naive attack | Judged reference if still in top 10 | 254/687 | 78/687 | 8/687 | 355/687 |
| CorruptRAG-AK attack | Judged reference if still in top 10 | 241/687 | 75/687 | 8/687 | 371/687 |

The unrestricted baseline's cutoff was selected independently on **clean
calibration** cases using the same wrong-answer penalty. Its frozen cutoff
`0.9833638668060303` left it abstaining on 663/687 clean test questions,
with 16 correct matches and 8 nonmatches. The no-cutoff rows show what happens
when that reader is required to answer; they are not a calibrated deployment
policy. The reference-only policy has a better correctness/abstention balance
on this benchmark, but it relies on being handed the judged source.

The judged reference remained in the attacked top 10 for 670/687 naive and
640/687 CorruptRAG-AK questions. The inline policy abstains when attacks
displace it. A hypothetical separate trusted channel would keep the clean
reference result of 257 correct, 79 nonmatches, and 351 abstentions in all
three conditions; that channel has not been built or validated for the live
product.

## Can it quarantine the poisoned documents?

No. At the confidence cutoffs used here, the reader found a different answer
from the reference in only 3 retrieved naive poison documents and 1 retrieved
CorruptRAG-AK poison document on the test split. Some clean non-reference
passages also disagreed. This signal cannot reliably label a document
malicious. The useful behavior is restricting *which source may supply the
answer*, provided that source can genuinely be selected and authenticated.

The reference-only reader still abstained on 51% of clean questions and
emitted 79 answers without a strict published-answer match. It does not pass
a credible live adoption gate. The code remains an offline research path and
does not change the deployed answer flow.

## What this changes in the project direction

The project should define a bounded domain with operator-reviewed source
origins, a signed corpus snapshot, and a query-to-source retrieval path that
does **not** consult benchmark relevance labels or answer strings. Its next
evaluation must include attacks added before the snapshot is signed, attacks
that compromise a nominally trusted source, and a new attack family. If no
authenticated relevant source is found, abstention is the honest outcome.
Further tuning a generic poison classifier on the current archive would not
resolve this trust-selection problem.

## Reproduction

The pinned readers' weights stay under ignored `artifacts/vendor/`. With the
v7 case files present, run:

```powershell
python scripts/download_reference_readers_v8.py --include-tiny
python -m evaluation.evaluate_trusted_reference_v8 --split calibration --reader roberta_nq
python -m evaluation.evaluate_trusted_reference_v8 --split test --reader roberta_nq
python -m evaluation.evaluate_trusted_replay_v8 --split calibration
python -m evaluation.evaluate_trusted_replay_v8 --split test
```

Aggregate results are in `results/trusted_reference_v8_roberta_nq_calibration.json`,
`results/trusted_reference_v8_roberta_nq_test.json`,
`results/trusted_replay_v8_calibration.json`, and
`results/trusted_replay_v8_test.json`.
