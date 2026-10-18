# Semantic conflict feasibility pilot (v11)

## Decision

**The proposed semantic route failed its predeclared development gate.** Do not
train another contradiction model on the current triples or put this scorer
into live quarantine. The current live defense and demo are unchanged.

This was an offline feasibility check on [PoisonArena](https://github.com/yxf203/PoisonArena),
using the released BEIR qrel as an oracle-like trusted-reference proxy. The
29-question development subset had already been inspected in v9/v10, so the
numbers below are diagnostic, not a new blind result. The protocol was frozen
before this run in `research_protocols/semantic_conflict_v11.md`.

## What ran

The pinned v8 Natural Questions extractive reader received each question and
reference passage or candidate passage. It was not given the published correct
answer or attacker target. Two signals were then calibrated on the 71-question
subset's 710 clean top-10 retrieval passages at at most 1% false flags:

1. A different extracted answer from the candidate and reference, weighted by
   the weaker QA confidence.
2. That signal multiplied by the frozen v6 [VitaminC](https://github.com/TalSchuster/VitaminC)
   factual verifier's refutation probability for a claim constructed from the
   candidate's extracted answer.

The thresholds were frozen before scoring the other 29 questions. The model
weights, source manifests, and counts are recorded in
`results/semantic_conflict_v11_pilot.json`. This did not train a new model:
the predeclared rule required a feasible extraction path before investing in
new semantic annotations and fine-tuning.

| Scorer | Calibration clean flags | Development clean flags | PoisonedRAG black-box caught | GasLite caught |
| --- | ---: | ---: | ---: | ---: |
| Answer difference | 7/710 | 0/290 | 3/145 | 5/145 |
| Answer difference + factual refutation | 7/710 | 3/290 | 7/145 | 12/145 |
| Prior v10 passage model | 23/2,340 on its separate calibration set | 2/290 | 2/145 | 108/145 |
| Prior frozen v3 detector | Reported in v10 evaluation | 5/290 | 44/145 | 11/145 |

The predeclared advancement gate was at least half of PoisonedRAG black-box
passages caught with at most 1% clean flags. Neither semantic signal came
close. The factual-refutation variant also flagged 3/290 clean passages
(1.03%) on development, slightly above the budget. These are passage-level
counts, not attack-success or downstream answer measurements.

## Why it failed

The candidate reader's extracted span strictly matched the published attacker
target in **117/145** PoisonedRAG black-box passages. Yet its reference span
strictly matched the published correct answer on only **13/29** development
questions. The answer was a literal substring of the qrel passage on **20/29**
questions, so the reader missed some answer-bearing references; the other nine
qrels did not contain that literal answer. Strict lexical matching can also
miss legitimate aliases. These audits used published answers **after scoring**
and did not provide them to the defense.

The failure is therefore not simply a shortage of poison examples. The
candidate can often expose the substituted answer, but the system cannot
reliably establish what the trusted passage answers. Factual refutation on a
templated claim did not repair that gap. Training a larger classifier on the
same weak reference path would be another shortcut-prone experiment.

## Next decision

A credible next experiment would first establish an independently maintained,
answer-bearing source collection and a query-to-source selection method that
does not use qrels or answer keys. Measure trusted-answer accuracy and
abstention on new questions before adding contradiction-based quarantine.
Where no reliable trusted answer exists, mark the candidate unverified and
abstain. If that reference path succeeds, then curate genuinely reviewed
supported, contradicted, and insufficient-evidence triples and evaluate once
on a newly frozen attack collection. This pilot provides no basis to claim
that fluent poisoned documents are reliably detected today.

Reproduce with the pinned local BEIR archive:

```powershell
python -m evaluation.pilot_semantic_conflict_v11 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
```
