# V6 factual verifier protocol

Frozen before opening `data/factual_verifier_v6/test.jsonl` for model evaluation.

## Question

Given a claim extracted from a retrieved passage and **independently trusted**
evidence, can a small model identify contradiction while rarely rejecting
supported or unsupported claims? This is a prerequisite experiment for factual
poison screening. It is not an end-to-end RAG defense result.

## Data

Source: [VitaminC](https://github.com/TalSchuster/VitaminC), official 22 MB
archive, SHA256 `49d82dc1690cbee420d18e2c26f687a7937710bb211845d2571430dfd4dc0337`.
Use only its `revision_type=real` pairs. The source dataset annotates whether
Wikipedia evidence supports, refutes, or does not establish a claim. The
annotations incorporate Wikipedia material and have the data license described
in the [source repository](https://github.com/TalSchuster/VitaminC/blob/main/DATA_LICENSE).
Raw and derived examples stay local and are excluded from Git.

`data/build_factual_verifier_v6.py` samples one row per source case, with
official train/test pages kept disjoint. Development pages are assigned by a
fixed page hash to either validation or calibration. The builder checks for
page, source-case, and exact-pair overlap across all four splits.

| Split | Supports | Refutes | Not enough info |
| --- | ---: | ---: | ---: |
| Train | 15,000 | 10,000 | 5,000 |
| Validation (epoch choice) | 2,000 | 1,500 | 2,000 |
| Calibration (threshold) | 5,000 | 1,000 | 1,000 |
| Sealed test | 5,000 | 1,000 | 1,000 |

## Model and decision

Fine-tune cached `sentence-transformers/all-MiniLM-L6-v2` on the ordered
`(claim, evidence)` pair for three labels. Choose the epoch with the highest
validation macro F1. Set the refutation-score threshold on calibration so no
more than 1% of either SUPPORTS or NOT ENOUGH INFO cases are marked refuted.
The test set is read only by the separate evaluator after model selection and
calibration are complete.

The model's refutation score is **not** the probability that a document was
maliciously planted. A contradictory claim might be a genuine correction,
outdated source, or an error in the purported trusted evidence. The intended
runtime response is to quarantine the candidate for review only when the
trusted reference is independently authenticated and the candidate makes the
same claim in the same temporal context. Unsupported claims are not called
poisoned.

## Decision rule for next stage

Advance to a RAG integration experiment only if the sealed test refutation
recall is at least 50% while false quarantine is below 1% separately on
SUPPORTS and NOT ENOUGH INFO. Report all three rates and intervals regardless
of the outcome. This is a development gate, not the final project target.

Afterward, a separate protocol must test independently authored RAG poison
passages, trusted-reference retrieval, clean answer accuracy, attack-induced
wrong answers, and prompt-injection attacks. No public or live quarantine
decision should be changed on the strength of VitaminC alone.

## Reproduction

Download the official archive to `artifacts/vendor/vitaminc.zip`, then run:

```powershell
python -m data.build_factual_verifier_v6
python -m detect.train_factual_verifier_v6
python -m evaluation.evaluate_factual_verifier_v6
```
