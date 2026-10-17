# Detector pilot v4 protocol

This protocol was written before running `evaluation.evaluate_detector_pilot`.

## Question

Does a fine-tuned MiniLM query–passage classifier reduce downstream wrong-answer
attack success compared with the frozen v3 eight-feature detector while keeping
clean answer quality? The experiment changes only the document gate. Retrieval,
top-5 candidates, and the extractive answerer are identical within each pair.

## Data and separation

- MS MARCO v1.1 source revision `a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`.
- SQuAD source revision `7b6d24c440a36b6815f21b70d25016731768db1f`.
- 1,200 new MS MARCO train questions, 150 validation questions, 150 untouched
  MS MARCO test questions; all exclude v1–v3 source IDs, questions, and passages.
- 100 SQuAD validation questions with distinct contexts are evaluation only.
- Threshold is set to at most 5% false flags on clean validation examples.
  Neither test split may affect model selection or threshold.

Training positives: answer replacement, false FAQ, and false addendum. Training
negatives: original support, a correct-answer FAQ, and a similar legitimate
other passage. The correct FAQ is intentionally hard: the model has no factual
reference beyond its own parameters. Test attacks use FAQ record, correction
notice, catalog record, and narrative claim layouts. All are synthetic and
locally authored; only their queries and source passages are newly held out.

## Primary measures and adoption rule

For each split, report clean exact alias match, attack-source wrong-answer
success for every attack family, attack retrieval rate, abstention, and detector
latency. Compare old and new decisions on exactly the same retrieved candidates.
Use paired bootstrap intervals for differences and Wilson intervals for rates.

Adopt the new detector only if **all** of these hold:

1. On the new MS MARCO test, aggregate wrong-answer attack success is lower
   than v3 and the paired 95% interval for old minus new excludes zero.
2. Clean alias-match accuracy on MS MARCO is no more than 2 percentage points
   below v3, and the paired 95% interval lower bound is above -5 points.
3. On the SQuAD domain test, aggregate attack success does not increase and
   clean alias-match accuracy is no more than 2 points below v3.

Regardless of results, this is not evidence of general factual verification:
the attack examples are synthetic, SQuAD only changes document domain, and a
query–passage classifier lacks a trusted fact source. A failed gate leaves the
existing deployed detector untouched.
