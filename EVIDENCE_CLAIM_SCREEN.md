# Evidence-backed document screening

The Research Lab now compares every retrieved passage with its **separate
preattack corpus** before a new answer path uses it. It quarantines a passage
that was inserted or changed after that corpus was pinned, shows the original
and changed sentence for same-ID substitutions, and answers from verified
passages with the existing snapshot recovery fallback. The frozen v3 detector
and its answer remain visible alongside this path.

This is an authorization boundary, not a general factual-poison classifier.
The benchmark corpus was not independently fact checked. A legitimate new
document also stays out of the strict path until it is reviewed and added to
the trusted source collection. An attacker who can alter that collection or
its manifest falls outside this guarantee.

## Measured on the previously inspected validation split

`python -m evaluation.evaluate_evidence_claim --model-diagnostic --qa-diagnostic`
produced [`results/evidence_claim_screen_validation.json`](results/evidence_claim_screen_validation.json):

| Passage set | Excluded by pinned-source screen |
| --- | ---: |
| 150 unchanged source passages | 0 |
| 150 new query-matched attack passages | 150 |
| 150 same-ID answer substitutions | 150 |

The model diagnostics make the remaining factual-detection gap explicit. The
published VitaminC ALBERT model gave a refutation probability of at least
0.5 to only **16/150** changed sentence pairs. A question-answering model
extracted different answer spans from **106/150** source/substitution pairs,
but only **50/150** differences had both span confidences at least 0.05.
There are no reviewed legitimate revisions in this replay to calibrate a
model-based quarantine threshold, and these counts must not be reported as
poison-detection recall at an acceptable false-positive rate.

The live replay in [`results/evidence_claim_live_validation.json`](results/evidence_claim_live_validation.json)
reran the **19 previously observed** accepted-ingest stealth failures. The
old detector still returned 19 attack-sourced answers; source screening
excluded all 19 retrieved substitutions and returned zero attack-sourced
answers. It returned a clean alias in 13 cases and abstained once. This is a
development replay selected for known failures, not fresh generalization.

## The next evidence requirement

To detect a false fact in a newly approved document, the system needs a dated,
reviewed answer-bearing source maintained outside the attacker's write path.
Its coverage and claim extraction must be measured before training a
reference-conditioned model. Source labels, benchmark answer keys, and a hash
of an attacked corpus do not supply that factual authority. An unseen attack
collection and realistic clean updates must then set the model's operating
threshold and measure downstream answer quality.
