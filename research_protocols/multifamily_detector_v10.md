# V10 multi-family development experiment

Recorded after the v9 independent PoisonArena aggregate was inspected. All
PoisonArena scores from here onward are **development** scores, including the
29-question subset below. There is no untouched confirmation benchmark.

Keep all v9 Architecture Matters training rows. Add PoisonArena's published
AdvDecoding, corpus-poison, GARAG and content-poison passages from questions
where the first 32 bits of SHA256(`v10-pilot:` + question text) modulo 10 are
less than 7. This yields 71 of the 100 questions for the first three families
and 13 of the 20 content-poison questions. Use the released clean top-10
retrievals and qrels for reference/clean examples on those same 71 questions.
Neither the answer strings nor the poison target strings enter model input.

Reserve the remaining 29 questions for evaluation. GasLite and PoisonedRAG
black-box are absent from training altogether; report them as held-out-family
development checks. Report all six attack families, clean false flags, and
reference false flags on the 29 questions. Choose the threshold using only
the 234 clean Architecture Matters calibration questions, at at most 1%
flags, and freeze it before PoisonArena scoring. Compare frozen v3 and v9 at
their prior thresholds on the same 29-question subset.

The qrel is an oracle-like benchmark reference, not live authenticated
provenance. Source labels remain `reference`, `clean_other`, `poison`, not
semantic entailment labels. A useful result would catch at least half of both
held-out families with at most 1% clean false flags; otherwise do not promote.

Diagnostic: on the 29 held-out questions, replace each reference with the next
question's qrel passage without changing the question or candidate, then
rescore clean, GasLite and PoisonedRAG passages at the frozen threshold. Report
how much predictions change. This out-of-distribution ablation probes whether
the model uses the reference; it is not a valid accuracy estimate.
