# V17 full-corpus clean-answer gate

Frozen before downloading NQ-Open answer annotations or scoring the selected
questions. This is a new question split within a public benchmark, not a
real-world deployment or a previously unpublished attack-family test.

## Data boundary

Use the already pinned BEIR Natural Questions FTS5 snapshot of 2,681,468
passages (`data/trusted_snapshot_v12/manifest.json`). Obtain the official
NQ-Open original development answer file from the Google Research dataset
repository and record its SHA-256. Match questions by normalized text, reject
ambiguous duplicate answer mappings, and require a BEIR relevance judgment.

Exclude every question ID from the Architecture Matters external replay
(calibration and test), every PoisonArena NQ question, and every question text
in the local v3 NQ challenge. Sort the remaining BEIR query IDs by SHA-256 of
`v17-full-corpus:` plus the ID. The first 200 are calibration; the next 500
are an untouched evaluation split. Save the ID split and source hashes before
opening any answer labels for those questions.

Pre-prediction amendment: the first frozen split
(`full_corpus_v17_split.json`, SHA-256
`e499584cd49c7d8209f7b48df297d979dd7a9b7917bd4dca56163d493b1235a0`)
matched only 456/700 selected question texts to NQ-Open dev. We measured the
join count without producing predictions or examining answer outcomes. Filter
the eligible BEIR questions to those with exactly one normalized question-text
match in NQ-Open dev, using only the `question` field. Then apply the same
hash order to select 200 calibration and 500 evaluation IDs. Retain the first
split as an audit artifact; the replacement split is
`full_corpus_v17_matched_split.json`. No answer string selects a question.

## Frozen clean pipeline

Retrieve 100 passages using only the question and the pinned FTS index. Rerank
with the local pinned Contriever checkpoint using the v13 dense and
reciprocal-rank-fusion methods. Extract spans from each method's top ten with
the pinned RoBERTa NQ reader. Verify that every emitted span is an exact
substring of its indexed, hash-checked source passage. No qrel, answer, or
support document ID may enter retrieval, reranking, or reading.

On the 200 calibration questions alone, select among lexical, dense, and
fusion rankings and top-one, max-QA-five, and two-source-consensus answer
policies. Pick the score threshold maximizing exact answer matches with no
more than 10% nonmatching emitted answers. Resolve ties in the order just
listed. Freeze the choice before scoring the 500 evaluation questions.

## Report and gate

Report exact NQ-Open alias matches, nonmatching emitted answers, abstentions,
qrel recall at 1/5/10, source/span verification failures, and latency. Record
both strict exact match and normalized substring match; use strict exact
match for the primary gate. The clean path is viable for larger poisoning
experiments only if it achieves at least 40% strict exact match and no more
than 10% nonmatching emitted answers on the 500 evaluation questions, with
zero unverified source/span emissions. Otherwise stop and diagnose retrieval
versus reading errors before making a scaled defense claim.

The BEIR corpus is an unreviewed research snapshot. Hash validation proves
identity, not truth. NQ-Open is a public benchmark, and models may have seen
similar material during pretraining. This protocol measures clean utility,
not poisoning robustness or safety in an operational document collection.
