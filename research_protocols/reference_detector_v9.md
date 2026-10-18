# V9 source-labelled detector experiment

Recorded before fitting or scoring v9. This is a development experiment: the
v7/v8 replay aggregates are already known. No resulting score is a fresh sealed
test, and no detector is promoted to live quarantine from this experiment.

## Data and labels

Use the 2,302 Natural Questions question IDs in the Architecture Matters archive
that have **no retrieval logs**. Exclude all 1,150 logged questions before
building training data. For each training question, use the first official BEIR
relevance-judged passage as a reference, the published naive attack passage as
the poison example, and five lexically nearest BEIR passages that are not
relevance-judged for that question as hard clean examples. The clean negatives
are *clean by source*, not semantically guaranteed to lack evidence. Do not
place answer keys, target answers, candidate IDs, or attack-family names in
model input. Keep manifests and archive checksums.

The three model labels are `reference`, `clean_other`, and `poison`. They are
**not** agreement, contradiction, and missing evidence labels. Calling them
that would silently invent annotations. Report this as a source-labelled
classifier and scrutinize whether it relies on generator style.

## Model and decision

Fine-tune locally cached MiniLM with question, reference and candidate text.
Hold question and up to 190 reference tokens fixed and score every candidate
window, taking the maximum poison probability. Choose a threshold from only
the 234 logged calibration questions to allow at most 1% clean flags. Apply it
unchanged to the 687 question-disjoint development replay. Report own-poison
catch separately for naive and CorruptRAG-AK, overall clean false flags, false
flags on the reference itself, and latency. Compare to frozen v3 on the same
replay. CorruptRAG-AK is an unseen attack *family* for v9 training, though it
comes from the same collection and questions as the logged replay.

An independently authenticated reference is a prerequisite for real use.
BEIR qrels stand in as an oracle-like benchmark reference only. If reference
provenance cannot be verified, the runtime interface returns `unassessable`.
An origin attestation proves source identity and integrity, not factual truth.

## Independent-generator extension (recorded before viewing scores)

Fetch the six pinned PoisonArena attack files listed in
`scripts/download_poisonarena_v9.py`, its Contriever NQ retrieval file, and its
published query and qrel files. **Question IDs are in a different namespace**:
the same `test58` can mean a different question in the two collections.
Exclude questions whose normalized *text* matches any Architecture Matters
training or logged query, and report that overlap. Do not inflate question
counts by counting five variants of one question as five independent questions.
Use PoisonArena's released qrel as the benchmark reference and recover clean
retrieval passages from the checksummed BEIR corpus archive. Audit whether
the released qrel text contains the published correct answer but do not use
that answer string to select or alter the model reference. Freeze the threshold using only the original
234-question calibration set above. Score each family independently and report
both document catch and question-level any-catch, plus clean false flags on the
same new-question retrievals. This is an independently generated attack-family
check, but the BEIR qrel still supplies an oracle-like reference and the
question selection is limited to the public PoisonArena files.

## Adoption gate

A candidate may be considered for a separate, fresh evaluation only if it
catches at least 50% of each retrieved poison family at at most 1% clean false
flags, without falsely flagging more than 1% of references. The eventual
target is 80% catch on genuinely new independently generated families at the
same false flag budget. This run cannot establish that target. Do not wire it
into the live filter unless a truly new attack collection and a non-oracle
trusted source selection path both pass review.
