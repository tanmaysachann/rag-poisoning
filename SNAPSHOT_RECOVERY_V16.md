# V16: pinned-source answer recovery in the live Research Lab

## Decision

The live Research Lab now compares its existing v3 detector with a second,
source-bound answer path. Under the **bounded** threat model where an attacker
can alter or insert retrieved passages but cannot modify the pinned pre-attack
corpus, the new path prevents the tested attack passages from supplying the
final answer and can independently recover an answer from the clean snapshot.
It is visible for every editable validation case with source hashes, selected
documents, the reason for rejection or recovery, and both answer outputs.

This is an engineering improvement against unauthorized retrieval-source
changes, **not a general factual-poison detector**. The pinned benchmark
corpus is not an independently reviewed factual authority. Its hash proves
identity and integrity, not truth. Existing v3 held-out test results do not
evaluate this new path. All v15/v16 results are development evidence on
previously inspected validation questions.

## Why this path

The offline v14 defender PPO run caught 144–145/145 withheld PoisonedRAG
passages but flagged 23–27/290 clean passages and caught 0/145 GasLite.
Training on the same weak source labels did not produce a usable quarantine
policy. The v15 live source-only guard blocked v3's 19 accepted-ingest stealth
failures, but abstained on 148/150 stealth cases. V16 retains that source
check and adds **question-only retrieval** from the pinned clean corpus for
answer recovery. It never consults the benchmark support ID, answer key,
attacker target, or attack family at inference.

## Runtime path

1. Check the clean corpus file against the versioned v3 benchmark SHA-256
   before building the independent retriever or manifest.
2. Run the unchanged main retrieval, integrity check, frozen v3 detector,
   extractive answerer, and leave-one-out audit on the attacked corpus.
3. Check the selected answer's document ID, exact text hash, and cited span
   against the pinned pre-attack manifest. Keep a verified answer.
4. When that source fails or v3 abstains, independently retrieve the clean
   corpus with the question alone. Extract a cited sentence, verify its
   document and span against the same pinned manifest, and emit it. If this
   cannot be done, abstain.

The interface shows undefended, v3-defended, and snapshot-recovered answers
separately. It exposes the actual source hashes and recovery trace. A separate
card shows the three-seed v14 defender PPO result and its failed gate; PPO is
not used in the live decision.

## Development replay

The [v16 protocol](research_protocols/snapshot_recovery_v16.md) was recorded
after v15's abstention failure and before v16 implementation. The replay uses
150 v3 validation questions per family/surface, one saved attack passage per
question, and the same hashing extractor. Exact aggregate counts and case-row
hashes are in [`results/snapshot_recovery_v16.json`](results/snapshot_recovery_v16.json).

| Condition | V3 attack-sourced target answers | Snapshot path attack-sourced target answers | Snapshot clean alias matches | Snapshot abstentions |
|---|---:|---:|---:|---:|
| Greedy insertion, accepted ingest | 0/150 | 0/150 | 28/150 | 80/150 |
| Greedy insertion, post-index tamper | 0/150 | 0/150 | 28/150 | 80/150 |
| Stealth substitution, accepted ingest | **19/150** | **0/150** | **28/150** (v3: 1/150) | 80/150 |
| Stealth substitution, post-index tamper | 0/150 | 0/150 | 28/150 | 80/150 |

On the unmodified clean corpus, exact-alias matches changed from **27/150**
with v3 to **28/150** with recovery; the new guard emitted no answer from an
unverified source. The accepted-ingest stealth path independently recovered
68 answers after v3 rejection or abstention. Of the 19 v3 attack-sourced
target answers, **18 were replaced by a pinned-source answer and one became
an abstention**. The attacker-target string was
absent from all final snapshot-path answers in this replay, even when source
ID was ignored for that check.

The central limitation is visible in the numbers: **28/150 clean alias
matches is still poor answer utility**, and 80/150 clean questions still
abstain. A verified source can contain an irrelevant or incorrect answer.
This should remain a bounded research defense until an independently reviewed
domain corpus and a substantially better clean answerer are evaluated on new
questions and attacks. An attacker who compromises the pinned source itself
is outside this replay's protection.

## Reproduce

```powershell
python -m evaluation.evaluate_snapshot_guard_v15
python -m evaluation.evaluate_snapshot_recovery_v16
```

Both evaluators refuse to overwrite their frozen aggregate and local case
rows. The v15 result records the source-only guard; the v16 result records
independent answer recovery. Local case rows are gitignored, while aggregate
reports and protocols are versionable.
