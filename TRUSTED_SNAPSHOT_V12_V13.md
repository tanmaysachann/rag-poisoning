# Qrel-free reference selection from a frozen corpus (v12–v13)

## Decision

The separate-source path is implemented and measured, but **it did not pass
the predeclared feasibility gate**. Do not use it for live factual-conflict
quarantine or present it as authenticated truth. The live defense and demo
remain unchanged.

The previous v11 pilot showed that a candidate reader often extracted the
attacker's wrong answer, while the supplied benchmark reference yielded a
strict correct span on only 13/29 development questions. V12 and v13 remove
the benchmark qrel from source selection. They test whether a separate frozen
corpus can provide a better reference through normal retrieval.

## What was built

V12 indexes **all 2,681,468 passages** from the pinned official
[BEIR Natural Questions corpus](https://github.com/beir-cellar/beir/wiki/Datasets-available)
in a local SQLite FTS5 database. The 2.69 GB database, source archive, and
individual passages are checksummed. The index contains no PoisonArena
attacks. Search uses question terms only: it first looks for passages
containing all retained terms and fills remaining slots with any-term
matches. The v8 extractive NQ reader then reads at most the top ten passages.

V13 retrieves the lexical top 100 and reranks them with the pinned local
Contriever checkpoint. It compares pure dense order with lexical/dense
reciprocal-rank fusion, then runs the same reader on each method's top ten.
Neither method sees qrels, the published answer, or attacker targets during
retrieval, reranking, or reading. Raw predictions were written before those
labels were opened for calibration and scoring. The 71/29 question split is
the same as v10–v11 and had already been inspected; all numbers remain
development evidence.

The corpus is a **research snapshot**, not an operator-reviewed source
registry. BEIR's own dataset page says it does not vouch for dataset quality.
Hashing establishes snapshot integrity, not the truth of its contents.

## Results

The answer policy and minimum QA score were chosen on the 71 calibration
questions, allowing at most seven nonmatching emitted answers (10% of 71).
The chosen v12 policy was agreement between two top-ten passages at QA score
at least `0.6744293570518494`. V13 selected the same two-passage policy
after dense reranking, at `0.7821144461631775`. These choices were frozen
before the 29-question development outcomes were scored.

| Development measure | V12 lexical top ten | V13 dense-reranked top ten |
| --- | ---: | ---: |
| Judged qrel retrieved at one | 1/29 | 3/29 |
| Judged qrel retrieved at five | 6/29 | 8/29 |
| Judged qrel retrieved at ten | 8/29 | 10/29 |
| Strict correct-answer match | 6/29 | 8/29 |
| Emitted answer without strict match | 6/29 | 5/29 |
| Abstained | 17/29 | 16/29 |

V13 improved all these development counts but failed the frozen gate: at
least **9/29** strict correct answers, at most **2/29** nonmatching emitted
answers, and qrel recall at ten of at least **9/29**. It met only the retrieval
condition. The calibration result was 18 strict correct, seven nonmatching,
and 46 abstentions out of 71; v12 had 17, five, and 49. The chosen policy
therefore did not transfer cleanly even across this small question split.

Strict answer matching has false negatives. In one nonmatching v13 case, the
reader emitted “Peter Cullen” while the benchmark key was “Peter Claver
Cullen.” Several other questions have plausible alternative interpretations.
We retained the frozen strict counts and did not silently relabel cases
after viewing the results. These 29 questions are too few for a strong
accuracy claim, and none of this measures downstream attack success.

## Practical conclusion

The same full-corpus search that improved judged-evidence recall still
produced too many unverified answers. It cannot reliably tell a contradiction
checker what is true. Adding more poison-labelled training data to a detector
will not fix that source-selection problem. A bounded deployment would need
an independently reviewed source collection with explicit document origins,
query-to-source relevance review, and a fresh evaluation in that domain.
When no reliable source answers the question, abstention is the correct
behavior; a fluent candidate cannot be quarantined as false solely because a
weak reference disagrees.

The frozen protocols are
`research_protocols/trusted_snapshot_v12.md` and
`research_protocols/trusted_snapshot_v13.md`. Aggregate reports and
per-question outcomes are `results/trusted_snapshot_v12*` and
`results/trusted_snapshot_v13*`. The 2.69 GB source index and raw predictions
stay local under the gitignored `data/trusted_snapshot_v12/` directory.

Reproduce from the pinned local BEIR archive and the already downloaded
PoisonArena data:

```powershell
python -m data.build_trusted_snapshot_v12 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m evaluation.evaluate_trusted_snapshot_v12
python -m evaluation.evaluate_trusted_snapshot_v13
```

The builders refuse to overwrite a frozen index or prior prediction/report.
Reproduction therefore requires new output paths or a clean checkout.
