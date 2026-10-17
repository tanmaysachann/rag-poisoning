# External RAG poisoning evaluation (v7)

## What was replayed

The independent [Architecture Matters attack archive](https://zenodo.org/records/19582218)
contains the generated poisoned passages and top-10 retrieved document IDs for
Natural Questions. Its logs omit the original passage text. We recovered those
passages from the official [BEIR NQ corpus](https://github.com/beir-cellar/beir/wiki/Datasets-available)
after checking the published archive MD5. The dataset builder extracts only the
8,928 passages needed for this replay; raw source archives and case files stay
local and are gitignored.

We followed the [frozen protocol](research_protocols/external_rag_v7.md):
excluded the 229 questions marked fully ambiguous by the archive; hashed the
remaining 921 question IDs into 234 calibration and 687 test questions; and
selected the first relevance-judged passage in each **clean** top 10 as a
reference. All 921 had one. The reference is a benchmark aid, **not an
authenticated source available to the live app**. Answer strings did not select
references, claims, thresholds, or model inputs.

The archive's `poison_retrieved` flag can refer to another question's poisoned
passage. We instead check whether each question's *own* poison ID appears in
its logged top 10. That happens for 565/921 naive attacks and 921/921
CorruptRAG-AK attacks. Another question's poison remains in the candidate set
and is included in a separate all-poison count in the raw result.

## Held-out results

Each reference-aware verifier threshold was fixed from **clean calibration
documents only**, allowing at most 1% to be quarantined. The v3 content
detector kept its existing threshold. None of the models met the predeclared
50% catch requirement on either attack family.

| Detector | Naive own poison caught | CorruptRAG-AK own poison caught | Clean passages falsely quarantined | Approx. scoring time per unique test pair |
| --- | ---: | ---: | ---: | ---: |
| Locally fine-tuned MiniLM factual verifier | 2/423 (0.47%) | 0/687 (0%) | 17/6,870 (0.25%) | 1.93 ms, GPU |
| Published VitaminC ALBERT verifier | 13/423 (3.07%) | 2/687 (0.29%) | 49/6,870 (0.71%) | 13.88 ms, GPU |
| Existing frozen v3 content detector | 43/423 (10.17%) | 24/687 (3.49%) | 45/6,870 (0.66%) | 3.20 ms, CPU |

The VitaminC result is substantially worse than its in-family development
score. Calibration examples show why one part of our representation fails:
some attack passages quote the true claim as an allegedly outdated view, then
state the false claim later. The single selected sentence can be the quoted
true claim. A **post-test, calibration-only** check scoring the top 1, 2, or 3
sentences did not fix the problem: all three variants caught 0/234
CorruptRAG-AK attacks at the 1% clean false flag limit. These variants were
not evaluated as a new sealed test.

## Downstream answer replay

The separate local extractive answerer replayed the same retrieved sets before
and after the frozen v3 detector. A normalized exact substring of the published
answer is the conservative match rule; synonyms and paraphrases can be missed.
On clean cases, the correct answer was matched in 61/687 before filtering and
59/687 after. The answerer returned insufficient evidence on 517/687 clean
cases before filtering. With naive attacks, target matches changed 49 to 38;
with CorruptRAG-AK, 57 to 54. For CorruptRAG-AK, correct-answer matches fell
from 194 to 177. This is a weak local answerer and **does not reproduce the
archive's original LLM attack-success rate**. The defense does not show a
convincing end-to-end benefit here.

## Decision and limits

**Do not promote these verifiers into live quarantine.** They meet the clean
false flag budget but miss almost all retrieved poisoned passages. V3 is the
strongest of these three on this archive, yet it still catches only 10% of
retrieved naive attacks and 3.5% of CorruptRAG-AK attacks.

The benchmark reference is relevance-judged, not independently authenticated;
real deployments would need a trusted source or provenance chain. The replay
uses archived retrieval IDs and attack text, and cannot reproduce the original
LLM generation without rerunning that model. The attack families come from one
generator collection; future model training on them would require an additional
unseen-family benchmark before making broad claims.

## Reproduction

Run `python scripts/download_beir_nq.py` to get and checksum the 498 MB BEIR
archive in the system temporary directory. Keep the architecture archive under
`artifacts/vendor/rag-poisoning-architecture-bench-data.zip`, then run:

```powershell
python -m data.build_external_rag_v7 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m evaluation.evaluate_external_rag_v7
python -m evaluation.explore_multisentence_v7
python -m evaluation.evaluate_external_rag_v7_answer
```

The case manifest and per-question passages are under ignored
`data/external_rag_v7/`. Aggregate results are
`results/external_rag_v7.json`,
`results/external_rag_v7_multisentence_dev.json`, and
`results/external_rag_v7_answer_v3.json`.
