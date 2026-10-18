# Multi-family poisoning detector pilot (v9–v10)

## Decision

**Do not put either new model into live quarantine.** The project now has a
trained, reference-conditioned passage scorer and reproducible cross-collection
evaluations, but neither model meets the frozen rule of at least 50% catch on
each unseen family at no more than 1% clean false flags. The existing live
defense and user-facing demo are unchanged.

## What was built

V9 used 2,302 Natural Questions questions absent from the earlier Architecture
Matters retrieval replay: 2,302 published naive poison passages, 2,302 BEIR
relevance-judged reference passages, and 11,510 lexically hard clean BEIR
passages. The resulting 16,114 source-labelled rows are checksummed. A MiniLM
cross encoder receives the question, reference text, and every window of the
candidate passage. Its three labels mean `reference`, `clean_other`, and
`poison`. These are **source labels**, not annotations of factual agreement,
contradiction, or missing evidence. Answer keys, target strings and document
IDs are absent from model input.

V10 kept those rows and added four [PoisonArena](https://github.com/yxf203/PoisonArena)
attack methods plus their clean retrievals on 71 other questions, for 18,008
rows total. GasLite and PoisonedRAG black-box were withheld entirely from v10
training. The other 29 PoisonArena questions were also withheld from training.
PoisonArena reuses identifiers such as `test58` for different question text, so
the cross-collection overlap check compares normalized **question text**;
none of its 100 questions matched the Architecture Matters questions.

`assess_attested_candidates` requires an explicitly selected reference ID,
matching document hash, and an operator-approved origin before it returns a
score. Without those, it returns `unassessable`. It is an offline research
interface; a reviewed query-to-source selector has not been established.
Source attestation proves origin and snapshot integrity, not factual truth.

## Architecture Matters development replay

All cutoffs were selected on the 234-question clean calibration split at at
most 1% clean flags. The 687-question replay is question-disjoint but its
aggregate v7/v8 results had already been inspected before v9 was designed.
This is development evidence, not fresh confirmation.

| Detector | Retrieved naive poison caught | Retrieved CorruptRAG-AK poison caught | Clean false flags |
| --- | ---: | ---: | ---: |
| Frozen v3 content detector | 43/423 | 24/687 | 45/6,870 |
| V9 naive-trained model | 402/423 | 146/687 | 66/6,870 |

V9 is a strong in-collection naive-style detector and an improvement over v3
on CorruptRAG-AK, but still misses nearly four in five CorruptRAG-AK passages.

## Independent-generator check and v10 revision

On all 100 PoisonArena questions, v9 caught just 30/2,600 generated attack
passages across six files and falsely flagged 16/1,000 clean retrieval
passages. The frozen v3 model caught 231/2,600, mostly one family, at 13/1,000
clean false flags. Thus the attractive in-collection v9 result did not
transfer. These attack files contain generated poison passages; this check
scores the passages themselves, not actual retrieval or downstream answer
success.

V10 was then trained on four PoisonArena methods. Its threshold
`0.9976394176483155` was frozen from the **separate** Architecture Matters
calibration split, where it flagged 23/2,340 clean passages. On the 29
training-question-disjoint PoisonArena questions:

| Method | V10 passages caught | V9 passages caught | V3 passages caught |
| --- | ---: | ---: | ---: |
| AdvDecoding (trained method, new questions) | 29/145 | 0/145 | 1/145 |
| Corpus Poison (trained method, new questions) | 60/145 | 0/145 | 0/145 |
| GARAG (trained method, new questions) | 2/145 | 0/145 | 1/145 |
| GasLite (**unseen method**) | **108/145** | 0/145 | 11/145 |
| PoisonedRAG black-box (**unseen method**) | **2/145** | 1/145 | 44/145 |

V10 falsely flagged **2/290** clean passages and **0/29** judged references
on those questions. V9 flagged 1/290 clean passages; v3 flagged 5/290. The
GasLite improvement is useful evidence that method diversity in training can
help, but PoisonedRAG black-box remains almost entirely undetected. The
29-question subset is development data because the full PoisonArena aggregate
was inspected before v10 design. It cannot support a fresh confirmation claim.

The released PoisonArena qrel text contained its published answer as a strict
normalized substring for 78/100 questions. That leaves substantial reference
quality uncertainty. Answers were used for this audit only, never as scorer
input or reference selection. The qrels are an oracle-like benchmark proxy,
not authenticated sources available to the live product.

## Does the model use its reference?

On 29 held-out questions, swapping each reference for another question's
reference changed **29/580** clean, GasLite and PoisonedRAG decisions. GasLite
catch fell from **108/145** to **89/145**. The mean absolute poison-probability
change was 0.125. This shows some reference dependence, while the 89 remaining
GasLite flags with a wrong reference show that style cues still drive much of
the detection. Shuffled references are out of distribution, so this is a
diagnostic, not an accuracy estimate.

## What the result requires next

The missing ingredient is a curated semantic training set that distinguishes
factual disagreement from benign off-topic or historical text. The source
labels used here cannot supply that distinction automatically. Reviewers
should annotate hard clean contradictions and covert answer substitutions
against independently selected, answer-bearing trusted passages. Then train a
semantic contradiction model and confirm it once on a genuinely new attack
collection, reporting each family and clean false flags. A larger pile of
naive-style poisons would repeat v9's shortcut.

## Reproduction

The raw downloaded archives, training rows and model weights are local and
gitignored. Scripts and aggregate results are versionable. With the previously
downloaded BEIR archive under the system temp directory:

```powershell
python -m data.build_reference_detector_v9 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m detect.train_reference_conditioned_v9
python -m evaluation.evaluate_reference_detector_v9
python -m scripts.download_poisonarena_v9
python -m evaluation.evaluate_poisonarena_v9 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m data.build_multifamily_v10 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m detect.train_reference_conditioned_v9 --data data/multifamily_v10 --output artifacts/multifamily_v10 --report results/multifamily_v10_training.json
python -m evaluation.calibrate_multifamily_v10
python -m evaluation.evaluate_poisonarena_v9 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip" --model artifacts/multifamily_v10 --calibration-report results/multifamily_v10_calibration.json --output results/multifamily_v10_poisonarena_heldout.json --heldout-v10 --compare-v9
python -m evaluation.ablate_reference_v10 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
```

Sources: [Architecture Matters attack archive](https://zenodo.org/records/19582218),
[official BEIR dataset index](https://github.com/beir-cellar/beir/wiki/Datasets-available),
[PoisonArena repository](https://github.com/yxf203/PoisonArena).
The protocols are in `research_protocols/reference_detector_v9.md` and
`research_protocols/multifamily_detector_v10.md`; the raw aggregate reports
are under `results/reference_detector_v9_*.json` and
`results/multifamily_v10_*.json`.
