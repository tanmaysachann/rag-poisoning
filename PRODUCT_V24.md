# Sentinel RAG V24: trained poisoned-document scanner and adaptive PPO loop

V24 is a runnable **research product**, locally and through a separate hosted
scanner. It scores a question, a supplied
reference, and a candidate passage with a trained detector; searches a separate
hash-pinned source snapshot for changed or embedded copies; and optionally
checks the exact SHA-256 expected for a known document ID. It returns `review`
or `pass` with all three reason codes.

Run it from this repository's root:

```powershell
python -m scripts.scan_poison_v24 --input data/multifamily_v20/evaluation.jsonl --limit 3
```

JSONL input needs `query`, `reference`, and `candidate`. A row may also contain
`expected_sha256` **only if that digest comes from an independently pinned
ID-to-hash map**. For a single document:

```powershell
python -m scripts.scan_poison_v24 --query "Who wrote the book?" --reference-file reference.txt --candidate-file candidate.txt --expected-sha256 PINNED_64_CHARACTER_SHA256
```

If no independently pinned digest exists, omit it; the learned and containment
branches still run. If no separate pre-attack snapshot exists, add
`--no-snapshot`. These inputs must be authenticated by the caller. A benchmark
qrel is not that authentication. The scanner does not auto-select a trusted
reference or prove that any passage is true. The research console has a
separate, explicitly selected V24 answer replay with authenticated source
admission; the Secure Query path retains its own policy. The [scanner](detect/product_v22.py),
[CLI](scripts/scan_poison_v24.py), and
[calibration bundle](artifacts/product_v24_calibration.joblib) are present here.

## Exactly what data trained it

| Role | Source | Rows and grouping |
| --- | --- | --- |
| MiniLM warm-start source | [BEIR Natural Questions](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip) with published naive poison passages from the [Architecture Matters attack archive](https://github.com/samkorn/rag-poisoning-architecture-bench) | **16,114 labeled candidates, 2,302 questions**: 2,302 references, 11,510 other clean passages, 2,302 poisons |
| Six-family fine-tune | Pinned [PoisonArena](https://github.com/yxf203/PoisonArena) NQ attacks with BEIR clean retrievals | **1,855 rows, 51 training questions**: 545 clean/reference and 1,310 published attack passages |
| Threshold calibration | Separate PoisonArena questions | **749 rows, 20 questions**, including 214 clean/reference passages |
| Development evaluation | Separate PoisonArena questions | **1,066 rows, 29 questions**, including 755 attack and 311 clean/reference passages |
| Character fluency | Clean/reference V9 texts | **13,812 documents**, used to fit a char trigram model |
| Source integrity | Pinned BEIR NQ corpus | **2,681,468 passages**, used as a read-only lookup snapshot, **not** as labeled training examples |
| RL attacker | Frozen MS MARCO benchmark | **350 training questions** and **75 validation questions**, with separate train/validation pre-attack corpus snapshots |

The six PoisonArena families are AdvDecoding, Content Poisoning, Corpus
Poisoning, GARAG, GasLite, and PoisonedRAG black-box. Questions do not cross
the three PoisonArena splits. The [dataset manifest](data/multifamily_v20/manifest.json)
records every row hash and source count; the [PoisonArena manifest](data/poisonarena_v9/manifest.json)
pins repository commit `001da5aefa1779f3d2cd1495e0a2d9d614a2fa81`
and file hashes. The [BEIR snapshot manifest](data/trusted_snapshot_v12/manifest.json)
pins the source archive and 2.69 GB FTS database. Source labels mean
"attacker-supplied" or "benchmark clean," not independent factual review.
The V9 MiniLM actually trained on **14,504** of its 16,114 rows from **2,072**
questions; its remaining **1,610 rows/230 questions** were validation. V24
then fine-tuned that checkpoint on the **1,855** PoisonArena training rows.

This is substantial local training data, but the six-family set has only 100
unique questions. They were inspected before this work. The 29-question
result is **development evidence**, not a blind generalization claim.

## Show the actual dataset

The local JSONL files contain the question, reference passage, candidate
passage, source label, and (for the six-family set) attack family. Open
`data/reference_detector_v9/train.jsonl` for the warm-start source and
`data/multifamily_v20/{train,calibration,evaluation}.jsonl` for the three
question-disjoint PoisonArena splits. The adjacent `manifest.json` files give
source hashes, row counts, and label counts. Raw pinned PoisonArena files are
under `data/poisonarena_v9/`. The 350/75 MS MARCO PPO questions are in
`data/benchmark/{train,validation}/queries.jsonl`.

From the repository root, these PowerShell commands show the manifest, verify
the three processed row counts, and print one actual poison example:

```powershell
Get-Content data/multifamily_v20/manifest.json
@('train','calibration','evaluation') | ForEach-Object { "$_`: $((Get-Content "data/multifamily_v20/$_.jsonl" | Measure-Object -Line).Lines) rows" }
Get-Content data/multifamily_v20/train.jsonl | ConvertFrom-Json | Where-Object { $_.label -eq 'poison' } | Select-Object -First 1 query,family,reference,candidate | Format-List
```

The 2,681,468-passage BEIR index is a separate source lookup, **not** 2.68
million labeled detector-training examples. The warm-start poison passages
come from the published Architecture Matters attack archive; their local
build is specified in `data/build_reference_detector_v9.py`.

## Detector and measured result

The learned branch is a reference-conditioned MiniLM plus a gradient boosting
model over eight V3 statistical/behavior signals, V3 risk, and four clean
trained char fluency signals. Their scores are fused by clean calibration
upper-tail probabilities. The integrity branch searches the beginning,
middle, and end of a candidate for a changed pinned passage; it detects both
near-copies and an original source embedded in longer inserted text. The
optional identity branch checks the exact expected SHA-256 for a known source
ID. Any branch can request review; each keeps its own reason code.

At the threshold selected using the 214 clean calibration passages, on the
previously inspected 29-question development split:

| Scanner | Attack passages flagged | Clean passages flagged |
| --- | ---: | ---: |
| Learned fusion | 409/755 | 9/311 |
| Source containment | 133/755 | 0/311 |
| **V24 combined** | **534/755 (70.7%)** | **9/311 (2.9%)** |

V24 caught 43/145 AdvDecoding, 27/30 Content Poisoning, 95/145 Corpus
Poisoning, 113/145 GARAG, 111/145 GasLite, and 145/145 PoisonedRAG
black-box passages. Calibration flagged 2/214 clean passages; the observed
development false alarm rate was **2.9%**, above the nominal 1% target.
The same confusion matrix is TP 534, FP 9, FN 221, TN 302: **78.4% binary
accuracy**, versus **70.8%** for always predicting the majority attack class.
This is a different metric and dataset from the earlier reported "30.3%," so
the two figures cannot be presented as a paired improvement.
Identity was not scored in these 1,066 rows because they do not have an
operator-authenticated expected hash. See [V24 aggregate](results/product_v24.json),
[learned fusion](results/fusion_v21.json), and
[containment](results/snapshot_containment_v23.json).

## What "we implemented RL" means

RL is the **attacker**, following the document-edit plan in
`Presentation.pdf`. The [environment](poison/env.py) is an edit MDP with a
versioned 774-value query/document observation. The [policy](poison/policy.py)
selects a masked operation, position, and payload. The [PPO update](poison/ppo.py)
uses clipped policy ratios, generalized advantage estimation, value losses,
and entropy. The [trainer](poison/train.py) includes edit validity, relevance
and detector-risk shaping, then a **terminal reward from actual retrieval and
the local RAG answer**, with a frozen defender within each run. Saved
checkpoints bind the data and defender hashes. Fixed substitution and random
edits run through the same environment and wrong-answer schedule.

Two new, 200-episode PPO runs were completed on the 350 MS MARCO training
questions, each evaluated on the separate 75-question validation set:

| Frozen attacker and validation replay | Defended attack successes | Same-schedule fixed / random |
| --- | ---: | ---: |
| PPO trained against V22 near-copy, tested on V22 | **67/75** | 5/75 / 25/75 |
| Same frozen PPO text replayed against V23 containment | **0/75** | Same attacker text |
| PPO retrained against V23 containment, tested on V23 | **55/75** | 5/75 / 20/75 |
| Same frozen PPO text replayed with V24 known-ID hash check | **0/75** | Same attacker text |

The first PPO learned to insert false answers before the original text; V23
searched the entire candidate and found the embedded original. The second PPO
inserted in the middle and often defeated text similarity. V24's exact
known-ID hash check catches those replacements because the attacker changed
the bytes of a document whose pre-attack ID and hash were pinned separately.
The [paired RL report](results/adaptive_ppo_v24.json) binds checkpoints,
baseline reports, case traces, and defender replays. This is a concrete
adversarial training and defense hardening loop. It is **one seed per new
round**, not a claim of statistically established PPO superiority across
seeds or a defense against newly inserted documents.

The clean validation replay with V24's ID/hash rule quarantined **0/375**
retrieved untouched passages and gave the **same 75/75 answers** as the
undefended local extractor. See [clean answer replay](results/clean_snapshot_v24_identity.json).
The local extractor and these 75 questions do not establish hosted LLM
answer quality or behavior on approved source updates.

## Reproduce the product

Builders for frozen data and model outputs refuse to overwrite existing
artifacts. Run them in a fresh checkout with the pinned source archives and
dependencies available. The main path is:

```powershell
python -m data.build_multifamily_v20 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m detect.train_multifamily_v20
python -m detect.train_fluency_v21
python -m evaluation.evaluate_fusion_v21
python -m evaluation.evaluate_snapshot_containment_v23
python -m evaluation.package_product_v22
python -m evaluation.package_product_v24
python -m evaluation.evaluate_product_v24
python -m scripts.scan_poison_v24 --input data/multifamily_v20/evaluation.jsonl --limit 3
```

`data.build_trusted_snapshot_v12` constructs the large pinned source index;
the V9 and V3 models are pre-existing inputs. PPO reproduction uses the
`--defender-aware --replace-existing --snapshot-detector` options in
`poison.train`, with `--snapshot-containment` for round two. The exact
checkpoints, seeds, source snapshot hashes, and replay commands are reflected
in [the adaptive report](results/adaptive_ppo_v24.json) and the CLI help.

The remaining security test is a **new, untouched** attack and clean-source
collection with reviewed factual labels and paired answer-level outcomes.
AdvDecoding recall is only 43/145 here. A new approved document may differ
from a pinned hash for legitimate reasons, and a new attacker document has no
known ID to compare. Keep those cases in a review workflow; do not call V24
production-ready from this development result.

The [deployment guide](DEPLOY_V24.md) records the hosted API, pinned snapshot
rebuild, deployment workflow, and the distinction between published local
evaluation and hosted smoke checks.

The later cross-domain SciFact transfer, three-seed PPO replication, initial
new-document failure, and revised answer replay are documented in
[V24 research release](V24_RESEARCH_RELEASE.md). Those results narrow the
claims above: scanner performance on new inserted documents is substantially
weaker than its inspected PoisonArena development result.
