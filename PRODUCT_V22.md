# Poisoned document scanner V22: runnable research product

This repository now has a local scanner, frozen model artifacts, source data
manifests, per-case results, and an RL attacker stress test. The scanner accepts
`query`, `reference`, and `candidate`, then returns a `review` or `pass` decision
with separate learned-model and source-integrity evidence. Run it from the repo
root:

```powershell
python -m scripts.scan_poison_v22 --input data/multifamily_v20/evaluation.jsonl --limit 3
```

For one document, put the reference and candidate in UTF-8 text files:

```powershell
python -m scripts.scan_poison_v22 --query "Who wrote the book?" --reference-file reference.txt --candidate-file candidate.txt
```

The reference is supplied by the caller. This research scanner does **not**
authenticate it or silently add itself to the live FastAPI answer path. The
integrity branch requires the separate pre-attack snapshot in
`data/trusted_snapshot_v12`. Use `--no-snapshot` to score only the learned
branch. JSONL input must contain `query`, `reference`, and `candidate` keys.

## The dataset answer

The V9 warm-start source had **16,114 labeled NQ candidate rows from
2,302 questions**: 2,302 benchmark references, 11,510 other clean passages,
and 2,302 generated poison passages. The MiniLM actually trained on **14,504**
of those rows from **2,072** questions and used **1,610** rows from **230**
questions for validation. V22 then fine-tuned its MiniLM scorer on
**1,855 PoisonArena/BEIR NQ rows from 51 training questions**: 545 clean or
reference rows and 1,310 published attack rows from six families. The char
trigram model used 13,812 V9 clean/reference documents. A separate gradient
boosting branch learned from V3 retrieval and behavior features plus char
fluency. The 20-question, 749-row calibration split set the operating point;
the 29-question, 1,066-row development split measured it. Questions do not
cross these three PoisonArena splits. The [manifest](data/multifamily_v20/manifest.json)
pins the rows, source checksums, family counts, and known overlap caveats.

The **2,681,468-passage BEIR NQ corpus** is a separate, hash-pinned source
snapshot used to find changed copies. It was **not** used as 2.68 million
supervised training examples. BEIR qrels select a benchmark reference; they do
not authenticate its factual truth. Attack labels denote published,
attacker-supplied passages, not an independent fact check or proof of a changed
answer. The PoisonArena questions had been inspected earlier in this project,
so the 29-question evaluation is development evidence, not a blind test.

## What the scanner actually computes

1. A reference-conditioned MiniLM scores the full candidate in token windows.
2. A gradient boosting model scores eight V3 statistical/behavior features,
   the V3 risk, and four clean-trained character fluency features.
3. Each model score is converted to an empirical upper-tail probability using
   the 214 clean calibration passages. The larger anomaly score is compared
   with the frozen 1% calibration threshold.
4. Independently, a read-only FTS index finds a near-copy in the separate,
   hash-pinned source snapshot. A changed copy with normalized similarity at
   least 0.90 raises an integrity alert. Exact snapshot matches do not raise
   this alert. Every retrieved source passage is hash-checked.
5. Either alert requests **review**. The output preserves both scores and the
   matched source ID. A `pass` is not a guarantee that the passage is true.

The immutable calibration bundle is
`artifacts/product_v22_calibration.joblib`. The scanner verifies SHA-256 hashes
for its neural weights, signal model, V3 feature extractor, data manifest,
fusion report, and snapshot manifest before scoring.

## Measured development result

At the clean-calibrated 1% learned threshold and 0.90 changed-copy threshold:

| Branch | Attack passages flagged | Clean passages flagged |
| --- | ---: | ---: |
| Learned fusion | 409/755 (54.2%) | 9/311 (2.9%) |
| Snapshot integrity | 99/755 (13.1%) | 0/311 |
| Combined review decision | **501/755 (66.4%)** | **9/311 (2.9%)** |

The combined result caught 43/145 AdvDecoding, 23/30 content poisoning,
95/145 corpus poisoning, 84/145 GARAG, 111/145 GasLite, and 145/145
PoisonedRAG black-box attack passages. Calibration flagged 2/214 clean rows;
the observed 29-question development clean false alarm rate was 9/311, above
the nominal 1% target. See [the aggregate report](results/product_v22.json),
[learned report](results/fusion_v21.json), and
[integrity report](results/snapshot_near_duplicate_v22.json). The reports have
case-file hashes so these counts can be audited. No downstream RAG answer
replay was performed for V22.

## The RL answer

The presentation's RL component is an **attacker** that edits a retrieved
document. It is implemented in `poison/env.py`, `poison/policy.py`,
`poison/ppo.py`, and `poison/train.py`. The observation is a versioned 774-value
query/document state; the policy chooses a masked, factored edit operation,
position, and payload. Training uses clipped PPO with generalized advantage
estimation, an edit/retrieval/stealth reward, and a terminal check on the
retrieved answer. The defender is frozen while each attacker policy trains.
There are three 200-episode defender-aware PPO training seeds on 350 MS MARCO
training questions and saved validation rollouts on 75 questions. The PPO
policy achieved **31/225 defended attack successes**, compared with **34/225**
for fixed answer substitution and 6/225 random edits. It did **not** beat the
strong fixed baseline. The [PPO comparison](results/ppo_multiseed_validation.json)
and `python -m evaluation.summarize_ppo_seeds` show the paired evidence.

V22 was also tested on the saved PPO final documents. On MS MARCO, the
NQ-trained learned branch flagged only **7/225** edited attempts, including
**1/34** raw successful attempts, while flagging **3/75** unmodified clean
support passages. This is a failed cross-corpus transfer test. Applying the
same 0.90 integrity rule to a separately pinned, unmodified 75-passage MS
MARCO validation snapshot flagged **90/225** edits and **12/34** raw successful
attempts, with **0/75** clean integrity alerts; combined alerts were 96/225
and 13/34. The PPO policy was trained against V3, not V22, and these are
offline flags on saved text, **not** measured prevented answers. See
[the PPO learned report](results/product_v22_ppo_stress.json) and
[PPO integrity report](results/product_v22_ppo_snapshot.json).

To scan against another pre-attack snapshot, the caller must pin its manifest:

```powershell
python -m scripts.scan_poison_v22 --input cases.jsonl --snapshot-dir data/benchmark_snapshot_v22 --snapshot-manifest-sha256 6ab99ce47d6702c035191e33f6ded5343be41205f107311099b008788c2f5173
```

## Rebuild and checks

The builders refuse to overwrite frozen outputs. In a fresh checkout with
the pinned source archives and dependencies installed, the main sequence is:

```powershell
python -m data.build_multifamily_v20 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m detect.train_multifamily_v20
python -m detect.train_fluency_v21
python -m evaluation.evaluate_fusion_v21
python -m evaluation.evaluate_snapshot_near_duplicate_v22
python -m evaluation.package_product_v22
python -m evaluation.evaluate_product_v22
python -m data.build_benchmark_snapshot_v22
python -m evaluation.evaluate_product_on_ppo_v22
python -m evaluation.evaluate_ppo_snapshot_v22
python -m unittest tests.test_snapshot_near_duplicate_v22 -v
```

The BEIR snapshot itself is built by `data.build_trusted_snapshot_v12` from
that pinned archive; V9, V3, and PPO artifacts are pre-existing prerequisites.
The read-only snapshot database is large (about 2.69 GB). The complete
PoisonArena source and archive hashes are in the manifests. The product remains
a local research scanner until an independently reviewed source collection,
a genuinely untouched attack collection, and paired answer-level replay show
both security benefit and acceptable clean answer utility.
