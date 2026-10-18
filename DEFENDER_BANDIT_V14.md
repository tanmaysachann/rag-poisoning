# V14: offline RL quarantine decision pilot

**Decision: failed the predeclared gate; do not deploy.** The requested RL
experiment was implemented and run. It is a one-step PPO policy over nine
frozen v3 detector signals, not an SLM fine-tune. The same network was first
trained with source labels as a supervised baseline. All policies chose an
allow/quarantine action for each retrieved question/passage pair.

The protocol was written in
[`research_protocols/defender_bandit_v14.md`](research_protocols/defender_bandit_v14.md)
before feature extraction and training. Exact aggregate results and artifact
hashes are in [`results/defender_bandit_v14.json`](results/defender_bandit_v14.json).

## Data and gate

The [PoisonArena](https://github.com/yxf203/PoisonArena) benchmark supplied
100 questions with ten clean retrieved passages each, four attack families
available to training, and two families entirely withheld. Content Poisoning
has passages for only 20 questions; no examples were fabricated for the other
80. A fixed question-level split used 50 questions for training, 21 for clean
threshold calibration, and 29 for development evaluation. The policy was
trained on 1,295 rows, including 500 clean passages, without a family label,
provenance label, qrel, answer, or target in its input. Every cutoff allowed
at most 2/210 clean calibration flags (1%).

An additional 500 published HotpotQA attack texts came from the
[original PoisonedRAG authors' attack file](https://github.com/sleeepeer/PoisonedRAG/blob/main/results/adv_targeted_results/hotpotqa.json).
Those rows are attack-only; they cannot establish a HotpotQA false-positive
rate. The corresponding questions had zero text overlap with PoisonArena's
100 questions.

The gate required at least two of three PPO seeds to beat both the supervised
warm-start and frozen v3 on withheld PoisonedRAG, while flagging no more than
2/290 clean development passages and losing at most ten GasLite catches
relative to the better baseline.

## Measured results

| Policy | Clean false flags / 290 | PoisonedRAG black-box / 145 | GasLite / 145 | HotpotQA attacks / 500 |
|---|---:|---:|---:|---:|
| Frozen v3, recalibrated | 2 | 13 | 9 | 0 |
| Supervised warm-start | 23 | 144 | 0 | 352 |
| PPO seed 11 | 23 | 144 | 0 | 350 |
| PPO seed 22 | 26 | 145 | 0 | 365 |
| PPO seed 33 | 27 | 145 | 0 | 387 |

All three PPO seeds failed. The increase from 144 to 145 PoisonedRAG catches
in two seeds came with *more* clean false flags than the supervised model.
All learned policies missed every held-out GasLite passage. The 1% clean
calibration condition transferred poorly to the 29-question development
split: learned policies flagged 7.9–9.3% of clean passages. Improved sampled
training reward did not translate into an acceptable quarantine policy.

The complete per-family and question-level counts are in the JSON result.
The development set was inspected in earlier v9–v13 research, so these are
development observations, not fresh blind test evidence. Released attack
provenance is a useful source label but does not establish that every attack
passage is factually contradictory. This document does not measure whether
the downstream answer was compromised or repaired. These weights remain
offline under ignored `artifacts/defender_bandit_v14/`, together with the
saved feature-scaling parameters. They are not wired into the serving path.

## Reproduce

The source archives and frozen v3 model are required locally. The feature
builder checks source hashes and refuses to overwrite existing rows.

```powershell
python -m data.build_defender_bandit_v14 --beir-archive "$env:TEMP\sentinel-rag-beir-nq.zip"
python -m evaluation.train_defender_bandit_v14
```

Feature extraction took 10.84 seconds and training/evaluation 9.57 seconds
on this machine. The stored feature matrix has SHA-256
`32ae0a4a11a6fed1266a221cf6bb265a69e126a7f6e1a7aa358b4117d8ce4961`.
The method is a contextual-bandit form of reinforcement learning: the actor
samples an action, receives the specified reward, and updates with clipped
PPO and a learned value baseline. Because both rewards can be derived from
the released labels, this experiment gave RL no additional information beyond
the supervised baseline.
