# V14 offline RL defender decision pilot

Frozen before feature extraction, policy training, or evaluation. This is a
development experiment. PoisonArena's 100 NQ questions and all aggregate
results through v13 were inspected before this design. No success here is a
fresh blind confirmation.

## Action and reward

The defender receives a retrieved question/passage pair and chooses `allow`
or `quarantine` once. This is a one-step contextual-bandit RL problem, not a
multi-step environment. Train a small actor-critic policy over **frozen v3
detector signals**: its eight recorded features and risk score. The policy
never receives a family name, provenance label, question ID, reference qrel,
correct answer, target answer, or attack score. No SLM weights are updated.

Reward for a published attack document: +1 if quarantined, -1 if allowed.
Reward for a clean BEIR retrieval: 0 if allowed, -10 if quarantined. These
are source-based labels, not human annotations of factual contradiction.
For each sampled action, update the actor with clipped PPO on the observed
one-step reward and a learned value baseline. Warm-start the same network
with weighted supervised BCE (clean weight 10, attack weight 1). Report the
warm-start alone, every RL seed, and the frozen v3 detector on identical
examples and calibration. If RL does not beat the warm-start, say so.

Use the nine-dimensional signal vector (eight v3 features and v3 risk),
signed `log1p` transform, then a scaler fitted on the 50-question training
split only, with values clipped to [-8, 8]. Actor and critic each use a
9→32→1 ReLU network. Train the supervised warm-start for 80 epochs using
AdamW at 0.001, batch 128. For each RL seed, clone that actor and train a
new critic with 60 PPO rounds of 1,024 on-policy actions sampled with
replacement from training rows. Use four epochs per round, batch 256,
AdamW at 0.0003, policy-ratio clipping 0.2, entropy coefficient 0.003,
and value loss coefficient 0.5. No hyperparameter or seed selection on the
development questions.

## Splits and comparisons

Use PoisonArena's official clean top-ten retrievals and its AdvDecoding,
content-poison, corpus-poison, and GARAG published attack passages. GasLite
and PoisonedRAG black-box are **entirely withheld** from policy training and
threshold selection. Of the 71 v10 training questions, assign the first 50
under SHA256(`v14-cal:` + normalized question text) to actor training and
the remaining 21 to calibration. Keep the 29 v10 question-disjoint questions
for development evaluation. No passage or question from the 29 enters
training or calibration.
Use every available published passage. The Content Poisoning file covers
only 20 of the 100 questions; do not synthesize the missing passages.

For v3, supervised warm-start, and each RL seed, set the cutoff using the
21-question clean calibration subset at no more than 1% clean flags. Freeze
the cutoffs before the 29-question evaluation. Select no RL seed using those
development scores; report all three seeds (11, 22, 33). Also score the
original PoisonedRAG authors' HotpotQA targeted attack file, SHA256
`5119d6f9fd53cb0ecb3f33de939bd940d77cbaf80284b7ccd3d256f63a430537`,
as an attack-only, cross-dataset stress check. It has no matched clean
retrievals in this protocol, so the PoisonArena clean false-positive rate
cannot be transferred to HotpotQA.

Report per-family catch, clean and reference false flags, question-level
catch, and HotpotQA attack-only catch. RL is useful only if **at least two of
three RL seeds** catch more PoisonedRAG black-box passages than both the
matched-FPR supervised warm-start and matched-FPR frozen v3, while keeping
development clean flags at most 1% and not losing more than ten GasLite
catches relative to the best baseline. This is a demanding development gate,
not a live deployment standard. A gain solely in simulated reward or trained
families does not count.

Because both action outcomes can be computed from the released labels, a
supervised learner can use the same information more directly. The PPO run
is a falsifiable test of whether policy-gradient optimization helps here,
not an assertion that RL itself provides missing ground truth.
