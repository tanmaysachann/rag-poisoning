# Poisoned-document detection and RL research plan

Prepared 2026-10-19. [PRODUCT_V24.md](PRODUCT_V24.md) records the concrete scanner, trained data, adaptive PPO work, and measured results now implemented. The targets below are remaining research gates, not claims that they have been achieved.

## 1. Reproduce the actual failure before changing a model

The reported "30.3%" has no clearly matching saved aggregate in the current repository. Record the exact command, checkpoint, split, denominator, class balance, confusion matrix, and whether the number is *binary accuracy*, *attack recall*, or *clean-answer alias match*. A 30% attack recall is not comparable to a 50% coin flip. In an imbalanced candidate stream, binary accuracy can also be misleading. Report attack recall at a fixed clean false-positive rate (FPR), clean FPR, precision, per-family recall, and downstream answer outcomes separately.

The existing evidence already shows the gap:

| Experiment | Measured observation | Interpretation |
| --- | --- | --- |
| V3 hashing detector | Very high recall on locally generated overt attacks, poor transfer to subtle substitutions and external attacks | The training distribution is too narrow. |
| V10 reference-conditioned model | On 29 previously inspected PoisonArena questions: 108/145 GasLite passages caught, but 2/145 PoisonedRAG black-box passages caught, at 2/290 clean flags | Multi-family training helped one unseen family, not the other. |
| V6 VitaminC verifier | 50/1,000 refutations caught at the strict threshold on sealed test | Generic NLI training does not solve the operating-point problem. |
| V14 defender PPO | 144-145/145 PoisonedRAG catches, 0/145 GasLite catches, 23-27/290 clean flags | Better sampled reward did not produce a usable detector. |
| Attacker PPO | 31/225 defended successes versus 34/225 fixed substitution, across three development seeds | The present RL attacker does not beat its strongest simple baseline. |
| V16 snapshot recovery | 0/150 attack-sourced target answers in one inspected stealth replay, but 28/150 clean alias matches and 80/150 abstentions | Source control helps under its bounded threat model; answer utility remains weak. |

These old validation and PoisonArena questions have been repeatedly inspected. Use them for diagnosis and development. Freeze a **new question- and source-disjoint confirmation set** before model selection.

## 2. Build data that matches the real decision

### Data sources and roles

1. **Clean retrieval and questions:** use the existing pinned NQ and MS MARCO assets, then add a smaller manually reviewed, versioned collection of authoritative source passages with question-answer evidence. Keep the clean corpus snapshot, source URL/identity, revision date, document hash, license, and reviewer note. A hash proves identity, not factual truth.
2. **Attack diversity:** use PoisonArena and PoisonedRAG as development sources; include answer substitutions, fabricated factual claims, instruction injection, URL insertion, refusal/jamming, authority spoofing, and benign-looking multi-sentence changes. Use BIPIA only for the instruction-injection branch because its labels are not factual-poison labels. Treat RAGuard as a separate misleading-evidence stress set, not as a drop-in poison training label.
3. **Semantic warm start:** VitaminC and FEVER can pretrain claim/evidence relations (`supports`, `refutes`, `insufficient`). Their labels must be mapped into a distinct task head; a refuted claim is not automatically an attacker-controlled document.
4. **Hard clean negatives:** highly relevant clean passages, legitimate updates, quoted malicious text used for analysis, conflicting historical facts, near-duplicate sources, and clean answer-bearing passages that strongly change the answer when removed. This is essential to prevent the model from learning "influential = malicious" or "strange style = malicious."
5. **Human review:** review a stratified sample of subtle substitutions and clean conflicts with two reviewers. Label (a) attacker control/source provenance, (b) factual relation to reviewed evidence, (c) instruction attempt, (d) whether the passage was retrieved, and (e) whether it changed the downstream answer. Resolve disagreements and retain evidence spans.

Start with a feasible target of 2,000-3,000 question episodes, roughly 10,000-20,000 clean candidate passages and 10,000-20,000 attack candidates, plus 800-1,200 reviewed difficult pairs. Counts are planning targets, not data already present. Reserve at least 1,000 genuine clean candidate passages and at least 200 attack candidates per unseen family for final measurement; enlarge if confidence intervals are too wide. Prevent near-duplicate passages, source pages, question paraphrases, and generated variants from crossing splits. Split by question **and** source document; hold out whole attack generators/templates and a separate source/domain slice. Record source archive, license, revision, SHA-256, generator model, prompt, seed, and attack goal for each row.

**Row schema:** `question`, `candidate_text`, `candidate_source`, `source_hash`, `trusted_reference_text` (if independently selected), `reference_hash`, `attack_family`, `attacker_controlled`, `claim_relation`, `instruction_attempt`, `retrieved_rank`, `answer_changed`, `reviewed_evidence_spans`, `split`, `group_id`. Attack family, gold answer, target answer, label, and benchmark qrel must never enter inference features. If no reviewed reference exists, return `unassessable` for factual contradiction rather than inventing one from the answer key.

## 3. Train the detection system with the right objective

Use a **multi-stage, multi-task** design with measured incremental value:

1. **Cheap screen:** current integrity/provenance checks, retrieval anomaly, instruction and URL patterns, existing statistical signals. Hash mismatch is a hard integrity rule only for a pinned snapshot. New approved documents need their own review workflow.
2. **Evidence retrieval:** question-only retrieval from an independently maintained, versioned trusted collection. Measure reference recall/coverage without using support IDs or answer keys. If a trustworthy reference is unavailable, record this as a separate outcome.
3. **Claim/evidence model:** extract answer-bearing claims and evidence spans; fine-tune a cross-encoder with query, candidate claim, reviewed reference, and source context. Predict `supports`, `refutes`, `insufficient`, and `instruction_attempt` with separate heads. Train with hard-negative mining and balanced batches across source, family, and document length. Test whether reference swapping changes semantic decisions, while recognizing that swapping is only a diagnostic.
4. **Fusion and selective checks:** combine calibrated semantic outputs with cheap signals and source confidence. Run SLM/SRQ, leave-one-out generation, or a second verifier only on uncertain or high-impact cases. LOO influence alone must not imply poisoning. Keep the source evidence and reason codes in the output.
5. **Decision calibration:** choose a quarantine threshold using genuine clean retrievals on calibration questions, targeting no more than 1-2% clean candidate FPR. Calibrate per mode when a trusted reference is available versus unavailable. Compare logistic fusion, a small gradient-boosted baseline, and a calibrated neural fusion head; adopt complexity only when it improves fresh held-out results.

Use weighted cross-entropy/focal loss only as training choices; optimize the selected operating point with a clean-FPR constraint. Evaluate both passage classification and what the answerer actually emits. A detector that blocks attacks by abstaining on almost every clean query is not an acceptable improvement.

## 4. Implement the presentation's RL attacker as a meaningful MDP

Keep `poison/env.py`, `poison/policy.py`, `poison/ppo.py`, and `poison/train.py` as the starting point. Expand their state schema to include retrieval rank/margin, query/document embeddings, semantic detector scores, claim-relation scores, edit history, and remaining query budget. Version the exact feature dimensions instead of forcing the slide's 1,556-D number. The current 774-D state is an implemented design choice, not a missing mathematical requirement.

Use a factored, masked action `(edit operation, span/position, payload candidate)`. Add answer-span replacement, evidence laundering, benign paraphrase, and limited multi-document edits to the current operations. A small language model may propose fluent text candidates, while PPO selects among bounded candidates; validate meaning, length, edit budget, and source control after every edit. Track whether the poison was actually retrieved and whether the frozen defender accepted it.

Train with terminal reward from **actual downstream attack success** on the frozen retrieval, detector, and answer path. Add smaller shaping terms for retrieval, stealth, edit cost, and diversity. Log each component and detect reward hacking. Freeze the defender within each attacker run. Start from behavior cloning of successful edits if useful, then PPO with correct action masks, conditional log probabilities, GAE, terminal/truncation handling, and checkpoint hashes. Compare at least three seeds against fixed substitution, greedy search, random edits, and the existing policy on the same queries and budgets. State the attacker's access: white-box scores during training; black-box query-limited transfer in a separate evaluation. Keep a held-out defender variant for transfer.

Run two or three **adversarial curriculum rounds**: train the attacker against a frozen defender, collect novel successful and failed edits, review a stratified subset, add only training-question examples to the detector's next training pool, then freeze the revised defender before the next attacker run. Track whether each round improves the *unchanged* development families and whether attack diversity collapses. No generated variant of a final-test question or source may enter this loop.

The slide's edit-effect cache currently rewards repeated nearby actions, so call it a **reuse cache** unless a measured novelty term is added. Its existing ablation showed no outcome gain. Keep it only if it improves a predeclared metric.

## 5. Add defender RL only where it has a genuine sequential decision

The V14 allow/quarantine bandit used source labels to compute both action rewards; a supervised classifier had the same information and beat it on the relevant utility gate. Keep V14 as a negative result. A more defensible RL task is **adaptive verification under a latency budget**:

- **State:** cheap detector scores, reference coverage, disagreement, retrieval ranks, provenance status, probe results so far, remaining time/model-call budget. No gold answer or source label in observations.
- **Actions:** allow, quarantine, abstain, retrieve independent reference, run claim verifier, run SLM/SRQ, run LOO answer ablation. Each information-gathering action changes the next state and has a measured cost.
- **Episode/reward:** finish after a decision or budget exhaustion. Reward correct and supported clean answers; penalize attack-sourced answers strongly, false quarantine/needless abstention, and measured latency/model calls. Use the training labels and answer outcomes only to compute reward, never as features.
- **Training:** warm-start from a transparent rule policy; compare contextual bandit, supervised selective classifier, and budgeted decision tree against masked PPO. Train against multiple frozen attacker pools and rotate them on a schedule. Fix the final detector and thresholds before the untouched confirmation run. Report 3-5 seeds and the full security/utility/cost Pareto frontier.

This policy should be promoted only if it beats the calibrated supervised selective system under the **same clean-FPR and latency limits**. RL is not a substitute for evidence quality or threshold calibration.

## 6. Evaluation contract and pass gates

Freeze protocol, data manifest, model revisions, thresholds, seeds, attack budgets, answerer, retriever, and score scripts before final test. Run paired defended/undefended cases using identical retrieval candidates where the experiment requires it. Report counts and 95% query/source-cluster intervals, including every seed and family.

| Axis | Required outputs |
| --- | --- |
| Detection | Per-family recall at 1% and 2% clean FPR; clean FPR; precision/PR-AUC; confusion matrix; calibration; reference coverage; review-needed rate. |
| Security | Attack success before/after defense, conditional on retrieval; source of emitted answer; citation validity; effect of 1/3/5 attacker documents. |
| Utility | Clean answer alias and manually judged support, abstention, clean false quarantine, retrieval recall@k. |
| Generalization | New questions and source pages, unseen generators, subtle factual edits, benign updates, alternate retriever and answerer. |
| Cost | p50/p95 serving latency, model calls, GPU/CPU memory, training time, annotation effort. |

**Proposed adoption gates:** at most 2% clean candidate FPR on the fresh set; at least 50% catch for *each* unseen attack family; at least a 50% relative reduction in attack-sourced wrong answers versus the paired undefended baseline; clean answer quality no more than two percentage points below the strongest non-RL defense; and a measured latency budget fixed before test. Set a stretch goal of 1% FPR and stronger per-family recall. Treat these as goals, not expected results. Attacker PPO needs a positive paired advantage over fixed/greedy attacks with a query-cluster interval excluding zero; defender PPO needs a positive paired security/utility gain over the supervised selective baseline at matched FPR and cost. If either fails, report it as an ablation rather than claiming RL improved detection.

## 7. Remaining evidence gates and compute

The V22 scanner now supplies a six-family development result, a pinned source
integrity branch, and a saved PPO attacker stress test. Before promotion, it
still needs a newly collected, untouched source- and question-disjoint test;
reviewed hard clean negatives and factual relation labels; paired defended and
undefended answer replay; and cross-corpus adversarial training that improves
the MS MARCO PPO failure. Any new PPO policy should be compared with fixed
substitution under the same answer path and attack budget.

The current RTX 3050 Laptop GPU has 4 GB VRAM. Fine-tune compact cross-encoders locally with mixed precision and small batches; run document-edit PPO with the existing small local model or cached candidate generation. The slide's Llama-3-8B NF4 profile does not fit the measured local GPU; use a separately documented larger GPU only if actually available. Keep the CPU demo reproducible with pinned checkpoints and precomputed examples. Do not make a large model a requirement for the method to work.

### Further code work beyond V22

1. Add a source- and question-disjoint, reviewer-labeled confirmation set with
   factual-relation and downstream answer labels.
2. Add independently authenticated reference selection and a paired answer
   replay evaluator; retain the existing live detector until the gates pass.
3. Retrain and compare the attacker PPO against a frozen V22-compatible
   defender on the same corpus, with fixed substitution and random edits.
4. Build a sequential defender verification environment only if it beats a
   supervised policy at matched security, clean utility, and measured cost.

## Primary sources for the proposed data and method

- [PoisonArena official repository](https://github.com/yxf203/PoisonArena)
- [PoisonedRAG official repository](https://github.com/sleeepeer/PoisonedRAG)
- [VitaminC official repository](https://github.com/TalSchuster/VitaminC)
- [FEVER official repository](https://github.com/awslabs/fever)
- [BIPIA official repository](https://github.com/microsoft/BIPIA)
- [RAGuard misleading-retrieval dataset](https://huggingface.co/datasets/UCSC-IRKM/RAGuard)
- [Original PPO paper](https://arxiv.org/abs/1707.06347)
