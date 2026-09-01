# Review-1 Implementation Results

## Implemented system

The Review-1 MVP implements a complete controlled secure-RAG path without RL training. The corpus contains 25 clean reference documents and five adversarial reports. Retrieval combines BM25 and dense similarity using Reciprocal Rank Fusion. Every retrieved document passes through SHA-256 integrity verification and a seven-signal fusion detector before accepted content is placed inside an isolated untrusted-context prompt.

## Poison retrieval results

| Attack document | Attack style | Target query | Top-5 rank |
|---|---|---|---:|
| 1001 | Misinformation | Where is the Eiffel Tower located? | 1 |
| 1002 | Instruction/URL injection | At what temperature does water boil at sea level? | 2 |
| 1003 | Misinformation | What is the capital of Australia? | 2 |
| 1004 | Instruction/URL injection | What is photosynthesis? | 1 |
| 1005 | Stealth misinformation edit | What is the largest ocean on Earth? | 2 |

All five controlled attacks satisfy the retrievability condition by appearing in the top two results for their target query.

## Detection result

The fusion detector was evaluated with leave-one-out cross-validation on 30 samples: 25 clean and five poisoned. The current controlled result is accuracy 1.00, precision 1.00, recall 1.00, F1 1.00, and ROC-AUC 1.00. These results show that the implemented signals separate the authored attack patterns in this small corpus; they are not evidence of production-scale generalization.

## Example defense outcomes

| Query | Defense OFF | Defense ON |
|---|---|---|
| Eiffel Tower location | Surfaces the injected London claim | Returns Paris, France from clean document 0 |
| Capital of Australia | Surfaces the injected Sydney claim | Returns Canberra from clean document 2 |
| Photosynthesis | Surfaces the injected URL/instruction | Returns the biological definition from clean document 3 |
| Largest ocean | Surfaces the injected smallest-ocean claim | Returns the Pacific Ocean from clean document 4 |

The dashboard exposes retrieval ranks, poison probabilities, individual signal strengths, quarantine reasons, selected evidence, per-stage latency, integrity hashes, and PDF evidence reports. A tamper-control option modifies one document after indexing and produces a visible SHA-256 mismatch and quarantine decision.

## Review boundary and Phase-2 direction

Objective 1 is represented by five manually authored adversarial documents using the report's `INSERT` and `SYNONYM_CAMOUFLAGE` operation vocabulary. PPO training is intentionally excluded from Review-1. Phase 2 will add the document-editing MDP, generate a larger and more diverse adversarial dataset, replace the deterministic SRQ approximation with an SLM-backed implementation, and repeat detection and ablation experiments at scale.
