# Presentation literature audit

Checked against the publisher or arXiv paper pages on 2026-09-25. This is a correction map for slides 9–14 and 35–37 of `Presentation.pdf`; the PDF itself has not been edited. The slide numbers below use the PDF's printed numbering. Paper titles are given as the source lists them. A paper's abstract supports its own contribution, but the slides' “critical insight” and “limitations” columns are often our interpretations and should be labeled as such in the final report.

## Priority corrections

1. Slides 36–37 contain a chain of **wrong arXiv IDs**. RobustRAG is [2405.15556](https://arxiv.org/abs/2405.15556), visual document poisoning is [2504.02132](https://arxiv.org/abs/2504.02132), Poison-RAG is [2501.11759](https://arxiv.org/abs/2501.11759), and Poisoned-MRAG is [2503.06254](https://arxiv.org/abs/2503.06254). The cited [2511.01268](https://arxiv.org/abs/2511.01268) is a different paper, *Rescuing the Unpoisoned*.
2. The Information Fusion DOI in [15] belongs to Zhao et al., [*Exploring knowledge poisoning attacks to retrieval-augmented generation*](https://www.sciencedirect.com/science/article/abs/pii/S1566253525009625), article 103900. It studies **knowledge graph RAG and perturbation triples**. References [16] and [23] incorrectly attach Zhao's name to RobustRAG's arXiv ID. Replace [15] with the real citation and remove [16] and [23] as duplicates.
3. Reference [24] duplicates [12]. Keep one corrected FilterRAG citation.
4. Slide 14 rows 22–24 have no bibliography entries. Add the [Xu et al. 2026 security taxonomy](https://arxiv.org/abs/2604.08304), [Xi et al. RIPRAG](https://arxiv.org/abs/2510.10008), and [Wang et al. Astute RAG](https://arxiv.org/abs/2410.07176). Astute RAG was first submitted in **2024** and published at ACL in 2025; label the venue/year explicitly.
5. Slide 9's `>90%` PoisonedRAG claim and slide 12's `up to 98%` Poisoned-MRAG claim need the paper's exact experimental setting next to the number. The latter is specifically five injected image–text pairs on InfoSeek, per the [paper abstract](https://arxiv.org/abs/2503.06254). Avoid presenting either as a general success guarantee.

## Reference-by-reference map

| Printed reference | Audit result and canonical source |
|---|---|
| [1] | Correct identity: Tan et al., [*RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis*](https://arxiv.org/abs/2411.18948) (2024). |
| [2] | Correct identity: Zou et al., [*PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models*](https://arxiv.org/abs/2402.07867) (2024). |
| [3] | Correct paper: Zhang et al., [*Practical Poisoning Attacks against Retrieval-Augmented Generation*](https://arxiv.org/abs/2504.03957) (2025). “CorruptRAG” is the method name, not part of the source title. |
| [4] | Correct identity: Baolei Zhang et al., [*Benchmarking Poisoning Attacks against Retrieval-Augmented Generation*](https://arxiv.org/abs/2505.18543) (2025). Its abstract specifies 13 attacks and 7 defenses. |
| [5] | Correct paper: Xue et al., [*BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models*](https://arxiv.org/abs/2406.00083) (2024). The source title has “Retrieval Augmented” without a hyphen. Its abstract describes poisoned passages and a retrieval backdoor, not a change to retriever weights. |
| [6] | Correct identity: Clop and Teglia, [*Backdoored Retrievers for Prompt Injection Attacks on Retrieval Augmented Generation of Large Language Models*](https://arxiv.org/abs/2410.14479) (2024). Use the source's exact title. |
| [7] | Correct identity: Cheng et al., [*Secure Retrieval-Augmented Generation against Poisoning Attacks*](https://arxiv.org/abs/2510.25025) (2025); RAGuard is the method name. |
| [8] | Correct identity: Su et al., [*Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks*](https://arxiv.org/abs/2412.16708) (2024). |
| [9] | Correct identity: Zhou et al., [*TrustRAG: Enhancing Robustness and Trustworthiness in Retrieval-Augmented Generation*](https://arxiv.org/abs/2501.00879) (2025). |
| [10] | Correct identity: Kim et al., [*Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection*](https://arxiv.org/abs/2507.18202) (2025). |
| [11] | Correct paper: Zhang et al., [*Traceback of Poisoning Attacks to Retrieval-Augmented Generation*](https://arxiv.org/abs/2504.21668) (2025); RAGForensics is the method name. |
| [12] | Correct paper, wrong printed title: Edemacu et al., [*Defending Against Knowledge Poisoning Attacks During Retrieval-Augmented Generation*](https://arxiv.org/abs/2508.02835) (2025). FilterRAG is the method name. |
| [13] | Correct paper, wrong printed title: Hong et al., [*Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against Counterfactual Noise*](https://arxiv.org/abs/2305.01579) (2023). |
| [14] | Correct identity: Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) (2020). |
| [15] | Replace placeholder “Fusion-based AI Security Study” with Tianzhe Zhao et al., [*Exploring knowledge poisoning attacks to retrieval-augmented generation*](https://www.sciencedirect.com/science/article/abs/pii/S1566253525009625), *Information Fusion*, article 103900, DOI `10.1016/j.inffus.2025.103900`. Cite as 2026 journal article; the DOI string's `2025` is not a publication-year field. Scope: KG-RAG. |
| [16] | **Wrong attribution and duplicate.** Printed `2405.15556` is Xiang et al.'s [*Certifiably Robust RAG against Retrieval Corruption*](https://arxiv.org/abs/2405.15556), not a Zhao paper. Remove [16] and cite corrected [15] for Zhao. |
| [17] | Correct paper, paraphrased printed title: Li et al., [*Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models*](https://arxiv.org/abs/2406.05948) (2024). This is an LLM backdoor paper, not specifically a RAG corpus-poisoning defense. |
| [18] | Correct paper, paraphrased printed title: Shafran et al., [*Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents*](https://arxiv.org/abs/2406.05870) (2024). |
| [19] | **Wrong ID.** Nazary et al.'s [*Poison-RAG: Adversarial Data Poisoning Attacks on Retrieval-Augmented Generation in Recommender Systems*](https://arxiv.org/abs/2501.11759) is `2501.11759`, which is printed under [20]. |
| [20] | **Wrong ID.** Shereen et al.'s [*One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image*](https://arxiv.org/abs/2504.02132) is `2504.02132`, which is printed under [21]. |
| [21] | **Wrong ID.** Xiang et al.'s [*Certifiably Robust RAG against Retrieval Corruption*](https://arxiv.org/abs/2405.15556) is `2405.15556`, which is printed under [16]/[23]. |
| [22] | **Wrong ID.** Liu et al.'s [*Poisoned-MRAG: Knowledge Poisoning Attacks to Multimodal Retrieval Augmented Generation*](https://arxiv.org/abs/2503.06254) is `2503.06254`, which is printed under [19]. Printed `2511.01268` belongs to Kim et al.'s [*Rescuing the Unpoisoned*](https://arxiv.org/abs/2511.01268). |
| [23] | Duplicate of the incorrect [16]; remove it. |
| [24] | Duplicate of [12]; remove it. Its title is closer to the source than [12]'s, but retain one entry. |

## Missing slide 14 sources

| Survey row | Recommended entry and scope |
|---|---|
| 22 | Yuming Xu et al., [*Securing Retrieval-Augmented Generation: A Taxonomy of Attacks, Defenses, and Future Directions*](https://arxiv.org/abs/2604.08304) (2026). The abstract discusses a six-stage external knowledge-access pipeline and the SLOT taxonomy. The slide's “operational boundary” wording matches this paper's earlier description. Call it a security taxonomy; do not cite the unrelated [agentic RAG SoK](https://arxiv.org/abs/2603.07379). |
| 23 | Meng Xi et al., [*RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning*](https://arxiv.org/abs/2510.10008) (2025 first submission; revised 2026). The abstract says it trains a document-generation model from black-box feedback, which differs from this project's discrete edit-action PPO. |
| 24 | Fei Wang et al., [*Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models*](https://arxiv.org/abs/2410.07176) (2024 preprint; [ACL 2025 publication](https://research.google/pubs/astute-rag-overcoming-imperfect-retrieval-augmentation-and-knowledge-conflicts-for-large-language-models/)). It iteratively consolidates LLM internal and retrieved knowledge with source awareness. |

## Presentation wording to tighten

- Slide 9 row 1: “internal reasoning behavior is more reliable” is broader than a demonstrated detector comparison. State that RevPRAG uses final-token activations and report its measured conditions separately.
- Slide 11 row 9: Lewis et al. introduced RAG for knowledge-intensive tasks; the security implication about trusting retrieved text is this project's inference, not that paper's finding.
- Slide 11 row 12: BadRAG's abstract says poisoned database passages create trigger-conditioned retrieval behavior. Keep it separate from [6]'s backdoored-retriever setting.
- Slide 12 row 16: retain the “up to 98%” number only with the five-pair InfoSeek setting and the models evaluated by the paper.
- Slide 13 row 17: specify that Zhao et al. study **knowledge graph** poisoning, not all RAG architectures.
- Slide 14 row 22: source the 2026 taxonomy above and avoid presenting its broad survey as an evaluated defense.
- Slide 14 row 23: distinguish RIPRAG's generator training from our discrete-action PPO, and avoid suggesting the same reward/feedback contract without an explicit comparison.
- All survey rows: separate the paper authors' reported result from our inferred design lesson and limitations. Verify precise numerical claims against experimental tables before the final slide export.
