# V6 factual verifier: measured result

The first reference-aware pilot **failed its quarantine gate**. It remains an
offline experiment. The live site and the v5 local combined gate were not
changed.

## What was done

- Downloaded the official [VitaminC](https://github.com/TalSchuster/VitaminC)
  archive and froze 30,000 real-revision claim/evidence pairs for training,
  5,500 for epoch selection, 7,000 for threshold calibration, and 7,000 from
  unseen source pages for a one-time test.
- Fine-tuned the local MiniLM encoder for SUPPORTS, REFUTES, and NOT ENOUGH
  INFO. It ran for two epochs on the laptop RTX 3050. The test was not opened
  during training or threshold selection.
- Calibrated a refutation threshold to reject at most 1% of supported and
  unsupported examples separately in calibration. A high-confidence
  contradiction is only a **review candidate**; it is not proof of malicious
  poisoning.

## Sealed test

| Gold relation to supplied evidence | Quarantined | Rate |
| --- | ---: | ---: |
| REFUTES | 50 / 1,000 | 5.0% catch rate |
| SUPPORTS | 3 / 5,000 | 0.06% false quarantine |
| NOT ENOUGH INFO | 10 / 1,000 | 1.0% false quarantine |

The model's ordinary three-class accuracy is a different operating point:
it chose REFUTES for 622/1,000 refuted examples, but also for 1,472/5,000
supported examples and 204/1,000 unsupported examples. Tightening the
threshold to protect clean documents removes most of the catch rate. The
predeclared 50% catch-rate gate was missed by a wide margin. Full figures and
Wilson intervals are in `results/factual_verifier_v6_test.json`.

As a development-only diagnostic, the
[VitaminC authors' pretrained ALBERT](https://huggingface.co/tals/albert-base-vitaminc)
was scored using its required evidence-then-claim input order. Its threshold
was set on the validation partition and checked on the separate calibration
partition: it caught 380/1,000 refuted cases (38%) while falsely quarantining
17/5,000 supported (0.34%) and 10/1,000 unsupported (1%). This is stronger
than the MiniLM pilot on development data, but still below the 50% gate, and
is not a sealed test or an independent RAG-poison result.

## Decision

Do **not** wire this model into document quarantine or claim that it detects
cunning RAG poisons. A 30,000-example model and a published model trained on
far more data both miss too many contradictions under a 1% false-quarantine
budget. Scaling the same labels alone has no demonstrated path to the target.

The next experiment must use complete retrieved documents, independently
authenticated reference passages, and attack outcomes. The
[external RAG-poisoning archive](https://zenodo.org/records/19582218) is
downloaded locally for source audit; it has independently generated poisoned
Natural Questions documents and recorded RAG runs, but its correct-answer
field is **not** a runtime trusted passage. We need the actual source evidence
before using it to evaluate a reference-aware gate. Keep this archive entirely
out of training and calibrate on genuine clean retrieval output.
