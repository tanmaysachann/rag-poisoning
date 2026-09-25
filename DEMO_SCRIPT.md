# Seven-minute Sentinel RAG demonstration

This script uses only the local, authored five-question demo. Keep the research
validation and sealed test results on the Research tab as separate evidence.

## Before the presentation

From the project root, run `python scripts/doctor.py`, then
`python -m evaluation.verify_test_v1`. Start the API with
`python backend/api.py` and open `http://127.0.0.1:8000`. Keep the local
Qwen option disabled; the live console uses the extractive answerer. The demo
index and classifier are already saved. If they are missing, follow the four
build commands in `README.md` before starting the server.

## 0:00–1:00 — Define the threat

Show the five-question console. Explain that retrieved documents are data,
even when they contain text that looks like an instruction or a credible
report. The attacker controls a document inside a closed research corpus.
No public site or third-party system is targeted.

## 1:00–2:30 — Compare defense off and on

Choose **Where is the Eiffel Tower located?** and run the pipeline. Point to
the poisoned document's rank and the two answers under Counterfactual
Comparison. The defense-off run admits every retrieved passage; the defense-on
run applies the same answer selection after filtering. Show the flagged
document's S1 geometry, S4 stability, behavior, and fusion risk bars. Repeat
with **At what temperature does water boil at sea level?** to show a second
answer type. If the answer does not change in a locally rebuilt corpus, use
the saved validation case explorer instead of claiming a live success.

## 2:30–3:30 — Explain integrity

Enable **Simulate a post-index modification** and rerun one scenario. Show
the mismatch status and quarantine decision. Explain that SHA-256 compares
retrieved text against a trusted snapshot. It detects a later edit; it cannot
prove a document was true when the snapshot was created. The reviewed-digest
workflow in `scripts/trusted_ingest.py` handles deliberate re-sealing.

## 3:30–5:00 — Show measured research outcomes

Open **Research**. The validation cards and case explorer use saved 75-query
development runs; they do not train or reindex when the page opens. Switch
between greedy insertion and stealth answer substitution. State the central
finding: the frozen detector catches overt insertion but misses most subtle
answer replacement. The SLM SRQ and Qwen decoder probes were genuinely run
on the local model, but their measured recall did not justify a live gate.

## 5:00–6:00 — Show the sealed test result

Point to the **Frozen test v1** card. On the 75-question MiniLM/ranker test,
greedy attack success was 37/75 before defense and 0/75 after. Stealth
success was 23/75 before and 22/75 after; only 3/75 stealth documents were
quarantined. Clean answer alias match was 39/75. The API verifies saved
hashes and per-case counts before serving these aggregate figures. The test
split has been used for v1 and is no longer an untouched test for changes.

## 6:00–7:00 — Explain the policy and limitations

Describe the five document-edit operations and the masked factored PPO
policy. The 200-episode undefended validation run tied a fixed insertion
baseline at 67/75 attack success. The defender-aware run tied fixed answer
substitution at 11/75 defended successes. These runs show that the attack
environment and policy are implemented; they do not establish an RL advantage.
Finish with the practical limitation: a trusted digest catches later changes,
while accepted false facts need stronger provenance or corroboration. The
local Qwen 0.5B generator was tested as a diagnostic, but strict cited output
failed in its ten-question sample, so the live demo keeps extractive answers.

For every number above, see `REPORT_EVIDENCE.md` and its linked raw result
files. `PRESENTATION_REFERENCES_READY.md` supplies corrected bibliography
entries for the slide deck.
