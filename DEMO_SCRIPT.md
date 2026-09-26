# Seven-minute Sentinel RAG demonstration

The deployed home page is now the Research Lab. It runs the bounded hashing
validation experiment live; the MiniLM, PPO and Qwen measurements are saved
offline evidence. The authored five-question demo remains in its own tab.

## Before the presentation

From the project root, run `python scripts/doctor.py`, then
`python -m evaluation.verify_test_v1`. Start the API with
`python backend/api.py` and open `http://127.0.0.1:8000`. Keep the local
Qwen option disabled; the live console uses the extractive answerer. The demo
index and classifier are already saved. If they are missing, follow the four
build commands in `README.md` before starting the server.

## 0:00–1:00 — Define the threat

Show the Research Lab workbench. Explain that it stages one edited passage
against a fixed 75-question validation corpus and builds a temporary index.
The attacker controls a document inside a closed research corpus. No public
site or third-party system is targeted.

## 1:00–2:30 — Compare defense off and on

Use the default Princeton-tuition stealth case and run the live workbench.
Point to the edited answer span, retrieval rank, verified digest, eight detector
features, and unchanged defended wrong answer. Then switch to greedy insertion
to show the research detector's different behavior. The source data and
detector are fixed validation artifacts; the attack passage can be edited.

## 2:30–3:30 — Explain integrity

Change the workbench trust surface from **Accepted at ingest** to **Edited after
indexing**, then rerun the same passage. Show the SHA-256 mismatch and
quarantine. Explain that the digest checks a trusted snapshot but cannot
establish whether an accepted document is factually true.

## 3:30–5:00 — Show measured research outcomes

Move to the sealed-test evidence and PPO sections. State the central finding:
the frozen detector catches overt insertion but misses most accepted-ingest
answer substitutions. Show one saved PPO edit trace and the matched fixed
baseline. The SLM SRQ and Qwen decoder probes were run offline but did not
justify a live gate. The full saved-metrics archive is collapsible at the end.

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
