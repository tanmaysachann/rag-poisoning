# V16 independent snapshot recovery: development protocol

Written after the v15 validation replay and before implementing v16 recovery.
V15's source guard stopped 19/19 v3 stealth attack-sourced answers under
accepted ingest, but abstained on 148/150 stealth cases. V16 addresses that
utility failure without changing the pre-attack trust boundary. All v3
questions and v15 results have been inspected; v16 is development only.

The guard first checks the v3-selected answer's exact source ID, document
hash, and cited span against the pinned pre-attack corpus. If it passes, keep
the answer. If it fails or v3 abstains, independently retrieve the top five
from the pinned corpus using only the question, extract a cited sentence
with the existing hashing answerer, and verify that recovered source and span
against the pinned manifest. Emit that recovered answer if valid; otherwise
abstain. No support ID, answer key, target phrase, attack label, or attack
family may influence the recovery. The original pinned corpus SHA is checked
against the versioned benchmark manifest before every workbench run. A
verified source is an identity guarantee, not an external truth guarantee.

Replay exactly the same 150 validation questions, saved greedy and stealth
payloads, one passage each, and two trust surfaces as v15. Report v3 and v16
attack-sourced answer success, target-string matches regardless of source,
clean answer alias recovery, abstentions, recovered answers, and emitted
answers with unverified sources. Report clean-corpus performance separately.
V16 is useful under this bounded threat model only if it keeps attack-sourced
answers at zero, recovers more clean aliases than v15 on accepted-ingest
stealth, emits no unverified source, and does not reduce clean-corpus alias
matches relative to the v3 path. The full replay is still not a fresh blind
test. No result here proves detection of malicious intent or generalizes to
an attacker who controls the pinned source.
