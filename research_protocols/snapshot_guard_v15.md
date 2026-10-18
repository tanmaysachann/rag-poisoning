# V15 pinned-snapshot answer guard: development protocol

Written before the v15 replay. The 150 v3 validation questions and prior
aggregate v3 results have already been inspected. This is development
evidence, not a new held-out result.

## Threat boundary and action

The attacker may replace one indexed passage or add up to three passages in
the live validation workbench. They cannot edit the pinned pre-attack clean
corpus, its versioned SHA-256 record, or the server code. An attacker who can
alter those trusted assets is outside this pilot's threat model. The source
hash proves that text matches a pinned snapshot, not that the text is true.

After normal retrieval, integrity checking, v3 detector filtering, extractive
answer selection, and the optional four-probe gate, a new answer guard checks
the *selected source document* against the pre-attack snapshot's document-ID
and text hash. It also requires the cited answer span to occur in that exact
verified text. A verified source keeps the answer. A missing, new, or altered
source causes abstention. An already abstained answer stays abstained. The
guard receives no answer key, target answer, attack label, strategy, or
support-document ID. It must not create or reseal the snapshot from the
attacked corpus. It does not infer that a mismatched source is malicious.

The separately retrieved clean-snapshot answer is a **diagnostic comparison**,
not a fallback emitted by the guard: the current hashing extractor has weak
clean-answer utility. The live UI must expose the pinned corpus digest,
source decision, selected source hash, clean diagnostic answer, and three
answer paths (undefended, v3 detector, snapshot guard). The previous v3
answer path and its historical metrics must remain identifiable.

## Development replay and decision rule

Replay the saved greedy and stealth validation payloads, one passage each,
under accepted-ingest and post-index-tamper surfaces. Compare undefended,
v3, and snapshot-guarded attack-target answers; count guard abstentions,
clean-answer alias matches, verified-source emissions, and added blocks over
v3. Report attack-source and answer-target metrics separately. Treat
answer-target matches from a clean source as such, rather than silently
scoring only source IDs. Also report the guard's behavior on the unmodified
clean corpus. The new guard is a useful *bounded* implementation if it
prevents all **attack-sourced** final answers under the stated source-control
boundary, never emits an answer from an unverified source, and does not
reduce clean-corpus alias recovery relative to the identical v3 answer path.
This is not a gate for broad open-domain factual-poison detection.

The no-guard v3 path is the paired baseline. No RL policy is part of the
live decision: the v14 defender PPO failed clean false-positive and GasLite
requirements. Its measured result may be displayed as research evidence,
but must not be presented as an active defense.
