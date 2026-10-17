# V8 trusted-reference answer pilot

Written before scoring v8 reference answers. The v7 external test aggregate is
already known, so this pilot's test split is question-disjoint but **not a new
sealed benchmark**. It can reject a bad approach; publication-level success
requires a fresh attack collection.

## Trust boundary

The BEIR relevance-judged document chosen from the archived clean top 10 is a
benchmark-only proxy for an independently reviewed, hash-verified source. The
answer key, target string, and attack documents are never QA inputs for the
reference answer. This benchmark cannot establish real-world source origin.

At live inference, a trusted reference must come through the repository's
operator-reviewed `SourceAttestations` with a verified document hash and an
explicit trusted-origin allowlist. If no such reference is retrieved, abstain.
The signed origin attests provenance, not factual truth.

## Reader and decision

Use the pinned `deepset/tinyroberta-squad2` extractive QA model, revision
`75c968b8219e055ce1a89533057b142bae65450f`, SHA-256
`39955cbf5f636762a2ef642b718c8c6e172d153c94a13de40b37efd00822cd80`
for its `model.safetensors`. Run the same query-conditioned reader on a trusted
reference and on retrieved candidates. Never pass the correct or target answer
as context. The reference answer is the only answer eligible for output;
candidate answers can supply conflict evidence but cannot replace it.

Amendment after the tiny reader's calibration result, before scoring any v8
test answers: compare a second pinned reader,
`nlpconnect/roberta-base-squad2-nq` revision
`8725d9fd10e9288c94fb259b513190b045c54af6`, weight SHA-256
`b2f61c7b1710fcc6826fc0cc12dca8c59c0fbdadd82c05bb6ebff2028583fcf6`.
This reader was trained on Natural Questions and may share source-data style
with BEIR NQ. Choose the reader using calibration utility alone: correct
strict matches minus twice wrong emitted answers at that reader's cutoff.
Evaluate the chosen reader once on the question-disjoint set.

Calibration decision: the NQ reader scored utility 42 (82 strict correct,
20 wrong emitted), versus tiny reader utility 40 (68 strict correct, 14 wrong
emitted). Freeze the NQ reader and its `0.24418172240257263` confidence
cutoff for the question-disjoint evaluation.

Replay comparison: use the same frozen NQ reader on every archived top-10
candidate. The unrestricted baseline picks the span with the highest QA
score. Choose its minimum score using **clean calibration** questions with
the same correct-minus-twice-wrong utility, then freeze it. The restricted
path emits only from the benchmark reference if it remains in the attacked
top 10; otherwise it abstains. Compare correct matches, wrong emissions,
attacker-target matches, abstentions, and conditional outcomes when the
question's own poison was retrieved. Candidate/reference answer differences
are diagnostic disagreements, not proof of malicious content.

Measure reference answer correctness and abstention on the 234-question v7
calibration split. Pick one reader confidence cutoff on calibration alone to
favor supported correct answers over wrong answers, and freeze it before
scoring the 687-question question-disjoint split. Report both normalized
substring and token-F1 metrics because NQ answer strings often paraphrase
passage spans. For attack replay, compare output on clean, naive, and
CorruptRAG-AK retrievals, including reference availability. Report the
fraction of poison documents producing an answer different from the reference,
but call these **disagreements**, not proven maliciousness; a clean off-topic
passage can also disagree.

## Decision rule

This approach is useful only if it materially increases correct answer
coverage over the current extractive baseline (61/687 strict matches) while
keeping attacker-target answers unchanged between clean and attacked retrieval
and maintaining a tolerable abstention rate. No live integration is justified
until the reference can be supplied from a real trusted source registry and
the answer path passes a fresh external attack-family evaluation.
