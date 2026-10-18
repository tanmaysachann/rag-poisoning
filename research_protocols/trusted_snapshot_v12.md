# V12 qrel-free reference selection feasibility check

Frozen before scoring v12. The PoisonArena 100-question aggregate and its
29-question v10 subset have already been inspected; all scores here are
development, not blind confirmation.

Build a separate SQLite FTS5 snapshot from every passage in the pinned official
BEIR NQ corpus zip (MD5 `d4d3d2e48787a744b6f6e691ff534307`). Do not add
PoisonArena attacks or use qrels, answers, or targets in the index. Record
the archive, database, and document hashes. This simulates a frozen source
channel; it is not an operator-reviewed or inherently truthful source.

For each of PoisonArena's 100 question texts, retrieve up to 10 passages using
fixed lexical FTS5 rules. First seek documents containing all non-stopword
query terms; fill remaining slots with an any-term query. Break rank ties by
database row order. Run the pinned v8 NQ extractive reader on each retrieved
passage. No qrel, answer key or attack target enters search or QA.

On the 71 v10 training-question subset, compare three policies: top-ranked
passage only; maximum QA-confidence span among top five; and an answer that
appears in at least two top-ten passages. For each, set a minimum QA score
using calibration answers such that wrong emitted answers are at most 10% of
all calibration questions, maximizing strict correct answer matches and then
preferring fewer wrong emissions. Choose the policy with the most calibration
correct matches, breaking ties by fewer wrong emissions and then in the order
listed above. Freeze policy and score before evaluating the other 29 questions.

Report strict answer matches, nonmatches, abstentions, qrel recall at 1/5/10,
and source document IDs and hashes. Published answers and qrels are read only
for scoring after predictions are saved. Gate for further work: at least 30%
strict correct answer coverage, at most 10% wrong emissions over all questions,
and at least 30% qrel recall at 10 on the 29-question development subset.
Passing would justify a new, genuinely independent domain and human source
review before live use. Failure means this snapshot selector cannot support
reliable contradiction quarantine.
