# V13 dense reranking of the frozen source snapshot

Recorded after v12 failed and its development scores were inspected. This is
another development experiment on the same 100 PoisonArena questions, not
independent confirmation.

Keep the identical v12 pinned full-corpus FTS5 index and 71/29 split. For each
question retrieve 100 passages using the fixed lexical selector, then compute
query/passage cosine similarity with the locally pinned
`facebook/contriever-msmarco` checkpoint, SHA256
`08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8`.
Compare dense cosine ordering with reciprocal-rank fusion of dense and lexical
rank (`k=60`). Run the identical v12 NQ extractive reader only on the union of
their top ten passages. No qrel, published answer, or attacker target is used
in retrieval, reranking, or QA.

Save per-question passage IDs, hashes, and QA predictions before opening
answer keys or qrels. On the 71 calibration questions, compare the two
rankings and the v12 top1, max-QA-five, and two-document-consensus policies.
For each combination, choose a QA cutoff allowing at most 10% wrong emitted
answers on calibration and maximizing strict correct matches. Freeze the
combination with most correct matches, then fewer wrong emissions, then prefer
dense over fusion and top1 over max-QA-five over consensus in ties. Evaluate
that one combination on the 29 development questions. Report all candidate
calibration policies, qrel recall for both rankings, correct/nonmatching
answers, and abstentions. Compare v12 at its frozen policy.

Use the same feasibility gate: at least 9/29 strict correct answers, no more
than 2/29 nonmatching emitted answers, and qrel recall at ten of at least
9/29. This is still far below a live deployment standard. The benchmark
snapshot is not an operator-reviewed source of truth.
