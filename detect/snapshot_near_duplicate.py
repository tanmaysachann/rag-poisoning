"""Find altered near-copies in a separate, hash-pinned BEIR NQ snapshot."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from retrieval.trusted_snapshot_v12 import TrustedSnapshotSearch

TERMS = re.compile(r"[a-z0-9]+")


def normalized_text(text: str) -> str:
    return " ".join(TERMS.findall(text.casefold()))


class SnapshotNearDuplicate:
    """Research signal: a close changed copy of a pinned passage is suspicious.

    A pinned corpus proves byte identity, not factual truth. This signal is
    unavailable when no separate pre-attack snapshot exists.
    """

    def __init__(self, directory: Path) -> None:
        self.snapshot = TrustedSnapshotSearch(directory)

    def close(self) -> None:
        self.snapshot.close()

    def assess(self, candidate: str, *, limit: int = 12) -> dict:
        words = list(dict.fromkeys(
            word for word in TERMS.findall(candidate[:500].casefold())
            if len(word) >= 4 and word not in ENGLISH_STOP_WORDS
        ))
        if len(words) < 3:
            return {"status": "unassessable_short_text", "near_copy_score": 0.0,
                    "exact_snapshot_match": False, "matched_doc_id": None}
        # Distinctive opening terms often identify the source despite a few
        # character corruptions. Query three overlapping subsets for failover.
        term_sets = [words[:4], words[1:5], words[:3]]
        source = self.snapshot.connection
        seen = set()
        candidates = []
        for terms in term_sets:
            if len(terms) < 3:
                continue
            expression = " AND ".join(f'"{term}"' for term in terms)
            statement = (
                "SELECT doc_id,title,body,sha256 FROM passages WHERE passages MATCH ? "
                "ORDER BY bm25(passages,0,5,1,0),rowid LIMIT ?"
            )
            for row in source.execute(statement, (expression, limit)):
                doc_id = str(row["doc_id"])
                if doc_id in seen:
                    continue
                text = "\n".join(part for part in (str(row["title"]), str(row["body"])) if part)
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["sha256"]:
                    raise ValueError("Pinned snapshot passage changed")
                seen.add(doc_id)
                candidates.append((doc_id, text, row["sha256"]))
        candidate_digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        exact = any(digest == candidate_digest for _, _, digest in candidates)
        cleaned = normalized_text(candidate)
        best_score = 0.0
        best_id = None
        for doc_id, text, digest in candidates:
            if digest == candidate_digest:
                continue
            other = normalized_text(text)
            if not cleaned or not other:
                continue
            length_ratio = len(cleaned) / len(other)
            if not 0.55 <= length_ratio <= 1.8:
                continue
            similarity = SequenceMatcher(None, cleaned, other, autojunk=False).ratio()
            if similarity > best_score:
                best_score, best_id = similarity, doc_id
        return {"status": "scored" if candidates else "no_snapshot_neighbor",
                "near_copy_score": best_score if not exact else 0.0,
                "exact_snapshot_match": exact,
                "matched_doc_id": best_id,
                "retrieved_neighbors": len(candidates)}
