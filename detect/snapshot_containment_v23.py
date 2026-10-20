"""Find pinned source passages embedded inside longer edited documents."""

from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from detect.snapshot_near_duplicate import TERMS, normalized_text
from retrieval.trusted_snapshot_v12 import TrustedSnapshotSearch


class SnapshotContainment:
    """Research integrity check with opening, middle, and ending retrieval."""

    def __init__(self, directory: Path) -> None:
        self.snapshot = TrustedSnapshotSearch(directory)

    def close(self) -> None:
        self.snapshot.close()

    @staticmethod
    def _queries(candidate: str) -> list[str]:
        regions = [candidate[:500], candidate[-500:]]
        if len(candidate) > 1000:
            middle = len(candidate) // 2
            regions.append(candidate[max(0, middle - 250):middle + 250])
        queries = []
        for region in regions:
            words = list(dict.fromkeys(
                word for word in TERMS.findall(region.casefold())
                if len(word) >= 4 and word not in ENGLISH_STOP_WORDS))
            for terms in (words[:4], words[1:5], words[-4:]):
                if len(terms) < 3:
                    continue
                expression = " AND ".join(f'"{term}"' for term in terms)
                if expression not in queries:
                    queries.append(expression)
        return queries

    def assess(self, candidate: str, *, limit: int = 12) -> dict:
        expressions = self._queries(candidate)
        if not expressions:
            return {"status": "unassessable_short_text", "near_copy_score": 0.0,
                    "exact_snapshot_match": False, "matched_doc_id": None,
                    "match_method": None, "retrieved_neighbors": 0}
        connection = self.snapshot.connection
        seen = set()
        neighbors = []
        statement = (
            "SELECT doc_id,title,body,sha256 FROM passages WHERE passages MATCH ? "
            "ORDER BY bm25(passages,0,5,1,0),rowid LIMIT ?"
        )
        for expression in expressions:
            for row in connection.execute(statement, (expression, limit)):
                doc_id = str(row["doc_id"])
                if doc_id in seen:
                    continue
                text = "\n".join(part for part in (str(row["title"]), str(row["body"])) if part)
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["sha256"]:
                    raise ValueError("Pinned snapshot passage changed")
                neighbors.append((doc_id, text, row["sha256"]))
                seen.add(doc_id)
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        exact_ids = [doc_id for doc_id, _, source_sha in neighbors if source_sha == digest]
        if exact_ids:
            return {
                "status": "scored", "near_copy_score": 0.0,
                "exact_snapshot_match": True, "matched_doc_id": exact_ids[0],
                "match_method": "exact_snapshot_match",
                "retrieved_neighbors": len(neighbors),
            }
        cleaned = normalized_text(candidate)
        best_score = 0.0
        best_id = None
        best_method = None
        for doc_id, source, source_sha in neighbors:
            if source_sha == digest:
                continue
            original = normalized_text(source)
            if not cleaned or not original:
                continue
            matcher = SequenceMatcher(None, cleaned, original, autojunk=False)
            length_ratio = len(cleaned) / len(original)
            similarity = matcher.ratio() if 0.55 <= length_ratio <= 1.8 else 0.0
            # One or more inserted attack sentences may dwarf a short source.
            # Require substantial source coverage before treating containment
            # as an integrity signal; a short shared phrase is insufficient.
            coverage = (matcher.find_longest_match(0, len(cleaned), 0, len(original)).size
                        / len(original) if len(original) >= 80 else 0.0)
            score = max(similarity, coverage)
            if score > best_score:
                best_score, best_id = score, doc_id
                best_method = "contained_source" if coverage > similarity else "near_copy"
        return {
            "status": "scored" if neighbors else "no_snapshot_neighbor",
            "near_copy_score": best_score,
            "exact_snapshot_match": False,
            "matched_doc_id": best_id,
            "match_method": best_method,
            "retrieved_neighbors": len(neighbors),
        }
