"""Read a frozen, separate BEIR NQ FTS snapshot without benchmark qrels."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from data.build_trusted_snapshot_v12 import digest


TERMS = re.compile(r"[a-z0-9]+")


class TrustedSnapshotSearch:
    """Research-only: a pinned corpus is not proof that its passages are true."""

    def __init__(self, directory: Path):
        manifest_path = directory / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        database = directory / self.manifest.get("database_filename", "beir_nq_fts5.sqlite")
        if digest(database, "sha256") != self.manifest["database_sha256"]:
            raise ValueError("Trusted snapshot database changed")
        self.connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro",
                                          uri=True, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def search(self, question: str, *, limit: int = 10) -> list[dict]:
        if not question.strip() or limit < 1:
            raise ValueError("Question and positive limit are required")
        terms = list(dict.fromkeys(
            token for token in TERMS.findall(question.casefold())
            if token not in ENGLISH_STOP_WORDS))
        if not terms:
            return []
        # Quoted single terms prevent FTS operator injection from a question.
        quoted = [f'"{term}"' for term in terms]
        results = []
        seen = set()
        for stage, separator in (("all_terms", " AND "), ("any_term", " OR ")):
            expression = separator.join(quoted)
            statement = ("SELECT doc_id,title,body,sha256,bm25(passages,0,5,1,0) AS lexical_score "
                         "FROM passages WHERE passages MATCH ? ORDER BY lexical_score,rowid LIMIT ?")
            for row in self.connection.execute(statement, (expression, limit * 3)):
                doc_id = str(row["doc_id"])
                if doc_id in seen:
                    continue
                title, body = str(row["title"]), str(row["body"])
                text = "\n".join(part for part in (title, body) if part)
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["sha256"]:
                    raise ValueError(f"Indexed passage changed: {doc_id}")
                results.append({"doc_id": doc_id, "text": text,
                                "sha256": row["sha256"], "stage": stage,
                                "lexical_score": float(row["lexical_score"])})
                seen.add(doc_id)
                if len(results) >= limit:
                    return results
        return results
