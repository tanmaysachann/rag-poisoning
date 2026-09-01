"""Optional live retrieval from the public MediaWiki API."""
from __future__ import annotations

import hashlib
import html
import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_documents(query: str, limit: int = 3, timeout: float = 5.0) -> list[dict]:
    """Fetch introductory extracts for live, query-dependent retrieval.

    Failure is intentionally non-fatal: the secure pipeline can still answer from
    its controlled local corpus or abstain when evidence is insufficient.
    """
    parameters = {
        "action": "query", "generator": "search", "gsrsearch": query,
        "gsrlimit": max(1, min(limit, 5)), "prop": "extracts|info",
        "exintro": "1", "explaintext": "1", "inprop": "url",
        "format": "json", "formatversion": "2",
    }
    request = Request(f"{WIKIPEDIA_API}?{urlencode(parameters)}",
                      headers={"User-Agent": "SentinelRAG-Review1/0.2 (academic demo)"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception:
        return []
    documents = []
    for page in payload.get("query", {}).get("pages", []):
        text = " ".join(html.unescape(page.get("extract", "")).split())
        if len(text) < 80:
            continue
        title = str(page.get("title", "Wikipedia result"))
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
        documents.append({
            "doc_id": 500000 + int(digest[:8], 16) % 400000,
            "title": title,
            "source_type": "live Wikipedia",
            "source_url": page.get("fullurl", "https://en.wikipedia.org"),
            "text": text[:2400],
            "live_source": True,
        })
    return documents
