"""Local text tools (no network): searching the paper body for evidence."""

from __future__ import annotations

import re
from pathlib import Path

from paperflow.core.jsonutil import json_dumps

_WHITESPACE = re.compile(r"\s+")


def search_text(path: str, query: str, window: int = 300, max_hits: int = 3) -> str:
    """Find passages of `query` in the paper text file at `path`.

    Returns a JSON string of the form::

        {"query": ..., "total": N, "matches": [{"index": i, "passage": "..."}]}

    The Critic uses this to verify that a claimed quote actually appears in
    the paper body without re-sending the whole text to the LLM.
    """
    doc = Path(path)
    if not doc.is_file():
        return json_dumps({"error": f"text file not found: {path}", "total": 0, "matches": []})

    text = doc.read_text(encoding="utf-8", errors="replace")
    needle = query.strip().lower()
    if not needle:
        return json_dumps({"query": query, "total": 0, "matches": []})

    matches: list[dict] = []
    start = 0
    while True:
        idx = text.lower().find(needle, start)
        if idx == -1 or len(matches) >= max_hits:
            break
        passage = text[max(0, idx - window) : idx + len(needle) + window]
        passage = _WHITESPACE.sub(" ", passage).strip()
        matches.append({"index": idx, "passage": passage})
        start = idx + len(needle)

    return json_dumps({"query": query, "total": len(matches), "matches": matches})
