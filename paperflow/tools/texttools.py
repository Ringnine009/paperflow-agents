"""Local text tools (no network): searching the paper body for evidence."""

from __future__ import annotations

import re
from pathlib import Path

from paperflow.core.jsonutil import json_dumps

_WHITESPACE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to a single space, lowercased.

    Used by quote verification so that a quote extracted across wrapped
    lines ("win rate\n44.2%") still matches the source text.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


def verify_quote(full_text: str, quote: str, min_chars: int = 4) -> dict:
    """Deterministic quote-presence check against the paper text.

    Whitespace-normalized and case-insensitive. This is a *code* guarantee:
    ``found`` is True only when the (normalized) quote literally occurs in
    the full text - no LLM involved.

    Returns ``{"found": bool, "reason": str}``.
    """
    if not quote or not quote.strip():
        return {"found": False, "reason": "empty quote"}
    needle = normalize_ws(quote)
    if len(needle) < min_chars:
        return {"found": False, "reason": "quote too short to verify"}
    haystack = normalize_ws(full_text)
    if not haystack:
        return {"found": False, "reason": "no full text available"}
    if needle in haystack:
        return {"found": True, "reason": "quote found verbatim (whitespace-normalized)"}
    return {"found": False, "reason": "quote not found in the paper text"}


def verify_claims(full_text: str, claims: list[dict]) -> list[dict]:
    """Annotate every claim with its deterministic verification result.

    Each claim dict gains ``quote_verified`` (bool) and
    ``quote_verification`` (reason string). Claims without a quote are
    marked not verified, so downstream stages never assume LLM provenance.
    """
    for claim in claims:
        result = verify_quote(full_text, claim.get("quote", ""))
        claim["quote_verified"] = result["found"]
        claim["quote_verification"] = result["reason"]
    return claims


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
