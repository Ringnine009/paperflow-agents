"""Smoke tests against the real arXiv / Crossref APIs (excluded by default).

Run explicitly with::

    pytest -m smoke

Set ``PAPERFLOW_SKIP_NETWORK=1`` to skip even then (offline environments).
These tests are the closest thing to end-to-end validation of the
Researcher's data sources without spending LLM tokens.
"""

from __future__ import annotations

import json
import os

import pytest

from paperflow.config import Settings
from paperflow.tools import build_default_registry
from paperflow.tools.parsers import parse_arxiv_atom

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("PAPERFLOW_SKIP_NETWORK") == "1",
        reason="network disabled via PAPERFLOW_SKIP_NETWORK=1",
    ),
]

SETTINGS = Settings(deepseek_api_key="sk-not-needed-for-apis")


def test_arxiv_lookup_by_id():
    from paperflow.tools import net

    net.throttle("export.arxiv.org", min_interval=3.0)
    xml = net.http_get(
        "https://export.arxiv.org/api/query?search_query=id:1706.03762&start=0&max_results=1",
        timeout=60,
    )
    items = parse_arxiv_atom(xml)
    assert items, "arXiv returned no entries for id:1706.03762"
    assert "Attention Is All You Need" in items[0]["title"]
    assert items[0]["pdf_url"]  # a PDF link must be present


def test_arxiv_search_by_title():
    from paperflow.tools import net

    net.throttle("export.arxiv.org", min_interval=3.0)
    xml = net.http_get(
        "https://export.arxiv.org/api/query?search_query=ti:%22transformer%22&start=0&max_results=3",
        timeout=60,
    )
    items = parse_arxiv_atom(xml)
    assert 1 <= len(items) <= 3


def test_crossref_doi_resolution_for_werewolf_paper():
    """The portfolio's own paper DOI must resolve via Crossref."""
    registry = build_default_registry(SETTINGS, ".cache-smoke")
    result = json.loads(registry.call("resolve_doi", doi="10.54254/2753-8818/2026.DL34010"))
    assert result.get("title"), f"Crossref did not return a title: {result}"
    # Crossref normalizes DOI case; compare case-insensitively
    assert result.get("doi", "").lower() == "10.54254/2753-8818/2026.dl34010"
