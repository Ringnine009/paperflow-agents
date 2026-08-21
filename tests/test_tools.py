"""Tests for paperflow.core.tools (registry) and paperflow.tools.parsers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperflow.core.tools import ToolError, ToolRegistry, make_schema
from paperflow.tools.parsers import parse_arxiv_atom, parse_crossref_work

ARXIV_ATOM_FIXTURE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:53:30Z</published>
    <title>Attention Is All You Need</title>
    <summary>We propose a new simple network architecture, the Transformer.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/1706.03762v7"/>
    <arxiv:doi>10.48550/arXiv.1706.03762</arxiv:doi>
  </entry>
</feed>
"""

CROSSREF_FIXTURE = {
    "status": "ok",
    "message": {
        "title": ["Werewolf Multi-Agent with DBN"],
        "author": [{"given": "Tongji", "family": "Student"}],
        "published-print": {"date-parts": [[2026, 5, 1]]},
        "DOI": "10.54254/2753-8818/2026.DL34010",
        "URL": "https://example.org/article",
        "link": [{"URL": "https://example.org/paper.pdf", "content-type": "application/pdf"}],
        "abstract": "<jats:p>We study social deduction with <jats:italic>DBNs</jats:italic>.</jats:p>",
    },
}


def test_registry_register_and_call():
    reg = ToolRegistry()
    reg.register_func(
        "echo",
        "Echo the input back",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: text.upper(),
    )
    assert "echo" in reg
    assert reg.call("echo", text="hi") == "HI"
    assert reg.names() == ["echo"]


def test_registry_call_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        reg.call("nope")


def test_registry_schemas_are_openai_compatible():
    reg = ToolRegistry()
    reg.register_func("echo", "Echo", make_schema({"text": {"type": "string"}}), lambda text: text)
    schemas = reg.schemas()
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"
    assert "parameters" in schemas[0]["function"]


def test_parse_arxiv_atom():
    results = parse_arxiv_atom(ARXIV_ATOM_FIXTURE)
    assert len(results) == 1
    item = results[0]
    assert item["arxiv_id"] == "1706.03762"
    assert item["title"] == "Attention Is All You Need"
    assert item["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert item["pdf_url"] == "http://arxiv.org/pdf/1706.03762v7"
    assert item["doi"] == "10.48550/arXiv.1706.03762"
    assert "Transformer" in item["abstract"]


def test_parse_arxiv_atom_empty_feed():
    feed = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_arxiv_atom(feed) == []


def test_parse_crossref_work():
    item = parse_crossref_work(json.dumps(CROSSREF_FIXTURE))
    assert item["title"] == "Werewolf Multi-Agent with DBN"
    assert item["authors"] == ["Tongji Student"]
    assert item["doi"] == "10.54254/2753-8818/2026.DL34010"
    assert item["pdf_url"] == "https://example.org/paper.pdf"
    assert "DBNs" in item["abstract"]  # JATS tags stripped


def test_parse_crossref_work_malformed():
    with pytest.raises(ValueError):
        parse_crossref_work("{not json")


def test_search_text_finds_matching_passages(tmp_path: Path):
    from paperflow.tools.texttools import search_text

    doc = tmp_path / "doc.txt"
    doc.write_text(
        "The DBN updates beliefs after each statement. "
        "Then voting accuracy improves. " * 20,
        encoding="utf-8",
    )
    result = search_text(path=str(doc), query="voting accuracy", max_hits=2)
    parsed = json.loads(result)
    assert parsed["total"] == 2
    assert len(parsed["matches"]) == 2
    assert all("voting accuracy" in h["passage"].lower() for h in parsed["matches"])


def test_search_text_no_match(tmp_path: Path):
    from paperflow.tools.texttools import search_text

    doc = tmp_path / "doc.txt"
    doc.write_text("nothing here", encoding="utf-8")
    result = json.loads(search_text(path=str(doc), query="zzzz"))
    assert result["total"] == 0
    assert result["matches"] == []
