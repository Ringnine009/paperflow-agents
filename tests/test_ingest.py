"""Tests for paperflow.ingest: entry parsing and slug sanitization."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperflow.ingest import InputSpec, arxiv_id_from_url, is_doi, parse_entry, sanitize_slug


def test_arxiv_abs_url():
    spec = parse_entry("https://arxiv.org/abs/1706.03762")
    assert spec.kind == "arxiv"
    assert spec.value == "1706.03762"


def test_arxiv_pdf_url():
    spec = parse_entry("http://arxiv.org/pdf/1706.03762v7")
    assert spec.kind == "arxiv"
    assert spec.value == "1706.03762"


def test_arxiv_export_api_url():
    spec = parse_entry("https://export.arxiv.org/api/query?id_list=1706.03762")
    assert spec.kind == "arxiv"
    assert spec.value == "1706.03762"


def test_doi_entry():
    spec = parse_entry("10.54254/2753-8818/2026.DL34010")
    assert spec.kind == "doi"
    assert spec.value == "10.54254/2753-8818/2026.DL34010"


def test_doi_with_prefix():
    spec = parse_entry("doi:10.1000/xyz123")
    assert spec.kind == "doi"
    assert spec.value == "10.1000/xyz123"


def test_local_pdf_path(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    spec = parse_entry(str(pdf))
    assert spec.kind == "pdf"
    assert spec.value == str(pdf)


def test_generic_url():
    spec = parse_entry("https://example.com/paper.html")
    assert spec.kind == "url"
    assert spec.value == "https://example.com/paper.html"


def test_free_text_title():
    spec = parse_entry("Attention Is All You Need")
    assert spec.kind == "title"
    assert spec.value == "Attention Is All You Need"


def test_pdf_override_attached(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    spec = parse_entry("https://arxiv.org/abs/1706.03762", pdf_override=str(pdf))
    assert spec.pdf_override == str(pdf)


def test_arxiv_id_from_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert arxiv_id_from_url("https://arxiv.org/abs/1706.03762v7") == "1706.03762"
    assert arxiv_id_from_url("https://example.com/x") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("10.1109/TPAMI.2023.123", True),
        ("10.48550/arXiv.1706.03762", True),
        ("doi:10.1000/abc", True),
        ("1706.03762", False),
        ("https://arxiv.org/abs/1706.03762", False),
        ("Attention", False),
    ],
)
def test_is_doi(text: str, expected: bool):
    assert is_doi(text) is expected


def test_sanitize_slug():
    assert sanitize_slug("Attention Is All You Need!") == "attention-is-all-you-need"
    assert sanitize_slug("Paper / With: Slashes") == "paper-with-slashes"
    assert sanitize_slug("短标题")  # CJK survives, no exception
    assert len(sanitize_slug("x" * 200)) <= 70
