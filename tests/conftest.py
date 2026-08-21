"""Shared pytest fixtures for PaperFlow offline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperflow.core.tools import ToolRegistry
from tests.helpers import FIXTURE_PAPER_TEXT, stub_tool


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """A scratch directory standing in for a run output directory."""
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "cache").mkdir()
    return tmp_path


@pytest.fixture
def fixture_pdf(tmp_path: Path) -> Path:
    """A real minimal PDF whose text is FIXTURE_PAPER_TEXT's first line."""
    from tests.pdfutil import make_pdf

    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf("Social Deduction with Dynamic Belief Networks\n"))
    return path


@pytest.fixture
def stub_network_tools(tmp_path: Path) -> ToolRegistry:
    """Default registry with network tools stubbed to fixture data."""
    from paperflow.config import Settings
    from paperflow.tools import build_default_registry

    registry = build_default_registry(Settings(deepseek_api_key="sk-test"), tmp_path / "cache")
    stub_tool(
        registry,
        "arxiv_search",
        lambda query, max_results=5: json.dumps(
            [
                {
                    "arxiv_id": "2601.12345",
                    "title": "On Social Deduction with Dynamic Belief Networks",
                    "authors": ["Tongji Student", "Another Author"],
                    "published": "2026-01-10",
                    "abstract": "We study multi-agent social deduction in Werewolf with DBNs.",
                    "pdf_url": "https://arxiv.org/pdf/2601.12345",
                    "abs_url": "https://arxiv.org/abs/2601.12345",
                    "doi": None,
                }
            ]
        ),
    )
    return registry


@pytest.fixture
def paper_text() -> str:
    return FIXTURE_PAPER_TEXT
