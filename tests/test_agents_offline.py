"""Per-agent offline tests: each agent driven by a scripted FakeLLM.

These exercise the real agent logic (context assembly, tool loop, artifact
persistence) without any network or real LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperflow.agents.critic import CriticAgent
from paperflow.agents.reader import ReaderAgent
from paperflow.agents.researcher import ResearcherAgent
from paperflow.agents.synthesizer import SynthesizerAgent
from paperflow.config import Settings
from paperflow.core.board import TaskBoard
from paperflow.core.llm import LLMClient
from paperflow.ingest import parse_entry
from paperflow.tools import build_default_registry
from tests.helpers import FakeLLM, text_response, tool_call_response, tool_result_text_path
from tests.pdfutil import make_pdf

SETTINGS = Settings(deepseek_api_key="sk-test")


def make_board(tmp_path: Path, entry: str, pdf_override: str | None = None) -> TaskBoard:
    path = tmp_path / "run" / "board.json"
    return TaskBoard.create(path, parse_entry(entry, pdf_override=pdf_override))


def make_llm(script: dict) -> LLMClient:
    return FakeLLM(script=script)


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

def test_researcher_ingests_local_pdf(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf("Social Deduction with Dynamic Belief Networks\n"))
    board = make_board(tmp_path, str(pdf))

    llm = make_llm(
        {
            "Researcher": [
                lambda m: tool_call_response("read_pdf", {"path": str(pdf)}),
                lambda m: text_response(
                    json.dumps(
                        {
                            "title": "Social Deduction with Dynamic Belief Networks",
                            "authors": ["Tongji Student"],
                            "abstract": "We study multi-agent social deduction in Werewolf.",
                            "doi": None,
                            "arxiv_id": None,
                            "url": None,
                            "published": None,
                            "source": "pdf",
                            "full_text_path": tool_result_text_path(m),
                            "full_text_chars": 100,
                            "note": "",
                        }
                    )
                ),
            ]
        }
    )
    agent = ResearcherAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    output = agent.run(board)

    assert output["title"].startswith("Social Deduction")
    # full text must be materialized as an artifact the Reader can consume
    full_text = board.artifacts["full_text"]
    assert Path(full_text["path"]).read_text(encoding="utf-8").startswith("Social Deduction")
    # paper metadata artifact persisted
    paper_info = json.loads(Path(board.artifacts["researcher_output"]["path"]).read_text(encoding="utf-8"))
    assert paper_info["source"] == "pdf"


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def test_reader_produces_claims_and_data_points(tmp_path: Path, paper_text: str):
    board = make_board(tmp_path, "https://arxiv.org/abs/2601.12345")
    board.set_artifact("researcher_output", str(tmp_path / "paper_info.json"), "")
    (tmp_path / "paper_info.json").write_text(
        json.dumps({"title": "On Social Deduction with Dynamic Belief Networks", "authors": []}),
        encoding="utf-8",
    )
    (tmp_path / "full_text.txt").write_text(paper_text, encoding="utf-8")
    board.set_artifact("full_text", str(tmp_path / "full_text.txt"), "")

    llm = make_llm(
        {
            "Reader": [
                text_response(
                    json.dumps(
                        {
                            "title": "On Social Deduction with Dynamic Belief Networks",
                            "summary": "DBNs improve social reasoning.",
                            "sections": [{"heading": "Method", "content": "DBN updates beliefs"}],
                            "claims": [
                                {
                                    "claim": "DBN raises villager win rate from 44.2% to 68.8%",
                                    "section": "Experiments",
                                    "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
                                    "evidence": "abstract",
                                }
                            ],
                            "data_points": [
                                {"metric": "villager win rate (baseline)", "value": "44.2%", "context": "no beliefs"}
                            ],
                            "method_summary": "posterior updates after statements",
                            "open_questions": ["scale to larger games"],
                        }
                    )
                )
            ]
        }
    )
    agent = ReaderAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    output = agent.run(board)
    assert len(output["claims"]) == 1
    # quotes may span wrapped lines in the raw text; compare whitespace-normalized
    normalized = " ".join(paper_text.split())
    assert output["claims"][0]["quote"] in normalized  # quote was drawn from the text
    assert output["data_points"][0]["value"] == "44.2%"


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def test_critic_verifies_claims_with_search_text(tmp_path: Path, paper_text: str):
    board = make_board(tmp_path, "https://arxiv.org/abs/2601.12345")
    (tmp_path / "full_text.txt").write_text(paper_text, encoding="utf-8")
    board.set_artifact("full_text", str(tmp_path / "full_text.txt"), "")
    (tmp_path / "reader_output.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim": "DBN raises villager win rate from 44.2% to 68.8%",
                        "section": "Experiments",
                        "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    board.set_artifact("reader_output", str(tmp_path / "reader_output.json"), "")

    llm = make_llm(
        {
            "Critic": [
                lambda m: tool_call_response(
                    "search_text", {"path": str(tmp_path / "full_text.txt"), "query": "44.2%"}
                ),
                lambda m: text_response(
                    json.dumps(
                        {
                            "verdicts": [
                                {
                                    "claim": "DBN raises villager win rate from 44.2% to 68.8%",
                                    "verdict": "supported",
                                    "evidence_quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
                                    "note": "verbatim in abstract",
                                }
                            ],
                            "limitations": [
                                {"issue": "single game size", "severity": "medium", "why": "only 9 players"}
                            ],
                            "overall_assessment": "sound but narrow evaluation",
                        }
                    )
                ),
            ]
        }
    )
    agent = CriticAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    output = agent.run(board)
    assert output["verdicts"][0]["verdict"] == "supported"
    assert len(output["limitations"]) == 1


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def test_synthesizer_writes_markdown_report(tmp_path: Path, stub_network_tools):
    board = make_board(tmp_path, "https://arxiv.org/abs/2601.12345")
    (tmp_path / "paper_info.json").write_text(
        json.dumps(
            {"title": "On Social Deduction with Dynamic Belief Networks", "authors": ["Tongji Student"]}
        ),
        encoding="utf-8",
    )
    board.set_artifact("researcher_output", str(tmp_path / "paper_info.json"), "")
    board.set_artifact("reader_output", str(tmp_path / "reader_output.json"), "")
    (tmp_path / "reader_output.json").write_text(json.dumps({"summary": "x", "claims": []}), encoding="utf-8")

    markdown = (
        "# On Social Deduction with Dynamic Belief Networks\n\n"
        "## Overview\nA study of DBNs in Werewolf.\n\n"
        "## Method\nbelief updates.\n\n"
        "## Key Claims & Evidence\n- claim supported\n\n"
        "## Strengths\nnovel model\n\n"
        "## Limitations\nnarrow eval\n\n"
        "## Related Work\n- [Related](https://arxiv.org/abs/2601.99999)\n\n"
        "## Relevance to My Research Direction\n**8/10** - directly aligned.\n\n"
        "## Suggested Next Steps\nreproduce on larger games\n"
    )
    llm = make_llm(
        {
            "Synthesizer": [
                lambda m: tool_call_response("arxiv_search", {"query": "ti:multi-agent werewolf", "max_results": 3}),
                lambda m: text_response(markdown),
            ]
        }
    )
    agent = SynthesizerAgent(stub_network_tools, llm, SETTINGS)
    output = agent.run(board)

    assert "## Overview" in output
    report_path = board.report["path"]
    assert Path(report_path).read_text(encoding="utf-8").strip() == markdown.strip()
    assert "Relevance to My Research Direction" in Path(report_path).read_text(encoding="utf-8")
