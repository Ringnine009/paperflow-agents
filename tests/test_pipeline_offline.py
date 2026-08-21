"""End-to-end offline pipeline tests: all four agents + orchestrator + board.

The real orchestrator, board persistence and tool loop run against a
scripted FakeLLM; only the network tools are stubbed. This is the closest
thing to the production flow that runs without any external service.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperflow.config import Settings
from paperflow.core.orchestrator import OrchestratorError
from paperflow.pipeline import Pipeline
from tests.helpers import FakeLLM, text_response, tool_call_response, tool_result_text_path
from tests.pdfutil import make_pdf

SETTINGS = Settings(deepseek_api_key="sk-test")


def _researcher_script(pdf_path: Path) -> dict:
    """Researcher for a local-PDF entry: read_pdf -> relay TEXT_PATH -> JSON."""
    return {
        "Researcher": [
            lambda m: tool_call_response("read_pdf", {"path": str(pdf_path)}),
            lambda m: text_response(
                json.dumps(
                    {
                        "title": "On Social Deduction with Dynamic Belief Networks",
                        "authors": ["Tongji Student"],
                        "abstract": "We study multi-agent social deduction in Werewolf with DBNs.",
                        "doi": None,
                        "arxiv_id": None,
                        "url": None,
                        "published": None,
                        "source": "pdf",
                        "full_text_path": tool_result_text_path(m),
                        "full_text_chars": 200,
                        "note": "",
                    }
                )
            ),
        ]
    }


def _reader_script() -> dict:
    """The Reader always returns a fixed analysis of the fixture paper."""
    return {
        "Reader": [
            text_response(
                json.dumps(
                    {
                        "title": "On Social Deduction with Dynamic Belief Networks",
                        "summary": "DBNs improve social reasoning in hidden-role games.",
                        "sections": [{"heading": "Method", "content": "DBN posterior updates"}],
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
                        "open_questions": [],
                    }
                )
            )
        ]
    }


def _critic_script() -> dict:
    return {
        "Critic": [
            lambda m: tool_call_response("search_text", {"path": _full_text_path(m), "query": "44.2%"}),
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
                        "limitations": [{"issue": "single game size", "severity": "medium", "why": "9 players"}],
                        "overall_assessment": "sound but narrow evaluation",
                    }
                )
            ),
        ]
    }


def _full_text_path(messages: list[dict]) -> str:
    # the Reader context includes the full-text artifact path; critic scripts
    # reuse the path embedded in the user context message
    user = next(m["content"] for m in messages if m["role"] == "user")
    for line in user.splitlines():
        if line.startswith("FULL_TEXT_PATH:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("no FULL_TEXT_PATH in critic user context")


def _synthesizer_script() -> dict:
    markdown = (
        "# On Social Deduction with Dynamic Belief Networks\n\n"
        "> metadata: arxiv 2601.12345\n\n"
        "## Overview\nA study of DBNs in Werewolf.\n\n"
        "## Method\nbelief updates.\n\n"
        "## Key Claims & Evidence\n- 44.2% -> 68.8% (supported)\n\n"
        "## Strengths\nnovel model\n\n"
        "## Limitations\nnarrow eval\n\n"
        "## Related Work\n- [Related](https://arxiv.org/abs/2601.99999)\n\n"
        "## Relevance to My Research Direction\n**8/10** - directly aligned.\n\n"
        "## Suggested Next Steps\nreproduce on larger games\n"
    )
    return {
        "Synthesizer": [
            lambda m: tool_call_response("arxiv_search", {"query": "ti:multi-agent werewolf", "max_results": 3}),
            lambda m: text_response(markdown),
        ]
    }


def _stub_arxiv(pipe: Pipeline) -> None:
    """Replace the network arxiv_search tool with a fixed local answer."""
    from tests.helpers import stub_tool

    def fake(query, max_results=5):
        return json.dumps(
            [
                {
                    "arxiv_id": "2601.99999",
                    "title": "Related Multi-Agent Work",
                    "authors": ["Someone"],
                    "published": "2025-01-01",
                    "abstract": "related",
                    "pdf_url": "https://arxiv.org/pdf/2601.99999",
                    "abs_url": "https://arxiv.org/abs/2601.99999",
                    "doi": None,
                }
            ]
        )

    stub_tool(pipe.registry, "arxiv_search", fake)


@pytest.fixture
def fixture_pdf_full(tmp_path: Path, paper_text: str) -> Path:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf(paper_text))
    return pdf


def test_pipeline_happy_path(tmp_path: Path, fixture_pdf_full: Path):
    llm = FakeLLM(
        script={
            **_researcher_script(fixture_pdf_full),
            **_reader_script(),
            **_critic_script(),
            **_synthesizer_script(),
        }
    )
    pipe = Pipeline(settings=SETTINGS, out_dir=tmp_path / "out", llm=llm)
    _stub_arxiv(pipe)

    result = pipe.run(str(fixture_pdf_full))
    assert result["status"] == "done"
    assert Path(result["board"]).exists()

    board = json.loads(Path(result["board"]).read_text(encoding="utf-8"))
    assert board["status"] == "done"
    for agent in ("researcher", "reader", "critic", "synthesizer"):
        assert board["agents"][agent]["status"] == "done", agent

    report = Path(result["report"])
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "## Overview" in text
    assert "## Related Work" in text
    assert "Relevance to My Research Direction" in text

    # critic evidence really came from search_text over the artifact text
    full_text_artifact = Path(board["artifacts"]["full_text"]["path"]).read_text(encoding="utf-8")
    assert "44.2%" in full_text_artifact


def test_board_id_matches_run_dir(tmp_path: Path, fixture_pdf_full: Path):
    """The web dashboard resolves runs by id; board id must equal its dir name."""
    llm = FakeLLM(
        script={**_researcher_script(fixture_pdf_full), **_reader_script(), **_synthesizer_script()}
    )
    pipe = Pipeline(settings=SETTINGS, out_dir=tmp_path / "out", llm=llm)
    _stub_arxiv(pipe)

    result = pipe.run(str(fixture_pdf_full))
    run_dir = Path(result["board"]).parent
    board = json.loads(Path(result["board"]).read_text(encoding="utf-8"))
    assert run_dir.name == board["id"]
    assert result["run_id"] == board["id"]


def test_pipeline_aborts_when_critical_agent_fails(tmp_path: Path, fixture_pdf_full: Path):
    # Reader has no script -> FakeLLM raises -> critical failure aborts
    llm = FakeLLM(script=_researcher_script(fixture_pdf_full))
    pipe = Pipeline(settings=SETTINGS, out_dir=tmp_path / "out", llm=llm)
    _stub_arxiv(pipe)

    with pytest.raises(OrchestratorError):
        pipe.run(str(fixture_pdf_full))

    # board lives under the run dir; find it
    board_path = next((tmp_path / "out").rglob("board.json"))
    board = json.loads(board_path.read_text(encoding="utf-8"))
    assert board["status"] == "failed"
    assert board["agents"]["researcher"]["status"] == "done"
    assert board["agents"]["reader"]["status"] == "failed"
    assert "synthesizer" not in board["agents"]


def test_pipeline_continues_when_optional_critic_fails(tmp_path: Path, fixture_pdf_full: Path):
    llm = FakeLLM(
        script={**_researcher_script(fixture_pdf_full), **_reader_script(), **_synthesizer_script()}
    )  # no Critic script
    pipe = Pipeline(settings=SETTINGS, out_dir=tmp_path / "out", llm=llm)
    _stub_arxiv(pipe)

    result = pipe.run(str(fixture_pdf_full))
    assert result["status"] == "done"
    board = json.loads(Path(result["board"]).read_text(encoding="utf-8"))
    assert board["agents"]["critic"]["status"] == "failed"
    assert board["agents"]["synthesizer"]["status"] == "done"
    # synthesizer still produced a report without the critic
    assert Path(result["report"]).exists()
