"""Targeted tests for demo-quality regressions found in real runs."""

from __future__ import annotations

import json
from pathlib import Path

from paperflow.agents.critic import CriticAgent
from paperflow.agents.synthesizer import SynthesizerAgent
from paperflow.config import Settings
from paperflow.core.board import TaskBoard
from paperflow.ingest import parse_entry
from paperflow.tools import build_default_registry
from tests.helpers import FakeLLM, text_response, tool_call_response

SETTINGS = Settings(deepseek_api_key="sk-test")


def test_critic_survives_many_search_rounds(tmp_path: Path, paper_text: str):
    """The Critic must tolerate verifying many claims (one search per claim)."""
    board = TaskBoard.create(
        tmp_path / "run" / "board.json", parse_entry("https://arxiv.org/abs/2601.12345")
    )
    (tmp_path / "full_text.txt").write_text(paper_text, encoding="utf-8")
    board.set_artifact("full_text", str(tmp_path / "full_text.txt"), "")
    claims = [
        {"claim": f"claim number {i}", "section": "Experiments", "quote": f"win rate {i}"}
        for i in range(10)
    ]
    (tmp_path / "reader_output.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")
    board.set_artifact("reader_output", str(tmp_path / "reader_output.json"), "")

    script = []
    for i in range(10):  # 10 sequential search_text calls
        script.append(
            lambda m, i=i: tool_call_response("search_text", {"path": str(tmp_path / "full_text.txt"), "query": f"claim {i}"})
        )
    script.append(
        text_response(
            json.dumps(
                {
                    "verdicts": [{"claim": "x", "verdict": "supported", "evidence_quote": "q", "note": ""}],
                    "limitations": [],
                    "overall_assessment": "ok",
                }
            )
        )
    )
    llm = FakeLLM(script={"Critic": script})
    agent = CriticAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    output = agent.run(board)
    assert output["verdicts"][0]["verdict"] == "supported"


def test_synthesizer_strips_leading_prose():
    """LLM preambles like 'Let me compose the final review.' must be dropped."""
    agent = SynthesizerAgent(build_default_registry(SETTINGS, ".cache-t"), FakeLLM(), SETTINGS)
    markdown = "# Dynamic Belief Networks\n\n## Overview\nbody\n"
    cleaned = agent.parse_output("I now have sufficient related work. Let me compose the final review.\n\n" + markdown)
    assert cleaned == markdown.strip()


def test_synthesizer_search_budget_terminates(tmp_path: Path, stub_network_tools):
    """A looping Synthesizer must be stopped deterministically by the search budget.

    Real-world regression: DeepSeek kept calling arxiv_search for related work
    until the tool round cap raised LLMError and the whole run failed.
    """
    from tests.helpers import tool_call_response

    board = TaskBoard.create(
        tmp_path / "run" / "board.json", parse_entry("https://arxiv.org/abs/2601.12345")
    )
    (tmp_path / "paper_info.json").write_text(
        json.dumps({"title": "Some Paper", "authors": ["A"]}), encoding="utf-8"
    )
    board.set_artifact("researcher_output", str(tmp_path / "paper_info.json"), "")
    (tmp_path / "reader_output.json").write_text(json.dumps({"summary": "s", "claims": []}), encoding="utf-8")
    board.set_artifact("reader_output", str(tmp_path / "reader_output.json"), "")

    # count how many times the real (stubbed) arxiv_search actually ran
    calls = {"n": 0}
    spec = stub_network_tools.get("arxiv_search")
    original = spec.func

    def spy(query, max_results=5):
        calls["n"] += 1
        return original(query, max_results=max_results)

    spec.func = spy

    markdown = (
        "# Some Paper\n\n## Overview\nx\n\n## Method\n\n## Key Claims & Evidence\n\n"
        "## Strengths\n\n## Limitations\n\n## Related Work\n\n"
        "## Relevance to My Research Direction\n\n## Suggested Next Steps\n"
    )

    def respond(messages):
        # a pathological model: keeps searching UNLESS the budget stop fires
        last_tool = next((m for m in reversed(messages) if m["role"] == "tool"), None)
        if last_tool and "BUDGET EXHAUSTED" in last_tool["content"]:
            return text_response(markdown)
        return tool_call_response("arxiv_search", {"query": "multi-agent LLM", "max_results": 3})

    llm = FakeLLM({"Synthesizer": [respond] * 25})
    agent = SynthesizerAgent(stub_network_tools, llm, SETTINGS)
    output = agent.run(board)
    assert output.startswith("# Some Paper")
    assert calls["n"] == 3  # the hard budget capped real searches at 3
