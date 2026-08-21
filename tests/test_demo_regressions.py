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
