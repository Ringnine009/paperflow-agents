"""Deterministic quote verification (P0).

Reader's quotes are checked against the full text by *code* - a quote that
cannot be found is annotated quote_verified:false and the Critic's final
verdict is downgraded to "unverified" regardless of what the LLM said.
Quote presence is a deterministic guarantee, not an LLM opinion.
"""

from __future__ import annotations

import json
from pathlib import Path

from paperflow.agents.critic import CriticAgent
from paperflow.agents.reader import ReaderAgent
from paperflow.config import Settings
from paperflow.core.board import TaskBoard
from paperflow.ingest import parse_entry
from paperflow.tools import build_default_registry
from paperflow.tools.texttools import verify_claims, verify_quote
from tests.helpers import FIXTURE_PAPER_TEXT, FakeLLM, text_response

SETTINGS = Settings(deepseek_api_key="sk-test")


def make_board(tmp_path: Path, entry: str = "https://arxiv.org/abs/2601.12345") -> TaskBoard:
    return TaskBoard.create(tmp_path / "run" / "board.json", parse_entry(entry))


# ---------------------------------------------------------------------------
# pure function: verify_quote
# ---------------------------------------------------------------------------

def test_verify_quote_matches_across_line_breaks():
    text = "the DBN\nraises the win rate\nfrom 44.2% to 68.8%."
    assert verify_quote(text, "DBN raises the win rate from 44.2% to 68.8%")["found"] is True


def test_verify_quote_case_insensitive():
    text = "dynamic belief networks update posterior beliefs"
    assert verify_quote(text, "DYNAMIC BELIEF NETWORKS")["found"] is True


def test_verify_quote_missing():
    text = "the baseline win rate is 44.2%"
    result = verify_quote(text, "a completely different sentence")
    assert result["found"] is False
    assert "not found" in result["reason"]
    assert result["loc"] is None
    assert result["context"] is None


def test_verify_quote_reports_location_and_context():
    text = "intro paragraph\n\n" + "the DBN raises the win rate from 44.2% to 68.8%.\n\n" + "outro paragraph"
    result = verify_quote(text, "DBN raises the win rate from 44.2% to 68.8%")
    assert result["found"] is True
    assert isinstance(result["loc"], int) and result["loc"] > 0
    # context comes from the whitespace-normalized (lowercased) text
    assert "dbn raises" in result["context"]  # surrounding passage included


def test_verify_quote_empty_and_too_short():
    assert verify_quote("anything", "")["found"] is False
    assert verify_quote("anything", "ab")["found"] is False


def test_verify_claims_annotates_each_claim():
    claims = [
        {"claim": "win rate improves", "quote": "44.2% without beliefs and 68.8% with dynamic beliefs"},
        {"claim": "imaginary result", "quote": "this sentence does not exist in the paper"},
    ]
    verify_claims(FIXTURE_PAPER_TEXT, claims)
    assert claims[0]["quote_verified"] is True
    assert claims[1]["quote_verified"] is False
    assert claims[0]["quote_verification"]  # reason string present
    assert "not found" in claims[1]["quote_verification"]
    # location + context annotations for the dashboard's Verification view
    assert isinstance(claims[0]["quote_loc"], int)
    assert claims[0]["quote_context"]
    assert claims[1]["quote_loc"] is None


# ---------------------------------------------------------------------------
# ReaderAgent annotates quotes onto its artifact (agent-level)
# ---------------------------------------------------------------------------

def test_reader_annotates_quote_verification_on_artifact(tmp_path: Path):
    board = make_board(tmp_path)
    (tmp_path / "full_text.txt").write_text(FIXTURE_PAPER_TEXT, encoding="utf-8")
    board.set_artifact("full_text", str(tmp_path / "full_text.txt"), "")

    llm = FakeLLM(
        {
            "Reader": [
                text_response(
                    json.dumps(
                        {
                            "title": "t",
                            "summary": "s",
                            "sections": [],
                            "claims": [
                                {
                                    "claim": "DBN raises the win rate",
                                    "section": "Experiments",
                                    "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
                                    "evidence": "abstract",
                                },
                                {
                                    "claim": "hallucinated number",
                                    "section": "Experiments",
                                    "quote": "the win rate reaches 99.9% with zero variance",
                                    "evidence": "made up",
                                },
                            ],
                            "data_points": [],
                            "method_summary": "m",
                            "open_questions": [],
                        }
                    )
                )
            ]
        }
    )
    agent = ReaderAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    output = agent.run(board)

    # output claims annotated in memory
    assert output["claims"][0]["quote_verified"] is True
    assert output["claims"][1]["quote_verified"] is False
    # and persisted onto the artifact the Critic/Synthesizer consume
    artifact = json.loads(Path(board.artifacts["reader_output"]["path"]).read_text(encoding="utf-8"))
    assert artifact["claims"][0]["quote_verified"] is True
    assert artifact["claims"][1]["quote_verified"] is False


# ---------------------------------------------------------------------------
# CriticAgent downgrades quotes the deterministic check rejected
# ---------------------------------------------------------------------------

def test_critic_downgrades_unverified_quotes(tmp_path: Path):
    board = make_board(tmp_path)
    (tmp_path / "full_text.txt").write_text(FIXTURE_PAPER_TEXT, encoding="utf-8")
    board.set_artifact("full_text", str(tmp_path / "full_text.txt"), "")
    claims = [
        {
            "claim": "DBN raises the win rate",
            "section": "Experiments",
            "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
            "quote_verified": True,
        },
        {
            "claim": "hallucinated number",
            "section": "Experiments",
            "quote": "the win rate reaches 99.9%",
            "quote_verified": False,
        },
    ]
    (tmp_path / "reader_output.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")
    board.set_artifact("reader_output", str(tmp_path / "reader_output.json"), "")

    # the LLM says "supported" for both - code must correct the second one
    llm = FakeLLM(
        {
            "Critic": [
                text_response(
                    json.dumps(
                        {
                            "verdicts": [
                                {"claim": "DBN raises the win rate", "verdict": "supported", "evidence_quote": "q", "note": ""},
                                {"claim": "hallucinated number", "verdict": "supported", "evidence_quote": "q", "note": "llm note"},
                            ],
                            "limitations": [],
                            "overall_assessment": "ok",
                        }
                    )
                )
            ]
        }
    )
    agent = CriticAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    output = agent.run(board)

    verdicts = {v["claim"]: v for v in output["verdicts"]}
    assert verdicts["DBN raises the win rate"]["verdict"] == "supported"
    assert verdicts["hallucinated number"]["verdict"] == "unverified"
    assert "deterministic" in verdicts["hallucinated number"]["note"]
    assert "llm note" in verdicts["hallucinated number"]["note"]  # LLM note preserved


def test_critic_downgrade_survives_claim_rephrasing(tmp_path: Path):
    """Verdict->claim alignment must use claim_index, not exact wording.

    Two independent LLM calls (Reader and Critic) phrase the same claim
    differently; matching by text alone silently skips the downgrade.
    """
    board = make_board(tmp_path)
    (tmp_path / "full_text.txt").write_text(FIXTURE_PAPER_TEXT, encoding="utf-8")
    board.set_artifact("full_text", str(tmp_path / "full_text.txt"), "")
    claims = [
        {
            "claim": "DBN raises the win rate",
            "section": "Experiments",
            "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
            "quote_verified": True,
        },
        {
            "claim": "hallucinated number",
            "section": "Experiments",
            "quote": "the win rate reaches 99.9%",
            "quote_verified": False,
        },
    ]
    (tmp_path / "reader_output.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")
    board.set_artifact("reader_output", str(tmp_path / "reader_output.json"), "")

    llm = FakeLLM(
        {
            "Critic": [
                text_response(
                    json.dumps(
                        {
                            "verdicts": [
                                {"claim_index": 0, "claim": "completely rephrased wording A",
                                 "verdict": "supported", "evidence_quote": "q", "note": ""},
                                {"claim_index": 1, "claim": "completely rephrased wording B",
                                 "verdict": "supported", "evidence_quote": "q", "note": "llm note"},
                            ],
                            "limitations": [],
                            "overall_assessment": "ok",
                        }
                    )
                )
            ]
        }
    )
    agent = CriticAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    output = agent.run(board)

    # claim_index wins over the (mismatched) wording
    assert output["verdicts"][0]["verdict"] == "supported"
    assert output["verdicts"][1]["verdict"] == "unverified"
    assert "llm note" in output["verdicts"][1]["note"]
