"""Abstract-only mode: an unchecked claim must never look verified (P0).

When no full text can be obtained (a DOI without an open-access PDF, a
paywalled landing page), the Reader used to return *before* annotating
anything - so ``quote_verified`` was simply **missing**, and the Critic's
``claim.get("quote_verified") is False`` never fired. Result: no downgrade,
no log line, no marker, and a report that presents unsupported claims as
SUPPORTED. "No answer" must never be read as "fine".

The README promised this behaviour before it existed:

    Abstract-only mode: ... the Critic marks claims *unverifiable*.

These tests pin the promise: the check always records an explicit status,
cannot run -> ``unverifiable``, ran and failed -> ``unverified``.
"""

from __future__ import annotations

import json
from pathlib import Path

from paperflow.agents.critic import CriticAgent
from paperflow.agents.reader import ReaderAgent
from paperflow.agents.synthesizer import SynthesizerAgent
from paperflow.config import Settings
from paperflow.core.board import TaskBoard
from paperflow.ingest import parse_entry
from paperflow.tools import build_default_registry
from tests.helpers import FIXTURE_PAPER_TEXT, FakeLLM, text_response

SETTINGS = Settings(deepseek_api_key="sk-test")

READER_JSON = {
    "title": "A Paywalled Paper",
    "summary": "Abstract-only review.",
    "sections": [],
    "claims": [
        {
            "claim": "The method improves accuracy by 12 points",
            "section": "Abstract",
            "quote": "improves accuracy by 12 points over the baseline",
            "evidence": "abstract",
        },
        {
            "claim": "The model uses a transformer encoder",
            "section": "Abstract",
            "quote": "a transformer encoder with eight heads",
            "evidence": "abstract",
        },
    ],
    "data_points": [],
    "method_summary": "m",
    "open_questions": [],
}


def make_board(tmp_path: Path, with_full_text: bool = False) -> TaskBoard:
    board = TaskBoard.create(tmp_path / "run" / "board.json", parse_entry("https://doi.org/10.1234/paywalled"))
    if with_full_text:
        path = tmp_path / "full_text.txt"
        path.write_text(FIXTURE_PAPER_TEXT, encoding="utf-8")
        board.set_artifact("full_text", str(path), "")
    return board


def reader_llm() -> FakeLLM:
    return FakeLLM({"Reader": [text_response(json.dumps(READER_JSON))]})


def run_reader(board: TaskBoard, tmp_path: Path) -> dict:
    agent = ReaderAgent(build_default_registry(SETTINGS, tmp_path / "cache"), reader_llm(), SETTINGS)
    agent.run(board)
    return json.loads(Path(board.artifacts["reader_output"]["path"]).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the Reader must always record a status
# ---------------------------------------------------------------------------

def test_reader_marks_claims_unverifiable_without_full_text(tmp_path: Path):
    board = make_board(tmp_path)
    artifact = run_reader(board, tmp_path)
    for claim in artifact["claims"]:
        assert "quote_verified" in claim, "no deterministic annotation was recorded at all"
        assert claim["quote_verified"] is False
        assert claim["quote_status"] == "unverifiable"
        assert "no full text" in claim["quote_verification"]
        assert claim["quote_loc"] is None


def test_reader_logs_the_unverifiable_count(tmp_path: Path):
    board = make_board(tmp_path)
    run_reader(board, tmp_path)
    messages = [entry["message"] for entry in board.log]
    assert any("unverifiable" in m for m in messages), messages


def test_reader_still_reports_verified_when_the_text_is_present(tmp_path: Path):
    """The two states stay distinct: found is not the same as uncheckable."""
    board = make_board(tmp_path, with_full_text=True)
    data = dict(READER_JSON)
    data["claims"] = [
        {
            "claim": "win rate improves",
            "section": "Experiments",
            "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
            "evidence": "abstract",
        }
    ]
    llm = FakeLLM({"Reader": [text_response(json.dumps(data))]})
    ReaderAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS).run(board)
    artifact = json.loads(Path(board.artifacts["reader_output"]["path"]).read_text(encoding="utf-8"))
    assert artifact["claims"][0]["quote_status"] == "verified"
    assert artifact["claims"][0]["quote_verified"] is True


# ---------------------------------------------------------------------------
# the verdict must follow the status
# ---------------------------------------------------------------------------

def test_critic_calls_uncheckable_claims_unverifiable(tmp_path: Path):
    board = make_board(tmp_path)
    artifact = run_reader(board, tmp_path)
    assert artifact["claims"][0]["quote_status"] == "unverifiable"  # precondition

    llm = FakeLLM(
        {
            "Critic": [
                text_response(
                    json.dumps(
                        {
                            "verdicts": [
                                {"claim_index": 0, "claim": "x", "verdict": "supported", "note": "abstract sounds fine"},
                                {"claim_index": 1, "claim": "y", "verdict": "supported", "note": ""},
                            ],
                            "limitations": [],
                            "overall_assessment": "ok",
                        }
                    )
                )
            ]
        }
    )
    output = CriticAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS).run(board)
    assert [v["verdict"] for v in output["verdicts"]] == ["unverifiable", "unverifiable"]
    assert "deterministic" in output["verdicts"][0]["note"]
    assert any("unverifiable" in entry["message"] for entry in board.log)


def test_critic_never_assumes_supported_without_an_annotation(tmp_path: Path):
    """Legacy/partial artifacts: a claim no check ever touched is not verified."""
    board = make_board(tmp_path)
    (tmp_path / "reader_output.json").write_text(
        json.dumps({"claims": [{"claim": "unchecked claim", "quote": "some quote", "section": "s"}]}),
        encoding="utf-8",
    )
    board.set_artifact("reader_output", str(tmp_path / "reader_output.json"), "")
    llm = FakeLLM(
        {
            "Critic": [
                text_response(
                    json.dumps(
                        {
                            "verdicts": [{"claim_index": 0, "claim": "unchecked claim", "verdict": "supported", "note": ""}],
                            "limitations": [],
                            "overall_assessment": "ok",
                        }
                    )
                )
            ]
        }
    )
    output = CriticAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS).run(board)
    assert output["verdicts"][0]["verdict"] != "supported"
    assert "deterministic" in output["verdicts"][0]["note"]


# ---------------------------------------------------------------------------
# and it must reach the report
# ---------------------------------------------------------------------------

def test_abstract_mode_status_reaches_the_report(tmp_path: Path):
    board = make_board(tmp_path)
    run_reader(board, tmp_path)
    critic_llm = FakeLLM(
        {
            "Critic": [
                text_response(
                    json.dumps(
                        {
                            "verdicts": [
                                {"claim_index": 0, "claim": "x", "verdict": "supported", "note": ""},
                                {"claim_index": 1, "claim": "y", "verdict": "supported", "note": ""},
                            ],
                            "limitations": [],
                            "overall_assessment": "ok",
                        }
                    )
                )
            ]
        }
    )
    CriticAgent(build_default_registry(SETTINGS, tmp_path / "cache"), critic_llm, SETTINGS).run(board)

    report = (
        "# A Paywalled Paper\n\n## Key Claims & Evidence\n\n"
        "- **The method improves accuracy by 12 points.** **[supported]** — from the abstract.\n"
        "- **The model uses a transformer encoder.** **[supported]** — from the abstract.\n"
    )
    synth = FakeLLM({"Synthesizer": [text_response(report)]})
    SynthesizerAgent(build_default_registry(SETTINGS, tmp_path / "cache"), synth, SETTINGS).run(board)
    final = Path(board.report["path"]).read_text(encoding="utf-8")

    assert "[unverifiable]" in final
    assert "no full text" in final
    assert "[supported]" not in final.split("## Verification Ledger")[0].split("## Key Claims")[1]
