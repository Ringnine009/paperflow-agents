"""The deterministic verdict must reach the deliverable (P0).

The Critic downgrades a claim to "unverified" in *code* when the quote was not
found in the paper. An audit of the archived runs found that 6 of the 7 runs
with downgraded claims still ship a ``report.md`` that calls those claims
**[supported]** - ``unverified`` appears zero times. The deterministic
decision therefore only ever reached ``critic_output.json``, and the one
artifact a human reads kept the LLM's optimistic wording.

These tests pin the fix: the machine verdict is injected into the report by
code, and a post-generation check verifies the report text agrees with it.
"""

from __future__ import annotations

import json
from pathlib import Path

from paperflow.agents.synthesizer import SynthesizerAgent
from paperflow.agents.verification import check_report_consistency, claim_statuses, inject_machine_verdicts
from paperflow.config import Settings
from paperflow.core.board import TaskBoard
from paperflow.ingest import parse_entry
from paperflow.tools import build_default_registry
from tests.helpers import FakeLLM, text_response

SETTINGS = Settings(deepseek_api_key="sk-test")

CLAIMS = [
    {
        "claim": "The DBN raises the villager win rate to 68.8%",
        "section": "Experiments",
        "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
        "quote_verified": True,
        "quote_verification": "quote found verbatim (whitespace-normalized)",
    },
    {
        "claim": "The model reaches a 99.9% win rate with zero variance",
        "section": "Experiments",
        "quote": "the win rate reaches 99.9% with zero variance",
        "quote_verified": False,
        "quote_verification": "quote not found in the paper text",
    },
]

#: what the Synthesizer LLM actually produced in the archived runs: every
#: claim marked supported, the downgraded one paraphrased
CANNED_REPORT = """\
# On Social Deduction

## Overview

A study of DBNs.

## Key Claims & Evidence

- **The DBN improves the villager win rate to 68.8%.** **[supported]** — quote found in the abstract.
- **The model attains a 99.9% win rate with zero variance.** **[supported]** — reported in Table 2.

## Limitations

Narrow evaluation.
"""


def make_board(tmp_path: Path, claims=None, verdicts=None) -> TaskBoard:
    board = TaskBoard.create(tmp_path / "run" / "board.json", parse_entry("https://arxiv.org/abs/2601.12345"))
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "reader_output.json").write_text(
        json.dumps({"claims": claims if claims is not None else CLAIMS}), encoding="utf-8"
    )
    board.set_artifact("reader_output", str(artifacts / "reader_output.json"), "")
    if verdicts is not None:
        (artifacts / "critic_output.json").write_text(json.dumps({"verdicts": verdicts}), encoding="utf-8")
        board.set_artifact("critic_output", str(artifacts / "critic_output.json"), "")
    return board


def run_synthesizer(board: TaskBoard, tmp_path: Path, markdown: str = CANNED_REPORT) -> str:
    llm = FakeLLM({"Synthesizer": [text_response(markdown)]})
    agent = SynthesizerAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    agent.run(board)
    return Path(board.report["path"]).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# claim_statuses: the machine-owned view
# ---------------------------------------------------------------------------

def test_claim_statuses_are_derived_from_the_reader_annotation(tmp_path: Path):
    board = make_board(tmp_path)
    rows = claim_statuses(board)
    assert [r["status"] for r in rows] == ["verified", "unverified"]
    assert rows[1]["reason"] == "quote not found in the paper text"
    assert rows[1]["claim_index"] == 1


# ---------------------------------------------------------------------------
# the report must carry the status
# ---------------------------------------------------------------------------

def test_report_must_carry_the_deterministic_status(tmp_path: Path):
    board = make_board(tmp_path)
    report = run_synthesizer(board, tmp_path)
    assert "[unverified]" in report, "report.md hides the downgrade"
    assert "the win rate reaches 99.9% with zero variance" in report  # the machine ledger lists it
    assert check_report_consistency(board, report) == []


def test_supported_marker_is_repaired_on_a_paraphrased_bullet(tmp_path: Path):
    """The LLM paraphrases the claim; the code must still fix that bullet."""
    board = make_board(tmp_path)
    report = run_synthesizer(board, tmp_path)
    bullet = next(line for line in report.splitlines() if "99.9%" in line and line.startswith("- "))
    assert "[supported]" not in bullet
    assert "[unverified]" in bullet


def test_all_verified_report_keeps_the_llm_verdicts(tmp_path: Path):
    board = make_board(
        tmp_path,
        claims=[dict(CLAIMS[0])],
        verdicts=[{"claim_index": 0, "claim": CLAIMS[0]["claim"], "verdict": "supported"}],
    )
    report = run_synthesizer(board, tmp_path)
    assert "[unverified]" not in report
    assert "1/1" in report  # the machine summary states the verified count
    assert check_report_consistency(board, report) == []


def test_ledger_records_the_critic_verdict_that_was_overridden(tmp_path: Path):
    board = make_board(
        tmp_path,
        verdicts=[
            {"claim_index": 1, "claim": CLAIMS[1]["claim"], "verdict": "unverified", "note": "quote not found in paper text (deterministic check)"}
        ],
    )
    report = run_synthesizer(board, tmp_path)
    assert "deterministic check" in report or "downgraded" in report.lower()


# ---------------------------------------------------------------------------
# the consistency check is a real check, not a formality
# ---------------------------------------------------------------------------

def test_consistency_check_flags_a_report_that_hides_the_status(tmp_path: Path):
    board = make_board(tmp_path)
    violations = check_report_consistency(board, CANNED_REPORT)
    assert violations, "a report with no machine status must be reported as inconsistent"
    assert any("unverified" in v for v in violations)


def test_consistency_check_flags_a_bullet_that_still_claims_supported(tmp_path: Path):
    board = make_board(tmp_path)
    report, _ = inject_machine_verdicts(board, CANNED_REPORT)
    # simulate a report whose bullet was re-edited back to [supported]
    tampered = report.replace("[unverified]", "[supported]")
    violations = check_report_consistency(board, tampered)
    assert any("supported" in v for v in violations)


def test_injection_is_idempotent(tmp_path: Path):
    board = make_board(tmp_path)
    once, _ = inject_machine_verdicts(board, CANNED_REPORT)
    twice, _ = inject_machine_verdicts(board, once)
    assert once == twice


def test_injection_leaves_unrelated_sections_alone(tmp_path: Path):
    board = make_board(tmp_path)
    report, _ = inject_machine_verdicts(board, CANNED_REPORT)
    assert "Narrow evaluation." in report
    assert report.lstrip().startswith("# On Social Deduction")
