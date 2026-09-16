"""Demo: what the fixed report looks like (used for docs evidence).

Runs the Synthesizer agent offline against a board whose second claim was
downgraded by the deterministic check, and prints the resulting report.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow.agents.synthesizer import SynthesizerAgent  # noqa: E402
from paperflow.config import Settings  # noqa: E402
from paperflow.core.board import TaskBoard  # noqa: E402
from paperflow.ingest import parse_entry  # noqa: E402
from paperflow.tools import build_default_registry  # noqa: E402
from tests.helpers import FakeLLM, text_response  # noqa: E402

SETTINGS = Settings(deepseek_api_key="sk-test")

CANNED = """\
# On Social Deduction with Dynamic Belief Networks

## Overview

A study of DBNs.

## Key Claims & Evidence

- **The DBN improves the villager win rate to 68.8%.** **[supported]** — quote found in the abstract.
- **The model attains a 99.9% win rate with zero variance.** **[supported]** — reported in Table 2.

## Limitations

Narrow evaluation.
"""

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    board = TaskBoard.create(tmp_path / "run" / "board.json", parse_entry("https://arxiv.org/abs/2601.12345"))
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    claims = [
        {"claim": "The DBN raises the villager win rate to 68.8%", "quote": "44.2% without beliefs and 68.8% with dynamic beliefs",
         "quote_verified": True, "quote_verification": "quote found verbatim (whitespace-normalized)"},
        {"claim": "The model reaches a 99.9% win rate with zero variance", "quote": "the win rate reaches 99.9% with zero variance",
         "quote_verified": False, "quote_verification": "quote not found in the paper text"},
    ]
    (artifacts / "reader_output.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")
    board.set_artifact("reader_output", str(artifacts / "reader_output.json"), "")

    llm = FakeLLM({"Synthesizer": [text_response(CANNED)]})
    agent = SynthesizerAgent(build_default_registry(SETTINGS, tmp_path / "cache"), llm, SETTINGS)
    agent.run(board)
    print(Path(board.report["path"]).read_text(encoding="utf-8"))
    print("--- board log ---")
    for entry in board.log:
        print(entry["ts"], entry["level"], entry["agent"], entry["message"][:160])
