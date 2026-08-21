"""Pipeline: one-call entry point that runs the whole agent team.

``Pipeline.run(entry)``:

1. normalizes the entry (arXiv URL / DOI / PDF path / URL / title)
2. creates a run directory with a fresh JSON task board
3. builds the default tool registry (network tools + local text tools)
4. assembles the four agents and lets the Orchestrator schedule them
5. returns the paths of the persisted board and the final report

The `llm` and `tool_stubs` parameters are test/offline seams: pass a fake
LLM to run without network, and stub specific tools to avoid external calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from paperflow.agents import CriticAgent, ReaderAgent, ResearcherAgent, SynthesizerAgent
from paperflow.config import Settings, get_settings
from paperflow.core.board import TaskBoard, make_run_id
from paperflow.core.llm import LLMClient
from paperflow.core.orchestrator import Orchestrator
from paperflow.core.tools import ToolRegistry
from paperflow.ingest import parse_entry
from paperflow.tools import build_default_registry

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs a full paper-review pipeline for one entry."""

    def __init__(
        self,
        settings: Settings | None = None,
        out_dir: str | Path | None = None,
        llm: LLMClient | None = None,
        tool_stubs: dict[str, Callable] | None = None,
    ):
        self.settings = settings or get_settings()
        self.out_dir = Path(out_dir) if out_dir else Path.cwd() / "outputs"
        self.cache_dir = self.out_dir / "_cache"
        self.llm = llm or LLMClient.from_settings(self.settings)
        self.tool_stubs = tool_stubs or {}
        #: registry is built up-front so callers can stub tools before `run`
        self.registry = build_default_registry(self.settings, self.cache_dir)
        for name, func in self.tool_stubs.items():
            self.registry.get(name).func = func

    # -- execution --------------------------------------------------------
    def run(self, entry: str, pdf_override: str | None = None) -> dict:
        """Run the team on `entry`; returns result paths + status."""
        spec = parse_entry(entry, pdf_override=pdf_override)
        run_id = make_run_id()
        run_dir = self.out_dir / run_id
        board = TaskBoard.create(run_dir / "board.json", spec)
        board.save()

        agents = [
            ResearcherAgent(self.registry, self.llm, self.settings),
            ReaderAgent(self.registry, self.llm, self.settings),
            CriticAgent(self.registry, self.llm, self.settings),
            SynthesizerAgent(self.registry, self.llm, self.settings),
        ]

        logger.info("paperflow run %s started: %s (%s)", run_id, spec.raw, spec.kind)
        orchestrator = Orchestrator(agents, board)
        orchestrator.run()

        logger.info("paperflow run %s finished: %s", run_id, board.status)
        return {
            "run_id": run_id,
            "status": board.status,
            "board": str(board.path),
            "report": str(board.report_path()) if board.report_path() else None,
            "input": spec.to_dict(),
        }
