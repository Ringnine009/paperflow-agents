"""Orchestrator: schedules the agent team against the task board.

It is deliberately a dumb, deterministic scheduler - the intelligence lives
in the agents, the orchestration is explicit:

1. run agents in declaration order
2. before an agent starts, verify its declared dependencies are done
3. a failure of a *critical* agent aborts the pipeline; a failure of an
   *optional* agent is logged and the team moves on
4. the board is persisted after every transition so the web dashboard can
   follow progress live, and every run is inspectable afterwards
"""

from __future__ import annotations

from typing import Any

from paperflow.core.agent import Agent
from paperflow.core.board import TaskBoard


class OrchestratorError(Exception):
    """Raised when a critical agent fails and the pipeline must abort."""


class Orchestrator:
    def __init__(self, agents: list[Agent], board: TaskBoard):
        self.agents = agents
        self.board = board

    # -- scheduling -------------------------------------------------------
    def run(self) -> TaskBoard:
        """Run the whole team. Returns the (persisted) board."""
        board = self.board
        board.set_status("running")
        board.add_log(f"orchestrator: pipeline started with {len(self.agents)} agents")

        for agent in self.agents:
            unmet = self._unmet_dependencies(agent)
            if unmet:
                if agent.critical:
                    message = f"agent '{agent.name}' skipped: missing dependencies {unmet}"
                    board.record_agent_end(agent.name, "failed", error=message)
                    board.set_status("failed", current_stage=agent.name)
                    board.save()
                    raise OrchestratorError(message)
                board.add_log(
                    f"agent '{agent.name}' skipped (optional): missing dependencies {unmet}",
                    level="warning",
                )
                continue

            board.record_agent_start(agent.name)
            board.save()
            try:
                output = agent.run(board)
                board.record_agent_end(agent.name, "done", summary=agent.summarize(output))
            except Exception as exc:  # noqa: BLE001 - any agent failure is a pipeline event
                board.record_agent_end(agent.name, "failed", error=f"{type(exc).__name__}: {exc}")
                if agent.critical:
                    board.set_status("failed", current_stage=agent.name)
                    board.save()
                    raise OrchestratorError(f"critical agent '{agent.name}' failed: {exc}") from exc
                board.add_log(
                    f"optional agent '{agent.name}' failed; continuing ({exc})", level="warning"
                )
            board.save()

        board.set_status("done")
        board.add_log("orchestrator: pipeline complete")
        board.save()
        return board

    # -- helpers ----------------------------------------------------------
    def _unmet_dependencies(self, agent: Agent) -> list[str]:
        """Dependencies that have not finished successfully yet."""
        unmet: list[str] = []
        for required in agent.requires:
            entry = self.board.agents.get(required)
            if entry is None or entry.get("status") != "done":
                unmet.append(required)
        return unmet
