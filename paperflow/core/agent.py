"""Agent base class.

An agent is a self-contained worker with:
  * an identity (name / title / description) for the board and the prompts,
  * a system prompt that defines its role and its output contract,
  * a set of tools it may call (from the shared registry),
  * a ``run(board)`` method that drives the LLM tool loop and persists an
    output artifact onto the task board.

Subclasses implement :meth:`system_prompt`, :meth:`user_context`,
:meth:`tools` and usually override :meth:`parse_output` / :meth:`summarize`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paperflow.config import Settings
from paperflow.core.jsonutil import extract_json, json_dumps
from paperflow.core.llm import LLMClient
from paperflow.core.tools import ToolRegistry


class Agent:
    """Base class for every PaperFlow agent."""

    name: str = "agent"
    title: str = "Agent"
    description: str = ""
    #: agents whose `done` output this agent depends on
    requires: list[str] = []
    #: when True, a failure aborts the whole pipeline
    critical: bool = True
    #: per-agent cap on tool-call rounds (None = use the global setting).
    #: Tools that loop per claim (e.g. the Critic's search_text) may need more.
    tool_rounds_override: int | None = None

    def __init__(self, registry: ToolRegistry, llm: LLMClient, settings: Settings | None = None):
        self.registry = registry
        self.llm = llm
        self.settings = settings or Settings()

    def max_tool_rounds(self) -> int:
        return self.tool_rounds_override if self.tool_rounds_override is not None else self.settings.max_tool_rounds

    # -- to be implemented by subclasses ----------------------------------
    def system_prompt(self) -> str:
        """Role definition + output contract. Must mention the role title."""
        raise NotImplementedError

    def user_context(self, board: Any) -> str:
        """What this agent needs to know about the current run."""
        raise NotImplementedError

    def tools(self) -> list[dict]:
        """OpenAI schemas of the tools this agent may call."""
        return []

    # -- optional overrides ------------------------------------------------
    def parse_output(self, content: str) -> Any:
        """Interpret the model's final text (default: extract embedded JSON)."""
        return extract_json(content)

    def summarize(self, output: Any) -> str:
        """Short human-readable summary stored on the board."""
        text = json_dumps(output)
        return text[:200] + ("..." if len(text) > 200 else "")

    # -- lifecycle ----------------------------------------------------------
    def run(self, board: Any) -> Any:
        """Execute this agent: LLM tool loop -> parse -> persist artifact."""
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.user_context(board)},
        ]
        final_message, _conversation = self.llm.solve(
            messages,
            tools=self.tools(),
            registry=self.registry,
            max_tool_rounds=self.max_tool_rounds(),
            max_tokens=self.settings.max_output_tokens,
            temperature=self.settings.temperature,
        )
        content = final_message.get("content") or ""
        if not content.strip():
            raise ValueError(f"agent {self.name} returned an empty answer")
        output = self.parse_output(content)
        self.save_output(board, output)
        self.after_run(board, output)
        return output

    def after_run(self, board: Any, output: Any) -> None:
        """Optional post-processing hook (e.g. materializing artifacts)."""
        return None

    def save_output(self, board: Any, output: Any) -> Path:
        """Persist this agent's output as a JSON artifact on the board."""
        artifacts_dir = board.artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / f"{self.name}_output.json"
        path.write_text(json_dumps(output), encoding="utf-8")
        board.set_artifact(f"{self.name}_output", str(path), self.description)
        return path
