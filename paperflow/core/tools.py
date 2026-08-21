"""Tool registry: the mechanism that lets agents call functions.

Design notes
------------
* A ``ToolSpec`` bundles a name, an LLM-facing description, a JSON Schema
  for its parameters (OpenAI function-calling format) and a plain Python
  callable.
* The ``ToolRegistry`` is the single entry point for both sides:
  - the orchestrator asks it for OpenAI-compatible schemas (``schemas()``)
    to hand to the LLM;
  - the LLM client executes tool calls through it (``call``/``call_safe``).
* Results are always converted to strings because that is what a tool
  message in the conversation expects. Dicts/lists become compact JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from paperflow.core.jsonutil import json_dumps


class ToolError(Exception):
    """Raised when a tool is unknown or fails to execute."""


@dataclass
class ToolSpec:
    """One registered tool."""

    name: str
    description: str
    parameters: dict  # JSON Schema object
    func: Callable[..., Any]

    def to_schema(self) -> dict:
        """OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def make_schema(properties: dict, required: list[str] | None = None) -> dict:
    """Build a JSON Schema object for a tool's parameters."""
    return {"type": "object", "properties": properties, "required": required or []}


class ToolRegistry:
    """Holds every tool an agent team may call."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # -- registration -----------------------------------------------------
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ToolError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def register_func(self, name: str, description: str, parameters: dict, func: Callable[..., Any]) -> ToolSpec:
        return self.register(ToolSpec(name=name, description=description, parameters=parameters, func=func))

    # -- lookup -----------------------------------------------------------
    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        """All registered tools as OpenAI function-calling schemas."""
        return [spec.to_schema() for spec in self._tools.values()]

    # -- execution --------------------------------------------------------
    def call(self, name: str, **kwargs: Any) -> str:
        """Execute a tool and stringify its result; raise on any failure."""
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool: {name}")
        result = spec.func(**kwargs)
        return _stringify(result)

    def call_safe(self, name: str, **kwargs: Any) -> str:
        """Like :meth:`call` but converts failures into an LLM-readable string.

        The LLM sees ``TOOL ERROR: <reason>`` and can decide to retry with
        different arguments instead of crashing the whole pipeline.
        """
        try:
            return self.call(name, **kwargs)
        except Exception as exc:  # noqa: BLE001 - we want every failure visible to the LLM
            return f"TOOL ERROR: {type(exc).__name__}: {exc}"


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json_dumps(result)
    return str(result)
