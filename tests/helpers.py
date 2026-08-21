"""Shared offline-test helpers: a scripted FakeLLM plus response builders.

These helpers let the full agent pipeline run without any network or real
LLM: every LLM answer is a deterministic script keyed by agent role and by
how many tool results have already been fed back.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from paperflow.core.llm import LLMClient

ROLE_KEYWORDS = ["Researcher", "Reader", "Critic", "Synthesizer"]


def detect_role(system_content: str) -> str:
    """Return the agent role whose title appears first in the system prompt.

    Agents' prompts begin with "You are the <Role> ..." but may mention
    other roles later ("integrate the Researcher's metadata"), so we pick
    the earliest occurrence instead of a fixed priority order.
    """
    hits = [(system_content.find(keyword), keyword) for keyword in ROLE_KEYWORDS if keyword in system_content]
    if not hits:
        raise AssertionError(f"cannot detect agent role from system prompt: {system_content[:80]!r}")
    return min(hits)[1]


def tool_call_response(name: str, args: dict) -> dict:
    """An assistant message that asks to call a tool."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}_{id(args) & 0xFFFF}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def text_response(content: str) -> dict:
    """A plain assistant text message."""
    return {"role": "assistant", "content": content}


class FakeLLM(LLMClient):
    """LLMClient whose `_post` is scripted. Real solve()/loop logic is used.

    `script` maps role name -> list of response makers (callables returning
    an assistant message dict, or plain dicts). The list index is the number
    of tool-result messages already present, so the pipeline determinism of
    "tool call -> tool result -> next response" is preserved.
    """

    def __init__(self, script: dict[str, list[Any]] | None = None):
        super().__init__(api_key="test-key", base_url="http://fake.local", model="fake-model")
        self.script: dict[str, list[Any]] = script or {}
        self.payloads: list[dict] = []

    def _post(self, payload: dict) -> dict:
        self.payloads.append(payload)
        messages = payload["messages"]
        system = next(m["content"] for m in messages if m["role"] == "system")
        role = detect_role(system)
        n_tool_results = sum(1 for m in messages if m["role"] == "tool")
        script = self.script.get(role)
        if not script:
            raise AssertionError(f"FakeLLM has no script for role {role!r}")
        maker = script[min(n_tool_results, len(script) - 1)]
        # callables receive the current conversation so scripts can echo
        # real tool results (e.g. an extracted-text path) into their answers
        message = maker(messages) if callable(maker) else maker
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def tool_result_text_path(messages: list[dict]) -> str:
    """Pull the TEXT_PATH line out of the last tool message."""
    for message in reversed(messages):
        if message["role"] == "tool":
            for line in message["content"].splitlines():
                if line.startswith("TEXT_PATH:"):
                    return line.split(":", 1)[1].strip()
    raise AssertionError("no TEXT_PATH found in tool messages")


def stub_tool(registry: Any, name: str, func: Callable) -> None:
    """Replace a registered tool's implementation with a stub (no network)."""
    spec = registry.get(name)
    spec.func = func  # ToolSpec is a mutable dataclass


# ---------------------------------------------------------------------------
# Fixture content
# ---------------------------------------------------------------------------

FIXTURE_PAPER_TEXT = """\
On Social Deduction with Dynamic Belief Networks
Abstract: We study multi-agent social deduction in the game of Werewolf.
We model players as agents holding dynamic belief networks over who is the
werewolf. Our experiments show a villager win rate of 44.2% without beliefs
and 68.8% with dynamic beliefs. Voting accuracy rises from 35.5% to 66.6%.
1. Introduction
Social deduction games require reasoning about other agents' hidden roles.
2. Method
We introduce a Dynamic Belief Network (DBN) that updates posterior
distributions after each public statement using an expectation-maximization
update. The model maintains a joint distribution over role assignments.
3. Experiments
We ran 500 games with 9 players. The DBN agent achieved villager win rate
68.8% versus 44.2% for the baseline. Brier score converges over rounds.
4. Conclusion
Dynamic belief networks improve social reasoning in hidden-role games.
"""


def fake_arxiv_search_result(**overrides: Any) -> str:
    """JSON string a stubbed arxiv_search returns."""
    item = {
        "arxiv_id": "2601.12345",
        "title": "On Social Deduction with Dynamic Belief Networks",
        "authors": ["Tongji Student", "Another Author"],
        "published": "2026-01-10",
        "abstract": "We study multi-agent social deduction in Werewolf with DBNs.",
        "pdf_url": "https://arxiv.org/pdf/2601.12345",
        "abs_url": "https://arxiv.org/abs/2601.12345",
        "doi": None,
    }
    item.update(overrides)
    return json.dumps([item])
