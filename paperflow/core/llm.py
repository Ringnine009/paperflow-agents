"""LLM client for DeepSeek (OpenAI-compatible) with an explicit tool loop.

This module deliberately speaks the raw chat/completions protocol instead of
wrapping an SDK, so the function-calling loop is visible and testable:

1. send the conversation plus tool schemas
2. if the model asks for a tool call, execute it via the ToolRegistry and
   append the result as a ``tool`` message
3. repeat until the model answers with plain text (or the round budget runs
   out, which raises :class:`LLMError`)

The network layer is isolated in :meth:`_post` so tests can substitute a
scripted fake without touching the loop logic.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from paperflow.config import Settings


class LLMError(Exception):
    """Raised when the LLM API fails or the tool loop does not terminate."""


class LLMClient:
    """A minimal DeepSeek chat client with function-calling support."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout: int = 90,
        max_retries: int = 2,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = session or requests.Session()

    # -- public API -------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: Any = None,
    ) -> dict:
        """One request; returns the assistant message dict."""
        payload: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(payload)
                return response["choices"][0]["message"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))  # simple backoff
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}")

    def solve(
        self,
        messages: list[dict],
        tools: list[dict],
        registry: Any,
        max_tool_rounds: int = 8,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[dict, list[dict]]:
        """Run the full tool loop; returns ``(final_message, conversation)``."""
        conversation = [dict(m) for m in messages]
        for _ in range(max_tool_rounds + 1):
            message = self.chat(
                conversation, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
            conversation.append(message)
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return message, conversation
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = registry.call_safe(name, **arguments)
                conversation.append(
                    {"role": "tool", "tool_call_id": call.get("id"), "content": result}
                )
        raise LLMError(f"tool loop exceeded {max_tool_rounds} rounds without a final answer")

    # -- network layer (overridden in tests) ------------------------------
    def _post(self, payload: dict) -> dict:
        response = self.http.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()

    # -- factories --------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        return cls(
            api_key=settings.require_api_key(),
            base_url=settings.base_url,
            model=settings.model,
            timeout=settings.http_timeout,
        )
