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
from paperflow.core.pricing import PRICING_PEAK_CNY, Pricing, cost_cny


class LLMError(Exception):
    """Raised when the LLM API fails or the tool loop does not terminate."""


class BudgetExceeded(LLMError):
    """Raised *before* sending a request that would break the spend cap.

    The cap is checked on the worst case of the request about to be sent
    (its input tokens + its `max_tokens` of output), so a run can never
    discover an overrun only after paying for it.
    """


#: chars per token, used only to bound a request *before* it is sent (the real
#: count comes back in `usage`). 2.5 is deliberately pessimistic for English:
#: over-estimating refuses a borderline call instead of overspending.
_CHARS_PER_TOKEN_ESTIMATE = 2.5


class UsageLedger:
    """Accumulated token usage of one client, in the buckets DeepSeek bills.

    ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens`` are what the
    invoice is actually computed from, so they are kept apart instead of
    collapsing everything into ``prompt_tokens``.
    """

    FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens")

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0

    def record(self, usage: dict | None) -> None:
        """Accumulate one response's `usage` block.

        A missing cache split means the endpoint did not report one; the whole
        prompt is then counted as a cache miss (the expensive bucket), so an
        unknown is never billed as a discount.
        """
        usage = usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        if hit is None and miss is None:
            hit, miss = 0, prompt
        else:
            hit, miss = int(hit or 0), int(miss or 0)
            # trust the total over the split if an endpoint reports both badly
            if hit + miss != prompt and prompt:
                miss = max(0, prompt - hit)
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cache_hit_tokens += hit
        self.cache_miss_tokens += miss

    def totals(self) -> dict:
        return {field: getattr(self, field) for field in self.FIELDS}

    def cost_cny(self, pricing: Pricing = PRICING_PEAK_CNY) -> float:
        return cost_cny(self.totals(), pricing)


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
        budget_cny: float | None = None,
        pricing: Pricing = PRICING_PEAK_CNY,
        thinking: str | None = None,
        record_payloads: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = session or requests.Session()
        #: hard spend cap in CNY; None = uncapped (the library default, so
        #: nothing changes for existing callers)
        self.budget_cny = budget_cny
        self.pricing = pricing
        #: "disabled" pins non-thinking mode (the project never used a
        #: reasoning model, and thinking output is billed as output tokens)
        self.thinking = thinking
        self.usage = UsageLedger()
        #: set True by an experiment that wants the exact prompts on record;
        #: payload bodies are truncated to keep the ledger small
        self.record_payloads = record_payloads
        self.payloads: list[dict] = []

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
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}

        self._assert_within_budget(payload, max_tokens)
        if self.record_payloads:
            self.payloads.append(_truncated_payload(payload))

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(payload)
                self.usage.record(response.get("usage"))
                return response["choices"][0]["message"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))  # simple backoff
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}")

    # -- accounting -------------------------------------------------------
    @property
    def calls(self) -> int:
        """How many requests this client has issued (paid calls, not attempts)."""
        return self.usage.calls

    def usage_totals(self) -> dict:
        """Cumulative tokens/calls of this client (real `usage`, not an estimate)."""
        return self.usage.totals()

    def cost(self) -> float:
        """Cumulative spend in CNY at this client's pricing."""
        return self.usage.cost_cny(self.pricing)

    def _assert_within_budget(self, payload: dict, max_tokens: int | None) -> None:
        """Refuse a request whose worst case would exceed `budget_cny`.

        The prompt is fully known before sending, and output is capped by
        `max_tokens`, so the maximum this call can cost is computable now.
        """
        if self.budget_cny is None:
            return
        worst_input = _estimate_prompt_tokens(payload)
        worst_output = int(max_tokens or 0)
        worst_case = cost_cny(
            {"cache_miss_tokens": worst_input, "completion_tokens": worst_output}, self.pricing
        )
        if self.cost() + worst_case > self.budget_cny:
            raise BudgetExceeded(
                f"budget guard: spent {self.cost():.6f} CNY, this call could add up to "
                f"{worst_case:.6f} CNY, cap is {self.budget_cny:.6f} CNY - request not sent"
            )

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
    def from_settings(cls, settings: Settings, **kwargs: Any) -> "LLMClient":
        return cls(
            api_key=settings.require_api_key(),
            base_url=settings.base_url,
            model=settings.model,
            timeout=settings.http_timeout,
            **kwargs,
        )


def _estimate_prompt_tokens(payload: dict) -> int:
    """Upper-bound estimate of a request's input tokens, before sending it.

    Only used by the budget guard; the authoritative count always comes from
    the API's `usage` block.
    """
    chars = 0
    for message in payload.get("messages", []):
        content = message.get("content")
        chars += len(content) if isinstance(content, str) else len(json.dumps(content, default=str))
        for call in message.get("tool_calls") or []:
            chars += len(json.dumps(call, default=str))
    if payload.get("tools"):
        chars += len(json.dumps(payload["tools"], default=str))
    return int(chars / _CHARS_PER_TOKEN_ESTIMATE) + 8


def _truncated_payload(payload: dict, limit: int = 400) -> dict:
    """A payload copy with every message body clipped, for the run ledger."""
    kept = {k: v for k, v in payload.items() if k != "messages"}
    kept["messages"] = [
        {
            **{k: v for k, v in message.items() if k != "content"},
            "content": (message.get("content") or "")[:limit]
            if isinstance(message.get("content"), str)
            else message.get("content"),
        }
        for message in payload.get("messages", [])
    ]
    return kept
