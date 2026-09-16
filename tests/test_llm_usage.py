"""Real token accounting + hard budget guard for the live three-arm experiment.

The offline harness could not measure cost: `LLMClient.chat()` threw the API's
`usage` block away, and `FakeLLM` returned a stub (1/1/2). These tests pin the
behaviour the live experiment needs:

* every real response's `usage` is accumulated (including DeepSeek's
  cache-hit / cache-miss split, which is what actually gets billed),
* cumulative spend is turned into CNY with published per-1M-token prices,
* a request that would push the run past the hard budget is refused *before*
  it is sent, so an experiment cannot overspend and report afterwards.

No network: `_post` is scripted by the fakes below.
"""

from __future__ import annotations

import pytest

from paperflow.core.llm import BudgetExceeded, LLMClient, UsageLedger
from paperflow.core.pricing import PRICING_PEAK_CNY, cost_cny


class RecordingClient(LLMClient):
    """LLMClient whose HTTP layer is scripted and counts real sends."""

    def __init__(self, usages: list[dict] | None = None, **kwargs):
        super().__init__(api_key="test-key", base_url="http://fake.local", model="fake-model", **kwargs)
        self.sent = 0
        self._usages = list(usages or [])

    def _post(self, payload: dict) -> dict:
        self.sent += 1
        usage = self._usages.pop(0) if self._usages else {"prompt_tokens": 0, "completion_tokens": 0}
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": usage}


# ---------------------------------------------------------------------------
# usage accounting
# ---------------------------------------------------------------------------

def test_deepseek_usage_block_is_accumulated_verbatim():
    """DeepSeek reports cache hit/miss separately; the ledger must keep both."""
    client = RecordingClient(
        usages=[
            {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_cache_hit_tokens": 800,
                "prompt_cache_miss_tokens": 200,
            },
            {
                "prompt_tokens": 500,
                "completion_tokens": 50,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 500,
            },
        ]
    )
    for _ in range(2):
        client.chat([{"role": "user", "content": "hi"}])

    totals = client.usage_totals()
    assert client.calls == 2
    assert totals["prompt_tokens"] == 1500
    assert totals["completion_tokens"] == 250
    assert totals["cache_hit_tokens"] == 800
    assert totals["cache_miss_tokens"] == 700


def test_usage_without_cache_detail_falls_back_to_whole_prompt():
    """Older/other endpoints omit the cache split - then the prompt is all miss."""
    client = RecordingClient(usages=[{"prompt_tokens": 300, "completion_tokens": 30}])
    client.chat([{"role": "user", "content": "hi"}])
    totals = client.usage_totals()
    assert totals["cache_miss_tokens"] == 300
    assert totals["cache_hit_tokens"] == 0


def test_cost_uses_published_prices_and_reports_cny():
    """cost = hit*price_hit + miss*price_miss + completion*price_out, per 1M."""
    pricing = PRICING_PEAK_CNY
    ledger = UsageLedger()
    ledger.record(
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "prompt_cache_hit_tokens": 400_000,
            "prompt_cache_miss_tokens": 600_000,
        }
    )
    expected = (
        400_000 * pricing.cache_hit_per_m
        + 600_000 * pricing.cache_miss_per_m
        + 1_000_000 * pricing.output_per_m
    ) / 1_000_000
    assert cost_cny(ledger.totals(), pricing) == pytest.approx(expected)
    assert ledger.cost_cny(pricing) == pytest.approx(expected)


def test_calls_and_cost_are_reported_per_run_even_when_shared():
    """A single client accumulates; `snapshot()` lets a harness slice per run."""
    client = RecordingClient(
        usages=[
            {"prompt_tokens": 100, "completion_tokens": 10},
            {"prompt_tokens": 200, "completion_tokens": 20},
        ]
    )
    client.chat([{"role": "user", "content": "one"}])
    first = client.usage_totals()
    client.chat([{"role": "user", "content": "two"}])
    second = client.usage_totals()

    assert first["calls"] == 1
    assert second["calls"] == 2
    delta = {k: second[k] - first[k] for k in first}
    assert delta["prompt_tokens"] == 200
    assert delta["calls"] == 1


# ---------------------------------------------------------------------------
# budget guard
# ---------------------------------------------------------------------------

def test_budget_guard_refuses_the_call_that_would_overspend():
    """The request must never be sent once the worst case breaks the cap.

    Worst case = this call's input + its `max_tokens` output, because the
    prompt is fully known before sending.
    """
    client = RecordingClient(budget_cny=0.001)
    with pytest.raises(BudgetExceeded) as excinfo:
        client.chat([{"role": "user", "content": "x" * 4000}], max_tokens=4000)
    assert client.sent == 0, "an over-budget request must not reach the network"
    assert "budget" in str(excinfo.value).lower()


def test_budget_guard_lets_affordable_calls_through_and_trips_on_the_next():
    """Spending accumulates: the call that crosses the cap is the one refused."""
    # 1 CNY per 1M output tokens-ish is not enough to reason about; use the
    # real prices and a tiny cap: 0.0003 CNY buys ~1000 peak cache-miss tokens.
    client = RecordingClient(
        usages=[{"prompt_tokens": 900, "completion_tokens": 100}],
        budget_cny=0.0003,
    )
    client.chat([{"role": "user", "content": "small"}], max_tokens=1)
    assert client.sent == 1

    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "small again"}], max_tokens=1)
    assert client.sent == 1


def test_budget_guard_is_off_when_no_budget_is_configured():
    """Default behaviour is unchanged: no cap, no surprises for other callers."""
    client = RecordingClient()
    assert client.budget_cny is None
    client.chat([{"role": "user", "content": "x" * 100_000}], max_tokens=4000)
    assert client.sent == 1


def test_zero_budget_blocks_everything():
    """0 is a real cap (not 'unset'): the experiment can be pinned to spend nothing."""
    client = RecordingClient(budget_cny=0.0)
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "hello"}])
    assert client.sent == 0
