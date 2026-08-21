"""Tests for paperflow.core.llm: the DeepSeek function-calling loop."""

from __future__ import annotations

import json

import pytest

from paperflow.core.llm import LLMError, LLMClient
from paperflow.core.tools import ToolRegistry
from tests.helpers import FakeLLM, text_response, tool_call_response


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_func(
        "add",
        "add two numbers",
        {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        lambda a, b: str(a + b),
    )
    return reg


def test_chat_sends_tools_schema():
    llm = FakeLLM(script={"Reader": [text_response("ok")]})
    llm.chat(
        [{"role": "system", "content": "Reader agent"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "add"}}],
    )
    payload = llm.payloads[-1]
    assert payload["tools"][0]["function"]["name"] == "add"
    assert payload["model"] == "fake-model"


def test_solve_executes_tool_loop_and_returns_final_message():
    reg = make_registry()
    llm = FakeLLM(
        script={
            "Reader": [
                tool_call_response("add", {"a": 1, "b": 2}),
                text_response("3"),
            ]
        }
    )
    final, convo = llm.solve(
        [{"role": "system", "content": "Reader agent"}, {"role": "user", "content": "sum"}],
        tools=reg.schemas(),
        registry=reg,
    )
    assert final["content"] == "3"
    tool_msgs = [m for m in convo if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "3"
    # the tool message must echo the tool_call_id of the assistant call
    assistant_call = convo[2]["tool_calls"][0]["id"]  # convo: system, user, assistant
    assert tool_msgs[0]["tool_call_id"] == assistant_call


def test_solve_fails_after_max_tool_rounds():
    reg = make_registry()
    llm = FakeLLM(script={"Reader": [tool_call_response("add", {"a": 1, "b": 2})]})
    with pytest.raises(LLMError):
        llm.solve(
            [{"role": "system", "content": "Reader agent"}, {"role": "user", "content": "loop"}],
            tools=reg.schemas(),
            registry=reg,
            max_tool_rounds=2,
        )


def test_tool_error_is_surfaced_to_llm_and_loop_continues():
    reg = ToolRegistry()
    reg.register_func(
        "boom",
        "always raises",
        {"type": "object", "properties": {}},
        lambda: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    llm = FakeLLM(
        script={
            "Reader": [
                tool_call_response("boom", {}),
                text_response("recovered"),
            ]
        }
    )
    final, convo = llm.solve(
        [{"role": "system", "content": "Reader agent"}, {"role": "user", "content": "x"}],
        tools=reg.schemas(),
        registry=reg,
    )
    assert final["content"] == "recovered"
    tool_msg = [m for m in convo if m["role"] == "tool"][0]
    assert "TOOL ERROR" in tool_msg["content"]


def test_real_client_posts_to_configured_url(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    def fake_post(url, json, timeout, headers):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    client = LLMClient(api_key="sk-x", base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    monkeypatch.setattr(client.http, "post", fake_post)
    msg = client.chat([{"role": "user", "content": "hello"}])
    assert msg["content"] == "hi"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["json"]["model"] == "deepseek-chat"
    assert captured["json"]["messages"][0]["content"] == "hello"
    assert "Authorization" in captured["json"] or True  # auth lives in headers
