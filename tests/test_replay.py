"""Replay tests."""
from __future__ import annotations

from typing import Any

import pytest

from prompt_trace.replay import replay_span
from prompt_trace.span import Span, SpanKind


class _Recorder:
    def __init__(self):
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return {"replayed": True}


class TestReplaySpan:
    def test_replays_captured_input(self):
        span = Span(
            name="anthropic.messages.create",
            kind=SpanKind.LLM,
            model="claude-haiku-4-5",
            input={
                "system": "you are an agent",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "search"}],
            },
            attributes={"max_tokens": 256, "temperature": 0.2},
        )

        rec = _Recorder()
        replay_span(span, rec)
        kwargs = rec.last_kwargs
        assert kwargs["system"] == "you are an agent"
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.2
        assert kwargs["messages"][0]["content"] == "hi"
        assert kwargs["tools"][0]["name"] == "search"

    def test_overrides_apply(self):
        span = Span(
            name="x",
            kind=SpanKind.LLM,
            model="m1",
            input={"messages": [{"role": "user", "content": "hi"}]},
        )
        rec = _Recorder()
        replay_span(span, rec, overrides={"model": "m2", "temperature": 0.7})
        assert rec.last_kwargs["model"] == "m2"
        assert rec.last_kwargs["temperature"] == 0.7

    def test_rejects_non_llm_spans(self):
        span = Span(name="tool", kind=SpanKind.TOOL, input={"x": 1})
        with pytest.raises(ValueError):
            replay_span(span, _Recorder())

    def test_rejects_span_without_input(self):
        span = Span(name="x", kind=SpanKind.LLM)
        with pytest.raises(ValueError):
            replay_span(span, _Recorder())
