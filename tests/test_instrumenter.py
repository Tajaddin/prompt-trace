"""Instrumenter tests with a stub Anthropic client."""
from __future__ import annotations

from typing import Any

from prompt_trace.instrumenters.anthropic import InstrumentedAnthropic
from prompt_trace.span import SpanKind, SpanStatus
from prompt_trace.storage import InMemoryStore
from prompt_trace.tracer import Tracer


class _StubResponse:
    class _Usage:
        def model_dump(self):
            return {"input_tokens": 7, "output_tokens": 14}

    class _Block:
        def __init__(self, text: str):
            self.type = "text"
            self.text = text

        def model_dump(self):
            return {"type": "text", "text": self.text}

    def __init__(self):
        self.content = [_StubResponse._Block("response text")]
        self.usage = _StubResponse._Usage()
        self.stop_reason = "end_turn"
        self.model = "stub-model"


class _StubMessages:
    def __init__(self):
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return _StubResponse()


class _StubAnthropic:
    def __init__(self):
        self.messages = _StubMessages()


class TestInstrumentedAnthropic:
    def test_captures_span_around_messages_create(self):
        store = InMemoryStore()
        tracer = Tracer(store=store)
        client = InstrumentedAnthropic(tracer, client=_StubAnthropic())

        resp = client.messages.create(
            model="stub-model",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "x"}],
        )
        assert resp.content[0].text == "response text"

        assert len(store.spans) == 1
        s = store.spans[0]
        assert s.name == "anthropic.messages.create"
        assert s.kind == SpanKind.LLM
        assert s.status == SpanStatus.OK
        assert s.model == "stub-model"
        assert s.usage == {"input_tokens": 7, "output_tokens": 14}
        assert s.attributes["max_tokens"] == 100
        assert s.attributes["tool_count"] == 1
        assert s.attributes["stop_reason"] == "end_turn"

    def test_capture_payloads_false_drops_input_output(self):
        store = InMemoryStore()
        tracer = Tracer(store=store)
        client = InstrumentedAnthropic(tracer, client=_StubAnthropic(), capture_payloads=False)
        client.messages.create(model="m", max_tokens=10, messages=[])
        s = store.spans[0]
        assert s.input is None
        assert s.output is None
        # Usage is still captured because it has no PII.
        assert s.usage is not None

    def test_records_wire_latency(self):
        store = InMemoryStore()
        tracer = Tracer(store=store)
        client = InstrumentedAnthropic(tracer, client=_StubAnthropic())
        client.messages.create(model="m", max_tokens=10, messages=[])
        assert "wire_latency_ms" in store.spans[0].attributes
        assert store.spans[0].attributes["wire_latency_ms"] >= 0.0

    def test_propagates_trace_id_to_nested_span(self):
        store = InMemoryStore()
        tracer = Tracer(store=store)
        client = InstrumentedAnthropic(tracer, client=_StubAnthropic())
        with tracer.span("outer") as outer:
            client.messages.create(model="m", max_tokens=10, messages=[])
            # spans are flushed on finish so we look for the inner span
            # in the store via its name and the outer trace_id.
        all_for_trace = [s for s in store.spans if s.trace_id == outer.trace_id]
        names = {s.name for s in all_for_trace}
        assert "outer" in names
        assert "anthropic.messages.create" in names
