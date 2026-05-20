"""Tracer tests."""
from __future__ import annotations

import pytest

from prompt_trace.span import SpanKind, SpanStatus
from prompt_trace.storage import InMemoryStore
from prompt_trace.tracer import Tracer


class TestTracerBasic:
    def test_span_context_manager_persists(self):
        store = InMemoryStore()
        tracer = Tracer(store=store)
        with tracer.span("op", kind=SpanKind.LLM) as s:
            s.attributes["foo"] = "bar"
        assert len(store.spans) == 1
        out = store.spans[0]
        assert out.name == "op"
        assert out.status == SpanStatus.OK
        assert out.attributes["foo"] == "bar"
        assert out.duration_ns >= 0

    def test_nested_spans_share_trace_id_and_set_parent(self):
        tracer = Tracer()
        with tracer.span("outer") as outer:
            outer_id = outer.span_id
            outer_trace = outer.trace_id
            with tracer.span("inner") as inner:
                assert inner.parent_span_id == outer_id
                assert inner.trace_id == outer_trace

    def test_exception_marks_span_error(self):
        store = InMemoryStore()
        tracer = Tracer(store=store)
        with pytest.raises(ValueError):
            with tracer.span("crash"):
                raise ValueError("nope")
        recorded = store.spans[0]
        assert recorded.status == SpanStatus.ERROR
        assert "ValueError" in (recorded.error or "")

    def test_current_span_returns_top_of_stack(self):
        tracer = Tracer()
        assert tracer.current_span() is None
        with tracer.span("outer") as outer:
            assert tracer.current_span() is outer
            with tracer.span("inner") as inner:
                assert tracer.current_span() is inner
            assert tracer.current_span() is outer
        assert tracer.current_span() is None

    def test_explicit_start_end(self):
        store = InMemoryStore()
        tracer = Tracer(store=store)
        s = tracer.start_span("manual", kind=SpanKind.TOOL)
        tracer.end_span(s, status=SpanStatus.OK)
        assert store.spans[0].name == "manual"
        assert store.spans[0].kind == SpanKind.TOOL
