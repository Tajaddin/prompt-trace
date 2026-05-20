"""Span model tests."""
from __future__ import annotations

import time

from prompt_trace.span import Span, SpanKind, SpanStatus


class TestSpan:
    def test_defaults(self):
        span = Span(name="test")
        assert span.span_id.startswith("sp_")
        assert span.trace_id.startswith("tr_")
        assert span.kind == SpanKind.INTERNAL
        assert span.status == SpanStatus.UNSET
        assert span.end_time_ns is None
        assert span.duration_ns == 0

    def test_finish_sets_end_time_and_status(self):
        span = Span(name="test")
        time.sleep(0.001)  # ensure measurable duration
        span.finish()
        assert span.end_time_ns is not None
        assert span.duration_ns > 0
        assert span.status == SpanStatus.OK

    def test_finish_with_error(self):
        span = Span(name="test")
        span.finish(error="boom")
        assert span.status == SpanStatus.ERROR
        assert span.error == "boom"

    def test_add_event(self):
        span = Span(name="test")
        span.add_event("hit_cache", token_count=420)
        assert len(span.events) == 1
        assert span.events[0]["name"] == "hit_cache"
        assert span.events[0]["attributes"]["token_count"] == 420

    def test_duration_ms_conversion(self):
        span = Span(name="test")
        span.start_time_ns = 1_000_000
        span.end_time_ns = 6_000_000
        assert span.duration_ms == 5.0
