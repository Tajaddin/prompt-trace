"""Diff tests."""
from __future__ import annotations

from prompt_trace.diff import diff_spans, diff_traces
from prompt_trace.span import Span, SpanKind
from prompt_trace.storage import InMemoryStore


class TestDiffSpans:
    def test_identical_spans_no_field_diffs(self):
        s1 = Span(name="a", kind=SpanKind.LLM, model="m1", input={"k": "v"})
        s2 = Span(name="a", kind=SpanKind.LLM, model="m1", input={"k": "v"})
        result = diff_spans(s1, s2)
        assert result["fields"] == {}

    def test_model_swap_surfaces(self):
        s1 = Span(name="a", model="m1")
        s2 = Span(name="a", model="m2")
        result = diff_spans(s1, s2)
        assert "model" in result["fields"]
        assert result["fields"]["model"]["before"] == "m1"
        assert result["fields"]["model"]["after"] == "m2"

    def test_input_diff_includes_unified_diff_lines(self):
        s1 = Span(name="a", input={"messages": [{"role": "user", "content": "hi"}]})
        s2 = Span(name="a", input={"messages": [{"role": "user", "content": "hello"}]})
        result = diff_spans(s1, s2)
        assert "input" in result["fields"]
        assert result["fields"]["input"]["unified_diff"], "expected unified diff content"


class TestDiffTraces:
    def test_pairs_spans_by_name(self):
        store = InMemoryStore()
        store.insert(Span(name="planner", trace_id="L"))
        store.insert(Span(name="analyzer", trace_id="L"))
        store.insert(Span(name="planner", trace_id="R"))
        store.insert(Span(name="analyzer", trace_id="R"))

        result = diff_traces("L", "R", store)
        assert len(result) == 2
        assert all(item["kind"] == "paired" for item in result)

    def test_reports_left_only_and_right_only(self):
        store = InMemoryStore()
        store.insert(Span(name="planner", trace_id="L"))
        store.insert(Span(name="analyzer", trace_id="L"))
        store.insert(Span(name="planner", trace_id="R"))
        store.insert(Span(name="verifier", trace_id="R"))

        result = diff_traces("L", "R", store)
        kinds = [item["kind"] for item in result]
        assert "left_only" in kinds
        assert "right_only" in kinds
        assert "paired" in kinds
