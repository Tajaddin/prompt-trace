"""Tracer tests."""
from __future__ import annotations

import asyncio
import threading

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

    def test_span_ended_out_of_order_is_not_left_as_parent(self):
        tracer = Tracer()
        outer = tracer.start_span("outer")
        inner = tracer.start_span("inner")
        tracer.end_span(outer)
        assert tracer.current_span() is inner
        tracer.end_span(inner)
        assert tracer.current_span() is None
        with tracer.span("next") as nxt:
            pass
        assert nxt.parent_span_id is None


class TestTracerConcurrency:
    async def test_concurrent_tasks_get_separate_traces(self):
        tracer = Tracer()

        async def request(name):
            with tracer.span(name) as outer:
                await asyncio.sleep(0.01)
                with tracer.span(f"{name}.child") as child:
                    await asyncio.sleep(0)
            return outer, child

        (a, a_child), (b, b_child) = await asyncio.gather(request("A"), request("B"))

        assert a.parent_span_id is None
        assert b.parent_span_id is None
        assert a.trace_id != b.trace_id
        assert a_child.parent_span_id == a.span_id
        assert b_child.parent_span_id == b.span_id
        assert tracer.current_span() is None

        later, _ = await request("C")
        assert later.parent_span_id is None
        assert later.trace_id not in {a.trace_id, b.trace_id}

    async def test_task_inherits_enclosing_span_as_parent(self):
        tracer = Tracer()

        async def step(name):
            with tracer.span(name) as s:
                await asyncio.sleep(0)
            return s

        with tracer.span("request") as root:
            left, right = await asyncio.gather(step("left"), step("right"))
            assert tracer.current_span() is root

        assert left.parent_span_id == root.span_id
        assert right.parent_span_id == root.span_id
        assert left.trace_id == right.trace_id == root.trace_id

    async def test_span_ended_in_another_task_leaves_the_stack(self):
        tracer = Tracer()
        stream = tracer.start_span("stream")

        async def finish():
            tracer.end_span(stream)

        await asyncio.create_task(finish())

        assert tracer.current_span() is None
        with tracer.span("next") as nxt:
            pass
        assert nxt.parent_span_id is None
        assert nxt.trace_id != stream.trace_id

    async def test_span_ended_in_a_done_callback_leaves_the_stack(self):
        tracer = Tracer()
        stream = tracer.start_span("stream")
        fut = asyncio.get_running_loop().create_future()
        fut.add_done_callback(lambda _: tracer.end_span(stream))
        fut.set_result(None)
        await fut
        await asyncio.sleep(0)

        assert stream.end_time_ns is not None
        assert tracer.current_span() is None
        with tracer.span("next") as nxt:
            pass
        assert nxt.parent_span_id is None

    def test_threads_do_not_share_a_stack(self):
        tracer = Tracer()
        entered = threading.Event()
        release = threading.Event()

        def worker():
            with tracer.span("thread-span"):
                entered.set()
                release.wait(timeout=5)

        t = threading.Thread(target=worker)
        t.start()
        assert entered.wait(timeout=5)
        with tracer.span("main-span") as main_span:
            pass
        release.set()
        t.join(timeout=5)

        assert main_span.parent_span_id is None
