"""Smoke test for the overhead benchmark."""
from __future__ import annotations

from benchmarks.overhead_benchmark import _measure_baseline, _measure_instrumented, _summarize
from prompt_trace.storage import InMemoryStore
from prompt_trace.tracer import Tracer


class TestOverheadSmoke:
    def test_baseline_runs(self):
        durs = _measure_baseline(50)
        assert len(durs) == 50
        summary = _summarize(durs)
        assert summary["median_us"] >= 0.0

    def test_instrumented_inmemory(self):
        tracer = Tracer(store=InMemoryStore())
        durs = _measure_instrumented(50, tracer)
        assert len(durs) == 50
        summary = _summarize(durs)
        assert summary["median_us"] >= 0.0
        # Each call should add a span to the store.
        assert len(tracer.store.spans) == 50
