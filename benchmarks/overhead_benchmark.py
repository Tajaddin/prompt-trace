"""Overhead benchmark.

Measures how much wall-clock time prompt-trace adds per LLM call. We avoid
the network entirely by using a stub Anthropic-shape client that returns
a constant response. The numerator is the instrumenter + storage round-trip.

Two backends are measured:
- InMemoryStore (cheapest, append-only list)
- SqliteStore (most realistic for production)

Usage:
    python -m benchmarks.overhead_benchmark --calls 10000

Hero: per-call instrumenter overhead under 100 microseconds with SQLite.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from prompt_trace.instrumenters.anthropic import InstrumentedAnthropic
from prompt_trace.storage import InMemoryStore, SqliteStore
from prompt_trace.tracer import Tracer

RESULTS_DIR = Path(__file__).parent / "results"


class _StubAnthropic:
    """Anthropic-shape stub that returns a constant response. Zero network."""

    class _Usage:
        def __init__(self):
            self.input_tokens = 10
            self.output_tokens = 20

        def model_dump(self):
            return {"input_tokens": 10, "output_tokens": 20}

    class _Block:
        def __init__(self, text: str):
            self.type = "text"
            self.text = text

        def model_dump(self):
            return {"type": "text", "text": self.text}

    class _Response:
        def __init__(self):
            self.content = [_StubAnthropic._Block("ok")]
            self.usage = _StubAnthropic._Usage()
            self.stop_reason = "end_turn"
            self.model = "stub"

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            return _StubAnthropic._Response()

    def __init__(self):
        self.messages = _StubAnthropic._Messages()


def _measure_baseline(n: int) -> list[float]:
    """Cost of calling the stub WITHOUT any instrumentation. Lower bound."""
    client = _StubAnthropic()
    durs: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        client.messages.create(model="stub", messages=[], max_tokens=10)
        durs.append((time.perf_counter_ns() - t0) / 1_000)  # microseconds
    return durs


def _measure_instrumented(n: int, tracer: Tracer) -> list[float]:
    client = InstrumentedAnthropic(tracer, client=_StubAnthropic())
    durs: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        client.messages.create(model="stub", messages=[], max_tokens=10)
        durs.append((time.perf_counter_ns() - t0) / 1_000)  # microseconds
    return durs


def _summarize(durs: list[float]) -> dict[str, float]:
    durs_sorted = sorted(durs)
    n = len(durs_sorted)
    return {
        "n": n,
        "mean_us": round(statistics.mean(durs_sorted), 2),
        "median_us": round(statistics.median(durs_sorted), 2),
        "p95_us": round(durs_sorted[int(0.95 * n)] if n else 0.0, 2),
        "p99_us": round(durs_sorted[int(0.99 * n)] if n > 99 else durs_sorted[-1], 2),
        "max_us": round(max(durs_sorted) if durs_sorted else 0.0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prompt-trace overhead benchmark.")
    parser.add_argument("--calls", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "overhead.json")
    args = parser.parse_args(argv)

    print(f"Warming up ({min(args.calls, 1000)} calls)...", file=sys.stderr)
    _ = _measure_baseline(min(args.calls, 1000))

    print(f"Measuring baseline ({args.calls} calls)...", file=sys.stderr)
    baseline = _measure_baseline(args.calls)

    print(f"Measuring instrumented + InMemoryStore ({args.calls} calls)...", file=sys.stderr)
    inmem_tracer = Tracer(store=InMemoryStore())
    inmem = _measure_instrumented(args.calls, inmem_tracer)

    print(f"Measuring instrumented + SqliteStore ({args.calls} calls)...", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_store = SqliteStore(Path(tmp) / "bench.db")
        sqlite_tracer = Tracer(store=sqlite_store)
        try:
            sqlite = _measure_instrumented(args.calls, sqlite_tracer)
        finally:
            # Close the connection BEFORE TemporaryDirectory tries to delete the file;
            # Windows holds an exclusive lock until close().
            sqlite_store.close()

    baseline_summary = _summarize(baseline)
    inmem_summary = _summarize(inmem)
    sqlite_summary = _summarize(sqlite)

    inmem_overhead = round(inmem_summary["median_us"] - baseline_summary["median_us"], 2)
    sqlite_overhead = round(sqlite_summary["median_us"] - baseline_summary["median_us"], 2)

    summary = {
        "n_calls": args.calls,
        "baseline": baseline_summary,
        "inmemory": inmem_summary,
        "sqlite": sqlite_summary,
        "headline": {
            "inmemory_overhead_us": inmem_overhead,
            "sqlite_overhead_us": sqlite_overhead,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n=== Overhead benchmark summary ===")
    print(json.dumps(summary["headline"], indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
