"""OpenTelemetry-style tracing for LLM calls.

Public surface:
    Tracer            - the main entry point; create one per process.
    Span              - one LLM call (or any child operation).
    SqliteStore       - persistent storage backend.
    InstrumentedAnthropic - drop-in wrapper around anthropic.Anthropic.
"""
from prompt_trace.diff import diff_spans, diff_traces
from prompt_trace.instrumenters.anthropic import InstrumentedAnthropic
from prompt_trace.replay import replay_span
from prompt_trace.span import Span, SpanKind, SpanStatus
from prompt_trace.storage import InMemoryStore, SqliteStore
from prompt_trace.tracer import Tracer

__version__ = "0.1.0"
__all__ = [
    "InMemoryStore",
    "InstrumentedAnthropic",
    "Span",
    "SpanKind",
    "SpanStatus",
    "SqliteStore",
    "Tracer",
    "diff_spans",
    "diff_traces",
    "replay_span",
]
