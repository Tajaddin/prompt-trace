"""Tracer.

A Tracer owns:
- One Store (in-memory or SQLite)
- A current-span stack so nested spans pick up the right parent

Two entry points:
- start_span(name, kind, ...) -> Span - manual start/end
- span(name, kind, ...) - context manager that ends automatically

The context manager is the common path; the manual start_span exists for
instrumenters that need to hand the span back to caller-defined cleanup
(e.g. when the SDK call returns a streaming iterator that finishes async).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from prompt_trace.span import Span, SpanKind, SpanStatus
from prompt_trace.storage import InMemoryStore, Store


@dataclass
class Tracer:
    """Coordinates Span creation and submission to a Store.

    Tracers are thread-local in the sense that the current-span stack is
    isolated per thread; spans submitted to the store are shared.
    """

    store: Store = field(default_factory=InMemoryStore)
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def current_span(self) -> Span | None:
        stack = self._stack()
        return stack[-1] if stack else None

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        *,
        attributes: dict[str, Any] | None = None,
        model: str | None = None,
        trace_id: str | None = None,
    ) -> Span:
        parent = self.current_span()
        span = Span(
            name=name,
            kind=kind,
            attributes=attributes or {},
            model=model,
        )
        if parent is not None:
            span.parent_span_id = parent.span_id
            span.trace_id = parent.trace_id
        elif trace_id is not None:
            span.trace_id = trace_id
        self._stack().append(span)
        return span

    def end_span(
        self,
        span: Span,
        *,
        status: SpanStatus = SpanStatus.OK,
        error: str | None = None,
    ) -> None:
        span.finish(status=status, error=error)
        # Persist after finish so end_time_ns is set.
        self.store.insert(span)
        stack = self._stack()
        if stack and stack[-1] is span:
            stack.pop()

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        *,
        attributes: dict[str, Any] | None = None,
        model: str | None = None,
        trace_id: str | None = None,
    ):
        s = self.start_span(
            name, kind=kind, attributes=attributes, model=model, trace_id=trace_id
        )
        try:
            yield s
        except Exception as exc:  # noqa: BLE001 - we re-raise after recording
            self.end_span(s, status=SpanStatus.ERROR, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.end_span(s, status=SpanStatus.OK)
