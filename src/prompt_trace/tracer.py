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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from prompt_trace.span import Span, SpanKind, SpanStatus
from prompt_trace.storage import InMemoryStore, Store


@dataclass
class Tracer:
    """Coordinates Span creation and submission to a Store.

    The current-span stack lives in a ContextVar, so each thread and each
    asyncio task sees its own stack. A task starts from a copy of its
    creator's stack: spans opened inside it nest under the span that was
    current when it was created, and never leak into sibling tasks. Spans
    submitted to the store are shared.

    A span may be ended anywhere: in another task, in a done-callback, or
    out of LIFO order. Ended spans are skipped when the stack is read, so
    they never become the parent of a later span.
    """

    store: Store = field(default_factory=InMemoryStore)
    _stack_var: ContextVar[tuple[Span, ...]] = field(
        default_factory=lambda: ContextVar("prompt_trace_span_stack", default=()),
        repr=False,
    )

    def _stack(self) -> tuple[Span, ...]:
        return tuple(s for s in self._stack_var.get() if s.end_time_ns is None)

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
        self._stack_var.set((*self._stack(), span))
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
        self._stack_var.set(self._stack())

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
