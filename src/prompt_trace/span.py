"""Span model.

A Span is one observable operation: an LLM call, a tool execution, a
retrieval, or any nested step. Spans form a tree via parent_span_id and
share a trace_id with sibling and ancestor spans.

The model matches the OpenTelemetry span shape closely enough that any
OTel-aware UI can consume an exported span. We avoid the full OTel
dependency to keep the package import-light.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    """High-level category of a span. Mirrors OTel SpanKind partially."""

    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    INTERNAL = "internal"


class SpanStatus(str, Enum):
    """Terminal status of a span."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Span(BaseModel):
    """One traced operation.

    A few invariants:
    - span_id is unique within a Tracer process; trace_id groups siblings.
    - start_time_ns and end_time_ns are perf_counter_ns values; subtract to
      get duration in nanoseconds. We avoid wall-clock so duration is robust
      to NTP adjustments.
    - input and output are arbitrary JSON-serializable payloads. The
      instrumenter decides what to capture.
    - usage is an optional dict with token counts when the operation hit an
      LLM. Keys mirror Anthropic's usage block: input_tokens, output_tokens,
      cache_creation_input_tokens, cache_read_input_tokens.
    """

    span_id: str = Field(default_factory=lambda: _new_id("sp"))
    trace_id: str = Field(default_factory=lambda: _new_id("tr"))
    parent_span_id: str | None = None

    name: str
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET

    start_time_ns: int = Field(default_factory=time.perf_counter_ns)
    end_time_ns: int | None = None

    model: str | None = None
    input: Any = None
    output: Any = None
    usage: dict[str, int] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

    @property
    def duration_ns(self) -> int:
        if self.end_time_ns is None:
            return 0
        return self.end_time_ns - self.start_time_ns

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append(
            {
                "name": name,
                "time_ns": time.perf_counter_ns(),
                "attributes": dict(attrs),
            }
        )

    def finish(self, status: SpanStatus = SpanStatus.OK, error: str | None = None) -> None:
        if self.end_time_ns is None:
            self.end_time_ns = time.perf_counter_ns()
        self.status = status
        if error is not None:
            self.error = error
            self.status = SpanStatus.ERROR
