"""Anthropic SDK instrumenter.

Drop-in wrapper: replace `Anthropic()` with `InstrumentedAnthropic(tracer)`
and every messages.create call writes a Span with input, output, usage,
and model.

Caveats:
- Wraps the synchronous messages.create only. Async + streaming need their
  own instrumenter (left as a TODO with the same Span shape).
- Captures the full request and response payloads. If you want PII-safe
  traces, pass `capture_payloads=False` to drop input/output bodies.
"""
from __future__ import annotations

import time
from typing import Any

from prompt_trace.span import SpanKind, SpanStatus
from prompt_trace.tracer import Tracer


class InstrumentedAnthropic:
    """Wraps an Anthropic client (or any compatible Messages API) with tracing.

    Construct it with `client=Anthropic(api_key=...)` to get real API calls,
    or pass any object that exposes `.messages.create(...)` for tests.
    """

    def __init__(
        self,
        tracer: Tracer,
        *,
        client: Any | None = None,
        capture_payloads: bool = True,
        span_name: str = "anthropic.messages.create",
    ):
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package not installed. Install it, or pass client=... to use a stub."
                ) from exc
            client = Anthropic()
        self._client = client
        self._tracer = tracer
        self._capture = capture_payloads
        self._span_name = span_name
        self.messages = _MessagesNamespace(self)


class _MessagesNamespace:
    """Reproduces the Anthropic SDK shape (`client.messages.create(...)`)."""

    def __init__(self, parent: InstrumentedAnthropic):
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        tracer = self._parent._tracer
        capture = self._parent._capture
        attrs: dict[str, Any] = {
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
            "tool_count": len(kwargs.get("tools", []) or []),
        }
        model = kwargs.get("model")
        with tracer.span(
            self._parent._span_name,
            kind=SpanKind.LLM,
            attributes=attrs,
            model=model,
        ) as span:
            if capture:
                span.input = {
                    "system": kwargs.get("system"),
                    "messages": kwargs.get("messages"),
                    "tools": kwargs.get("tools"),
                }
            t0 = time.perf_counter_ns()
            response = self._parent._client.messages.create(**kwargs)
            elapsed_ns = time.perf_counter_ns() - t0

            usage = self._extract_usage(response)
            if usage is not None:
                span.usage = usage
            if capture:
                span.output = self._extract_output(response)
            span.attributes["wire_latency_ms"] = round(elapsed_ns / 1_000_000, 3)
            span.attributes["stop_reason"] = getattr(response, "stop_reason", None)
            span.finish(status=SpanStatus.OK)
            return response

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int] | None:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return None
        if hasattr(usage_obj, "model_dump"):
            return usage_obj.model_dump()
        if isinstance(usage_obj, dict):
            return dict(usage_obj)
        return None

    @staticmethod
    def _extract_output(response: Any) -> Any:
        content = getattr(response, "content", None)
        if content is None:
            return None
        if isinstance(content, list):
            out = []
            for block in content:
                if hasattr(block, "model_dump"):
                    out.append(block.model_dump())
                elif isinstance(block, dict):
                    out.append(dict(block))
                else:
                    out.append(str(block))
            return out
        return str(content)
