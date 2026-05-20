"""Replay a stored span against a live client.

Given a span_id, look up the captured input (system, messages, tools, max
tokens, etc.) and replay it against any callable that matches the
Anthropic Messages API shape.

Why this is useful:
- Production regression: a change to a prompt template; replay yesterday's
  span against the new template and diff the output.
- Local debug: pull a span from prod, replay locally with verbose logging.
- Cost what-if: replay against a cheaper model and compare quality.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_trace.span import Span


def replay_span(
    span: Span,
    create_messages: Callable[..., Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Replay one span's captured input against `create_messages`.

    `create_messages` must accept the same kwargs as
    `anthropic.Anthropic().messages.create`.

    `overrides` lets you swap model, temperature, etc. without editing the
    stored span. Useful for A/B replay against a different model.
    """
    if span.kind.value != "llm":
        raise ValueError(f"replay_span only supports LLM spans, got kind={span.kind!r}")
    if span.input is None:
        raise ValueError(f"span {span.span_id} has no captured input to replay")

    payload: dict[str, Any] = {}
    payload["system"] = span.input.get("system")
    payload["messages"] = span.input.get("messages") or []
    if span.input.get("tools"):
        payload["tools"] = span.input["tools"]
    if span.model is not None:
        payload["model"] = span.model
    for key in ("max_tokens", "temperature"):
        if key in span.attributes and span.attributes[key] is not None:
            payload[key] = span.attributes[key]
    if "max_tokens" not in payload:
        payload["max_tokens"] = 1024

    if overrides:
        payload.update(overrides)

    # Drop keys with None values so the receiver does not see explicit nulls.
    payload = {k: v for k, v in payload.items() if v is not None}

    return create_messages(**payload)
