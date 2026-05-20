"""Diff utilities.

Two functions:
- diff_spans(left, right) - structural diff of two spans (input/output/usage)
- diff_traces(left_id, right_id, store) - aggregate diff of all spans in
  two traces with matching names.

Both return a list of dicts with `path`, `before`, `after`. The result is
JSON-serializable so it can be piped to a UI or saved next to test data.
"""
from __future__ import annotations

import difflib
import json
from typing import Any

from prompt_trace.span import Span
from prompt_trace.storage import Store


def _normalize(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str, indent=2)


def _line_diff(before: Any, after: Any) -> list[str]:
    before_lines = _normalize(before).splitlines()
    after_lines = _normalize(after).splitlines()
    return list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            lineterm="",
            n=2,
            fromfile="before",
            tofile="after",
        )
    )


def diff_spans(left: Span, right: Span) -> dict[str, Any]:
    """Structural diff between two spans.

    Reports differences on input, output, model, usage, and any attribute
    that differs by value. Returns a dict suitable for `json.dumps`.
    """
    out: dict[str, Any] = {
        "left_span_id": left.span_id,
        "right_span_id": right.span_id,
        "fields": {},
    }
    for field_name in ("name", "kind", "model", "status"):
        before = getattr(left, field_name)
        after = getattr(right, field_name)
        if before != after:
            out["fields"][field_name] = {"before": before, "after": after}

    for field_name in ("input", "output", "usage", "attributes"):
        before = getattr(left, field_name)
        after = getattr(right, field_name)
        if before != after:
            out["fields"][field_name] = {
                "before": before,
                "after": after,
                "unified_diff": _line_diff(before, after),
            }

    out["duration_delta_ms"] = round(right.duration_ms - left.duration_ms, 3)
    return out


def diff_traces(left_trace_id: str, right_trace_id: str, store: Store) -> list[dict[str, Any]]:
    """Diff two traces span-by-span.

    Pairs spans by (name, parent_span_id offset) in start_time_ns order.
    Reports spans that exist on only one side as orphans.
    """
    left_spans = store.list_trace(left_trace_id)
    right_spans = store.list_trace(right_trace_id)

    results: list[dict[str, Any]] = []
    # Bucket by name so we can pair siblings.
    right_by_name: dict[str, list[Span]] = {}
    for s in right_spans:
        right_by_name.setdefault(s.name, []).append(s)

    for left in left_spans:
        bucket = right_by_name.get(left.name, [])
        if not bucket:
            results.append(
                {
                    "kind": "left_only",
                    "left_span_id": left.span_id,
                    "name": left.name,
                }
            )
            continue
        right = bucket.pop(0)
        results.append({"kind": "paired", **diff_spans(left, right)})

    for _, leftovers in right_by_name.items():
        for s in leftovers:
            results.append(
                {
                    "kind": "right_only",
                    "right_span_id": s.span_id,
                    "name": s.name,
                }
            )
    return results
