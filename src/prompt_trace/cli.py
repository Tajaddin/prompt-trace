"""Inspect a SQLite trace store from the terminal.

Usage:
    prompt-trace recent --db ./traces.db
    prompt-trace show <trace_id> --db ./traces.db
    prompt-trace diff <trace_a> <trace_b> --db ./traces.db
"""
from __future__ import annotations

import argparse
import json
import sys

from prompt_trace.diff import diff_traces
from prompt_trace.storage import SqliteStore


def _cmd_recent(args: argparse.Namespace) -> int:
    store = SqliteStore(args.db)
    for trace_id in store.recent_traces(limit=args.limit):
        spans = store.list_trace(trace_id)
        total_ms = sum(s.duration_ms for s in spans)
        print(f"{trace_id}  spans={len(spans):>3}  total={total_ms:>9.2f}ms")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    store = SqliteStore(args.db)
    spans = store.list_trace(args.trace_id)
    if not spans:
        print(f"no spans for trace_id={args.trace_id}", file=sys.stderr)
        return 1
    for s in spans:
        prefix = "  " if s.parent_span_id else ""
        print(
            f"{prefix}{s.span_id}  {s.kind.value:>9}  {s.name}  "
            f"{s.duration_ms:.2f}ms  status={s.status.value}"
        )
        if s.usage:
            print(f"{prefix}    usage: {s.usage}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    store = SqliteStore(args.db)
    result = diff_traces(args.left, args.right, store)
    print(json.dumps(result, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect prompt-trace SQLite stores.")
    parser.add_argument("--db", default="./traces.db", help="Path to the SQLite trace store.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_recent = sub.add_parser("recent", help="List recent traces.")
    p_recent.add_argument("--limit", type=int, default=20)
    p_recent.set_defaults(func=_cmd_recent)

    p_show = sub.add_parser("show", help="Show one trace's spans.")
    p_show.add_argument("trace_id")
    p_show.set_defaults(func=_cmd_show)

    p_diff = sub.add_parser("diff", help="Diff two traces.")
    p_diff.add_argument("left")
    p_diff.add_argument("right")
    p_diff.set_defaults(func=_cmd_diff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
