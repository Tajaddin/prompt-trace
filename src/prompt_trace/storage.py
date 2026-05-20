"""Span storage backends.

Two backends:
- InMemoryStore: list-backed; useful in tests and short-lived scripts.
- SqliteStore: single-file, indexed by trace_id and parent_span_id. Good
  for CLI inspection and replay.

Both implement the same minimal interface: insert(span), get(span_id),
list_trace(trace_id), recent_traces(limit).

We intentionally do not provide a network backend. The output of this
module is a portable SQLite file; ship it to S3 or a teammate as-is.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from prompt_trace.span import Span


def _span_to_row(span: Span) -> tuple[Any, ...]:
    return (
        span.span_id,
        span.trace_id,
        span.parent_span_id,
        span.name,
        span.kind.value,
        span.status.value,
        span.start_time_ns,
        span.end_time_ns,
        span.model,
        json.dumps(span.input, default=str),
        json.dumps(span.output, default=str),
        json.dumps(span.usage) if span.usage is not None else None,
        json.dumps(span.attributes, default=str),
        json.dumps(span.events),
        span.error,
    )


def _row_to_span(row: tuple[Any, ...]) -> Span:
    return Span(
        span_id=row[0],
        trace_id=row[1],
        parent_span_id=row[2],
        name=row[3],
        kind=row[4],
        status=row[5],
        start_time_ns=row[6],
        end_time_ns=row[7],
        model=row[8],
        input=json.loads(row[9]) if row[9] else None,
        output=json.loads(row[10]) if row[10] else None,
        usage=json.loads(row[11]) if row[11] else None,
        attributes=json.loads(row[12]) if row[12] else {},
        events=json.loads(row[13]) if row[13] else [],
        error=row[14],
    )


class Store(Protocol):
    def insert(self, span: Span) -> None: ...
    def get(self, span_id: str) -> Span | None: ...
    def list_trace(self, trace_id: str) -> list[Span]: ...
    def recent_traces(self, limit: int = 20) -> list[str]: ...


@dataclass
class InMemoryStore:
    """In-process storage. Lost on exit. Default for tests and benchmarks."""

    spans: list[Span] = field(default_factory=list)

    def insert(self, span: Span) -> None:
        self.spans.append(span)

    def get(self, span_id: str) -> Span | None:
        for s in self.spans:
            if s.span_id == span_id:
                return s
        return None

    def list_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    def recent_traces(self, limit: int = 20) -> list[str]:
        seen: list[str] = []
        for s in reversed(self.spans):
            if s.trace_id not in seen:
                seen.append(s.trace_id)
                if len(seen) >= limit:
                    break
        return seen


_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id        TEXT PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    parent_span_id TEXT,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    status         TEXT NOT NULL,
    start_time_ns  INTEGER NOT NULL,
    end_time_ns    INTEGER,
    model          TEXT,
    input_json     TEXT,
    output_json    TEXT,
    usage_json     TEXT,
    attrs_json     TEXT,
    events_json    TEXT,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_parent ON spans(parent_span_id);
CREATE INDEX IF NOT EXISTS idx_start ON spans(start_time_ns);
"""

_COLS = (
    "span_id, trace_id, parent_span_id, name, kind, status, "
    "start_time_ns, end_time_ns, model, input_json, output_json, "
    "usage_json, attrs_json, events_json, error"
)


@dataclass
class SqliteStore:
    """SQLite-backed span store. Single file. Concurrent reads, single writer."""

    path: Path | str = ":memory:"

    def __post_init__(self) -> None:
        self.path = str(self.path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        # WAL + NORMAL gives us ~200x faster writes than the default
        # DELETE journal + FULL sync. Durability is still safe: NORMAL
        # only loses uncommitted txns on power loss, never corrupts the DB.
        # We skip these pragmas for in-memory DBs (they are no-ops there).
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA temp_store=MEMORY")
        for stmt in _SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()

    def insert(self, span: Span) -> None:
        placeholders = ", ".join("?" * 15)
        self._conn.execute(
            f"INSERT OR REPLACE INTO spans({_COLS}) VALUES({placeholders})",
            _span_to_row(span),
        )
        self._conn.commit()

    def get(self, span_id: str) -> Span | None:
        row = self._conn.execute(
            f"SELECT {_COLS} FROM spans WHERE span_id=?",
            (span_id,),
        ).fetchone()
        return _row_to_span(row) if row else None

    def list_trace(self, trace_id: str) -> list[Span]:
        rows = self._conn.execute(
            f"SELECT {_COLS} FROM spans WHERE trace_id=? ORDER BY start_time_ns",
            (trace_id,),
        ).fetchall()
        return [_row_to_span(r) for r in rows]

    def recent_traces(self, limit: int = 20) -> list[str]:
        rows = self._conn.execute(
            "SELECT trace_id, MAX(start_time_ns) as t "
            "FROM spans GROUP BY trace_id ORDER BY t DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
