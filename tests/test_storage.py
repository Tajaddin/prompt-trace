"""Storage backend tests."""
from __future__ import annotations

from prompt_trace.span import Span, SpanKind, SpanStatus
from prompt_trace.storage import InMemoryStore, SqliteStore


class TestInMemoryStore:
    def test_insert_and_get(self):
        store = InMemoryStore()
        s = Span(name="t")
        store.insert(s)
        assert store.get(s.span_id) is not None
        assert store.get(s.span_id).name == "t"

    def test_missing_get_returns_none(self):
        assert InMemoryStore().get("nope") is None

    def test_list_trace_groups_by_trace_id(self):
        store = InMemoryStore()
        a = Span(name="a", trace_id="tr_1")
        b = Span(name="b", trace_id="tr_1")
        c = Span(name="c", trace_id="tr_2")
        for s in (a, b, c):
            store.insert(s)
        assert {s.name for s in store.list_trace("tr_1")} == {"a", "b"}
        assert {s.name for s in store.list_trace("tr_2")} == {"c"}

    def test_recent_traces_dedupes(self):
        store = InMemoryStore()
        store.insert(Span(name="a", trace_id="tr_1"))
        store.insert(Span(name="b", trace_id="tr_1"))
        store.insert(Span(name="c", trace_id="tr_2"))
        recent = store.recent_traces(limit=10)
        assert recent == ["tr_2", "tr_1"]


class TestSqliteStore:
    def test_insert_and_get_roundtrip(self):
        store = SqliteStore(":memory:")
        s = Span(
            name="llm.call",
            kind=SpanKind.LLM,
            status=SpanStatus.OK,
            model="claude-haiku-4-5",
            input={"messages": [{"role": "user", "content": "hi"}]},
            output=[{"type": "text", "text": "hello"}],
            usage={"input_tokens": 5, "output_tokens": 10},
            attributes={"max_tokens": 1024},
        )
        s.finish()
        store.insert(s)

        out = store.get(s.span_id)
        assert out is not None
        assert out.name == "llm.call"
        assert out.kind == SpanKind.LLM
        assert out.model == "claude-haiku-4-5"
        assert out.usage == {"input_tokens": 5, "output_tokens": 10}
        assert out.input["messages"][0]["content"] == "hi"
        store.close()

    def test_list_trace_orders_by_start_time(self):
        store = SqliteStore(":memory:")
        a = Span(name="a", trace_id="tr_x", start_time_ns=300)
        b = Span(name="b", trace_id="tr_x", start_time_ns=100)
        c = Span(name="c", trace_id="tr_x", start_time_ns=200)
        for s in (a, b, c):
            store.insert(s)
        spans = store.list_trace("tr_x")
        assert [s.name for s in spans] == ["b", "c", "a"]
        store.close()

    def test_recent_traces(self):
        store = SqliteStore(":memory:")
        store.insert(Span(name="a", trace_id="tr_old", start_time_ns=100))
        store.insert(Span(name="b", trace_id="tr_new", start_time_ns=500))
        assert store.recent_traces() == ["tr_new", "tr_old"]
        store.close()

    def test_persists_to_file(self, tmp_path):
        db = tmp_path / "trace.db"
        store = SqliteStore(db)
        s = Span(name="persistent", trace_id="tr_p")
        s.finish()
        store.insert(s)
        store.close()

        store2 = SqliteStore(db)
        out = store2.get(s.span_id)
        assert out is not None
        assert out.name == "persistent"
        store2.close()
