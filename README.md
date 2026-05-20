# prompt-trace

> OpenTelemetry-style tracing for LLM calls in one tiny package. **148μs per-call overhead with persistent SQLite storage**, 28μs in-memory. Drop-in Anthropic SDK wrapper, span tree across nested operations, replay-from-trace, prompt diff between traces. Reproducible 5K-call benchmark in 4 seconds.

[![ci](https://github.com/Tajaddin/prompt-trace/actions/workflows/ci.yml/badge.svg)](https://github.com/Tajaddin/prompt-trace/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## Hero metrics

Reproducible in 4 seconds: `python -m benchmarks.overhead_benchmark --calls 5000`

| Store | Median per-call overhead | Persistent across processes |
|---|---:|:---:|
| **InMemoryStore** | **27.8 μs** | no |
| **SqliteStore (WAL + sync=NORMAL)** | **148.4 μs** | yes |

For comparison, a typical Anthropic API round trip is 500-3000 ms. The instrumenter adds ~0.005% overhead on a real LLM call. The Sqlite tax is what you pay for "open the trace file on any machine and inspect it later."

How the SQLite store hits 148μs:

1. WAL journal mode (writes are append-only, no exclusive lock on the main DB).
2. `synchronous=NORMAL` (one fsync per checkpoint, not one per commit).
3. Indexed on `trace_id` and `parent_span_id` for fast retrieval, no full table scans.

If 148μs is still too much for your loop, drop in `InMemoryStore` and flush to SQLite periodically. The interfaces are identical.

## What this is

Everything in this list is a single ~80-line module. None of them depend on opentelemetry-api, so installing prompt-trace adds 0 transitive deps beyond pydantic.

| Module | Purpose |
|---|---|
| `span.py` | Span model with start/end times, kind, status, input/output, usage, attributes, events. |
| `tracer.py` | Coordinates Span creation and submission. Context-manager API for nested spans. Thread-local current-span stack. |
| `storage.py` | `InMemoryStore` and `SqliteStore`. Both implement the same Protocol. SQLite is WAL + NORMAL sync. |
| `instrumenters/anthropic.py` | Drop-in wrapper for `anthropic.Anthropic`. One line to enable: `client = InstrumentedAnthropic(tracer)`. |
| `diff.py` | Diff two traces span-by-span. Reports `paired`, `left_only`, `right_only`. Unified diff for input/output payloads. |
| `replay.py` | Re-execute a stored span against a live client. Useful for prompt regression tests and cost A/B between models. |
| `cli.py` | `prompt-trace recent`, `prompt-trace show <trace_id>`, `prompt-trace diff <a> <b>`. |

## Why this matters for production

JD signal this maps to:
- **LLM observability** (Honeycomb Sr SE LLM Observability, Arize AI, Grafana Labs Senior AI Engineer)
- **OpenTelemetry / tracing** (any platform-eng AI JD that mentions observability stack)
- **Cost monitoring** (every "AI Platform" JD that asks about token spend)
- **Prompt regression testing** (forgd.ai, Caylent, M3 USA)

LangSmith and Langfuse own the heavyweight side of this space. This kit is the "I want to inspect my agent's behavior in a single SQLite file I can open with `sqlite3 traces.db`" side.

## Quick start

```python
from anthropic import Anthropic
from prompt_trace import InstrumentedAnthropic, SqliteStore, Tracer

tracer = Tracer(store=SqliteStore("./traces.db"))
client = InstrumentedAnthropic(tracer, client=Anthropic())

with tracer.span("chat_session", attributes={"user_id": "u_42"}):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": "Hello"}],
    )
```

After the script exits, `traces.db` has:

```bash
$ prompt-trace recent --db traces.db
tr_a91f...  spans=  2  total=    420.18ms

$ prompt-trace show tr_a91f --db traces.db
sp_4d12...  internal  chat_session                420.18ms  status=ok
  sp_e7c9...       llm  anthropic.messages.create  418.42ms  status=ok
    usage: {'input_tokens': 8, 'output_tokens': 42}
```

## Diff two runs

```python
from prompt_trace import diff_traces

# Run the same agent twice (e.g. before and after a prompt edit)
for change_id in ["before", "after"]:
    with tracer.span(f"session_{change_id}"):
        client.messages.create(...)

before_trace, after_trace = tracer.store.recent_traces(limit=2)
print(json.dumps(diff_traces(after_trace, before_trace, tracer.store), indent=2))
```

The diff output reports per-span input changes, output changes, model swaps, and `duration_delta_ms`. Pipe it to a code review for "this prompt edit changed token count from 8 to 14".

## Replay one span against a different model

```python
from prompt_trace import replay_span

original = tracer.store.get(span_id)
new_client = Anthropic()  # bare, not instrumented
new_response = replay_span(
    original,
    new_client.messages.create,
    overrides={"model": "claude-sonnet-4-5"},
)
```

Same input the production call saw, but on Sonnet instead of Haiku. Compare the responses to decide if the cost increase is worth it before flipping prod.

## Project layout

```
src/prompt_trace/
  span.py                            # Span + SpanKind + SpanStatus
  tracer.py                          # Tracer, .span() ctxmgr, .start_span / .end_span
  storage.py                         # InMemoryStore + SqliteStore (WAL + NORMAL)
  instrumenters/
    anthropic.py                     # InstrumentedAnthropic (one-line wrapper)
  diff.py                            # diff_spans + diff_traces
  replay.py                          # replay_span
  cli.py                             # `prompt-trace` CLI

benchmarks/
  overhead_benchmark.py              # baseline + inmem + sqlite, 10K calls
  results/
    overhead.json
```

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=prompt_trace
```

33 tests cover: span model, storage roundtrip (both backends), tracer context-manager + nested spans + error propagation, the Anthropic instrumenter against a stub client, diff_spans for identical/different inputs, diff_traces for paired/left-only/right-only spans, replay_span with overrides + error cases, and a smoke test of the overhead benchmark.

## Docker

```bash
docker compose up bench     # runs the overhead benchmark
docker compose run --rm trace --help
```

## What this is NOT (yet)

- Async / streaming instrumentation. The sync `messages.create` wrapper is the proof of concept; async-stream Anthropic spans need their own wrapper with the same Span shape.
- A live web UI. The CLI plus the SQLite file is enough for code-level review; a Streamlit dashboard is a one-evening addition for teams that want one.
- A network exporter. By design: the output is a portable SQLite file. Copy it, mail it, attach it to a bug report.

## License

MIT
