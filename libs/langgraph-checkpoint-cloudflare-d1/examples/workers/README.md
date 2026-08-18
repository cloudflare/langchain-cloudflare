# langgraph-checkpoint-cloudflare-d1 Python Worker Example

This example demonstrates using `WorkerCloudflareD1Saver` -- the D1 **binding**
based checkpoint saver -- inside a Cloudflare Python Worker. Unlike
`CloudflareD1Saver` / `AsyncCloudflareD1Saver`, which talk to D1 over the
Cloudflare REST API (account ID + API token), this saver talks to D1 directly
through the Worker's `env.DB` binding: no network round-trip to the Cloudflare
API, no API token required.

> **Note**: Python Workers are currently in beta. APIs may change before
> official release.

## What This Example Shows

```python
from langgraph_checkpoint_cloudflare_d1.worker import WorkerCloudflareD1Saver
from langgraph.graph import StateGraph, START, END
from workers import WorkerEntrypoint, Response


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        saver = WorkerCloudflareD1Saver(self.env.DB)

        builder = StateGraph(int)
        builder.add_node("add_one", lambda state: state + 1)
        builder.add_edge(START, "add_one")
        builder.add_edge("add_one", END)
        graph = builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "1"}}
        result = await graph.ainvoke(3, config)
        return Response.json({"result": result})
```

`WorkerCloudflareD1Saver` is async-only -- Workers run a single-threaded event
loop with no blocking I/O story, so only the `a`-prefixed methods
(`aget_tuple`, `alist`, `aput`, `aput_writes`, `adelete_thread`) are
implemented. Use `graph.ainvoke(...)` / `graph.astream(...)`, not the sync
`graph.invoke(...)`.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A Cloudflare account with Workers enabled
- A D1 database (this example reuses the same `test-db` database configured
  for the other examples in this repo; update `wrangler.jsonc` to point at
  your own database if needed)

## Development

```bash
uv run pywrangler sync
bash ./scripts/setup_pyodide_deps.sh
uv run pywrangler dev
```

Or via the Makefile targets in the package root:

```bash
make worker_sync
make dev_server
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | API documentation |
| `/health` | D1 binding + checkpoint table health check |
| `/checkpointer-put` | POST - save a checkpoint |
| `/checkpointer-put-writes` | POST - save writes for a checkpoint |
| `/checkpointer-get-tuple` | POST - get a checkpoint for a thread |
| `/checkpointer-list` | POST - list checkpoints for a thread |
| `/checkpointer-delete-thread` | POST - delete a thread's checkpoints |
| `/checkpointer-graph` | POST - compile and run a `StateGraph` using the checkpointer |

## Why the setup script?

`langgraph` and `langgraph-checkpoint` pull in `xxhash` and `ormsgpack`, both
C extensions with no Pyodide wheels, so a plain `pywrangler sync` fails to
resolve them. `scripts/setup_pyodide_deps.sh` extracts the exact
langchain-core/langgraph/langgraph-sdk/langgraph-checkpoint wheels this repo's
`uv.lock` resolves and swaps in pure-Python stubs (`stubs/`) for `xxhash` and
`ormsgpack`. See that script's header comment for details.

## Running Tests

The integration tests are in the main `tests/` directory. From the package
root:

```bash
make worker_tests
```

This starts a `pywrangler dev` server against this example and runs HTTP
requests through `tests/worker_tests/test_worker_integration.py`.

## Resources

- [langgraph-checkpoint-cloudflare-d1 Documentation](../../README.md)
- [Python Workers Documentation](https://developers.cloudflare.com/workers/languages/python/)
- [D1 Documentation](https://developers.cloudflare.com/d1/)
