# langmem-cloudflare-vectorize Python Worker Example

This example demonstrates using `CloudflareVectorizeBaseStore` -- the LangGraph
`BaseStore` implementation backed by Cloudflare Vectorize -- inside a
Cloudflare Python Worker via the native `VECTORIZE` and `AI` bindings instead
of the REST API: no network round-trip to the Cloudflare API, no API token
required.

> **Note**: Python Workers are currently in beta. APIs may change before
> official release.

## What This Example Shows

```python
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langmem_cloudflare_vectorize import CloudflareVectorizeBaseStore
from workers import WorkerEntrypoint, Response

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        embeddings = CloudflareWorkersAIEmbeddings(
            binding=self.env.AI, model_name="@cf/baai/bge-base-en-v1.5"
        )
        store = CloudflareVectorizeBaseStore(
            embedding_function=embeddings, binding=self.env.VECTORIZE
        )

        await store.aput(("docs",), "k1", {"text": "hello world"})
        item = await store.aget(("docs",), "k1")
        return Response.json({"value": item.value})
```

`CloudflareVectorize`'s (langchain-cloudflare) binding support is async-only,
so `CloudflareVectorizeBaseStore`'s sync methods (`get`/`put`/`delete`/
`search`/`batch`) bridge to their async counterparts via
`pyodide.ffi.run_sync()` when a `binding` is set -- the same mechanism
`sqlalchemy_cloudflare_d1.SyncWorkerConnection` uses for SQLAlchemy's sync
engine, and `WorkerCloudflareD1Saver` uses in `langgraph-checkpoint-cloudflare-d1`.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A Cloudflare account with Workers, Vectorize, and Workers AI enabled
- A Vectorize index -- this example reuses the same `langchain-test-persistent`
  index (768 dimensions, matching `@cf/baai/bge-base-en-v1.5`) the
  `langchain-cloudflare` Worker example already uses; update `wrangler.jsonc`
  to point at your own index if needed

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
| `/health` | Vectorize/AI binding health check |
| `/store-put` | POST - save an item |
| `/store-get` | POST - get an item |
| `/store-delete` | POST - delete an item |
| `/store-search` | POST - search items within a namespace prefix |
| `/store-graph` | POST - compile and run a `StateGraph` with the store attached, exercised from inside a node. Writes a memory by default; pass `"recall": true` to read one back in a separate run instead (see note below) |

**Note on `/store-graph` and eventual consistency**: Vectorize inserts are
processed by an asynchronous mutation queue, not indexed synchronously, so a
`get()` immediately after a `put()` for the same id can legitimately miss for
anywhere from a few seconds up to tens of seconds under load. `/store-graph`
therefore does the write and the read as two separate requests/graph runs
(`"recall": true` reads) rather than a write-then-immediate-read in the same
node -- which matches how a memory store is actually used (read back in a
later turn/session), and avoids a demo that flakes on an instant re-read.

## Why the setup script?

`langgraph` and `langgraph-checkpoint` (a transitive dependency via
`langgraph.store`/`langgraph.graph`) pull in `xxhash` and `ormsgpack`, both C
extensions with no Pyodide wheels, so a plain `pywrangler sync` fails to
resolve them. `scripts/setup_pyodide_deps.sh` extracts the exact
langchain-core/langgraph/langgraph-sdk/langgraph-checkpoint wheels this
repo's `uv.lock` resolves and swaps in pure-Python stubs (`stubs/`) for
`xxhash`, `ormsgpack`, `uuid_utils`, and `websockets`. See that script's
header comment for details -- it's the same gap `langgraph-checkpoint-cloudflare-d1`'s
Worker example hits, since both pull in the same `langgraph` core.

## Running Tests

The integration tests are in the main `tests/` directory. From the package
root:

```bash
make worker_tests
```

This starts a `pywrangler dev` server against this example and runs HTTP
requests through `tests/worker_tests/test_worker_integration.py`.

## Resources

- [langmem-cloudflare-vectorize Documentation](../../README.md)
- [Python Workers Documentation](https://developers.cloudflare.com/workers/languages/python/)
- [Vectorize Documentation](https://developers.cloudflare.com/vectorize/)
