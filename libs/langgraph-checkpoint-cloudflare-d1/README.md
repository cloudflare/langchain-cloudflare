# langgraph-checkpoint-cloudflare-d1

## Installation

```bash
pip install -U langgraph-checkpoint-cloudflare-d1
```

## Usage

This package provides both synchronous and asynchronous interfaces for saving and retrieving LangGraph checkpoints in Cloudflare D1.

### Synchronous

```python
from langgraph_checkpoint_cloudflare_d1 import CloudflareD1Saver

# Cloudflare credentials
account_id = "your-cloudflare-account-id"
database_id = "your-d1-database-id"
api_token = "your-cloudflare-api-token"

# Configuration for checkpoint operations
write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

# Initialize the saver with proper credentials
with CloudflareD1Saver(
    account_id=account_id, database_id=database_id, api_token=api_token
) as checkpointer:
    # Setup the database tables (idempotent operation)
    checkpointer.setup()

    # Sample checkpoint data
    checkpoint = {
        "v": 2,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {"my_key": "meow", "node": "node"},
        "channel_versions": {"__start__": 2, "my_key": 3, "start:node": 3, "node": 3},
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": 1},
            "node": {"start:node": 2},
        },
        "pending_sends": [],
    }

    # Store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # Load checkpoint
    loaded_checkpoint = checkpointer.get_tuple(read_config)

    # List checkpoints
    checkpoints = list(checkpointer.list(read_config))
```


### Async

```python
from langgraph_checkpoint_cloudflare_d1 import AsyncCloudflareD1Saver
import asyncio


async def main():
    # Cloudflare credentials
    account_id = "your-cloudflare-account-id"
    database_id = "your-d1-database-id"
    api_token = "your-cloudflare-api-token"

    # Configuration for checkpoint operations
    write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
    read_config = {"configurable": {"thread_id": "1"}}

    # Initialize the async saver with proper credentials
    async with AsyncCloudflareD1Saver(
        account_id=account_id, database_id=database_id, api_token=api_token
    ) as checkpointer:
        # Sample checkpoint data
        checkpoint = {
            "v": 2,
            "ts": "2024-07-31T20:14:19.804150+00:00",
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "channel_values": {"my_key": "meow", "node": "node"},
            "channel_versions": {
                "__start__": 2,
                "my_key": 3,
                "start:node": 3,
                "node": 3,
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {"__start__": 1},
                "node": {"start:node": 2},
            },
            "pending_sends": [],
        }

        # Setup happens automatically but can be called explicitly
        await checkpointer.setup()

        # Store checkpoint
        await checkpointer.put(write_config, checkpoint, {}, {})

        # Load checkpoint
        loaded_checkpoint = await checkpointer.get_tuple(read_config)

        # List checkpoints
        checkpoints = [cp async for cp in checkpointer.list(read_config)]


# For local execution
if __name__ == "__main__":
    asyncio.run(main())
```

### Python Worker (D1 Binding)

Inside a Cloudflare Python Worker, `WorkerCloudflareD1Saver` talks to D1 directly
through the Worker's `env.DB` binding instead of the REST API -- no network
round-trip to the Cloudflare API and no API token required. It requires the
optional `worker` extra:

```bash
pip install 'langgraph-checkpoint-cloudflare-d1[worker]'
```

Use it with `graph.ainvoke(...)` / `graph.astream(...)`:

```python
from workers import WorkerEntrypoint, Response
from langgraph.graph import StateGraph, START, END
from langgraph_checkpoint_cloudflare_d1.worker import WorkerCloudflareD1Saver


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        # self.env.DB is the D1 binding configured in wrangler.jsonc
        checkpointer = WorkerCloudflareD1Saver(self.env.DB)

        builder = StateGraph(int)
        builder.add_node("add_one", lambda state: state + 1)
        builder.add_edge(START, "add_one")
        builder.add_edge("add_one", END)
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}
        result = await graph.ainvoke(3, config)
        return Response.json({"result": result})
```

The saver's synchronous methods (`get_tuple`, `list`, `put`, `put_writes`,
`delete_thread`) also work, bridged to their async counterparts via
`pyodide.ffi.run_sync()` -- the same mechanism `sqlalchemy_cloudflare_d1.SyncWorkerConnection`
uses so SQLAlchemy's sync engine can run in a Worker without `greenlet`. Call
them directly (`checkpointer.put(...)`, `checkpointer.get_tuple(...)`) when
you want synchronous code outside of a compiled graph.

**This does not make `graph.invoke()` (sync) usable inside a Worker.**
LangGraph's synchronous `Pregel` loop submits checkpoint writes to a real
`concurrent.futures.ThreadPoolExecutor` regardless of which checkpointer is
attached, and Workers/Pyodide can't spawn real OS threads --
`graph.invoke()` fails with `RuntimeError: can't start new thread` with *any*
checkpointer plugged in, this one included. `graph.ainvoke(...)` /
`graph.astream(...)` are the only graph-level entry points that work inside a
Worker.

See `examples/workers/` for a complete, runnable Worker example and
`tests/worker_tests/` for the integration tests that exercise it.

### Integration with LangGraph

To use this checkpoint saver with LangGraph, you can pass it when compiling your graph:

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_cloudflare_d1 import CloudflareD1Saver

# Create a simple graph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

# Create the checkpoint saver
checkpointer = CloudflareD1Saver(
    account_id="your-account-id",
    database_id="your-database-id",
    api_token="your-api-token",
)
checkpointer.setup()  # Create necessary tables

# Compile the graph with the checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Use the graph with checkpointing
config = {"configurable": {"thread_id": "my-thread-1"}}
result = graph.invoke(3, config)
```

## Release Notes

v0.1.6

- Added `WorkerCloudflareD1Saver` for use inside Cloudflare Python Workers via the D1 binding (`worker` extra)

v0.1.2 (2025-05-11)

- Added support for environmental variables
