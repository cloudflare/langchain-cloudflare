# Usage

This package provides both synchronous and asynchronous interfaces for saving and retrieving LangGraph checkpoints in Cloudflare D1.

# Synchronous

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
    account_id=account_id,
    database_id=database_id,
    api_token=api_token
) as checkpointer:
    # Setup the database tables (idempotent operation)
    checkpointer.setup()
    
    # Sample checkpoint data
    checkpoint = {
        "v": 2,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
                "__start__": 1
            },
            "node": {
                "start:node": 2
            }
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

You can also use a connection string to initialize the saver:

```python
from langgraph_checkpoint_cloudflare_d1 import CloudflareD1Saver

# Connection string format: "account_id:database_id:api_token"
conn_string = "your-account-id:your-database-id:your-api-token" 

# Use the saver with a connection string
with CloudflareD1Saver.from_conn_string(conn_string) as checkpointer:
    # Your code here
    checkpointer.setup()
    # ...

# Async

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
        account_id=account_id,
        database_id=database_id,
        api_token=api_token
    ) as checkpointer:
        # Sample checkpoint data
        checkpoint = {
            "v": 2,
            "ts": "2024-07-31T20:14:19.804150+00:00",
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "channel_values": {
                "my_key": "meow",
                "node": "node"
            },
            "channel_versions": {
                "__start__": 2,
                "my_key": 3,
                "start:node": 3,
                "node": 3
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {
                    "__start__": 1
                },
                "node": {
                    "start:node": 2
                }
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

You can also use a connection string to initialize the async saver:

```python
from langgraph_checkpoint_cloudflare_d1 import AsyncCloudflareD1Saver
import asyncio

async def main():
    # Connection string format: "account_id:database_id:api_token"
    conn_string = "your-account-id:your-database-id:your-api-token" 
    
    # Use the async saver with a connection string
    async with await AsyncCloudflareD1Saver.from_conn_string(conn_string) as checkpointer:
        # Your code here
        # Setup is automatic when using context manager
        # ...

# For local execution
if __name__ == "__main__":
    asyncio.run(main())
```

## Integration with LangGraph

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
    api_token="your-api-token"
)
checkpointer.setup()  # Create necessary tables

# Compile the graph with the checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Use the graph with checkpointing
config = {"configurable": {"thread_id": "my-thread-1"}}
result = graph.invoke(3, config)
```