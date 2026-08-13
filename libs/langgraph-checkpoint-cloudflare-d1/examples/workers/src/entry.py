"""Example Python Worker using WorkerCloudflareD1Saver.

This Worker exercises the D1-binding-backed LangGraph checkpointer end to
end: table setup, put/put_writes/get_tuple/list/delete_thread, and a full
StateGraph compiled with the saver as its checkpointer -- all through the
`DB` D1 binding, with no Cloudflare REST API calls or API token involved.

Note: Python Workers are currently in beta.
"""

# MARK: - Imports
import uuid

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph
from workers import Response, WorkerEntrypoint

from langgraph_checkpoint_cloudflare_d1.worker import WorkerCloudflareD1Saver

# MARK: - Worker Entrypoint


class Default(WorkerEntrypoint):
    """Default Worker entrypoint that handles HTTP requests."""

    async def fetch(self, request):
        """Route incoming HTTP requests to the matching handler."""
        url = request.url
        path = url.split("/")[-1].split("?")[0] if "/" in url else ""

        try:
            if path in ("", "index"):
                return self._index()
            elif path == "health":
                return await self.handle_health()
            elif path == "checkpointer-put":
                return await self.handle_put(request)
            elif path == "checkpointer-put-writes":
                return await self.handle_put_writes(request)
            elif path == "checkpointer-get-tuple":
                return await self.handle_get_tuple(request)
            elif path == "checkpointer-list":
                return await self.handle_list(request)
            elif path == "checkpointer-delete-thread":
                return await self.handle_delete_thread(request)
            elif path == "checkpointer-graph":
                return await self.handle_graph(request)

            return Response.json({"error": "not found", "path": path}, status=404)
        except Exception as e:
            return Response.json(
                {"error": str(e), "error_type": type(e).__name__}, status=500
            )

    # MARK: - Helpers

    def _index(self) -> Response:
        """Return API documentation."""
        return Response.json(
            {
                "name": "langgraph-checkpoint-cloudflare-d1 Worker example",
                "description": (
                    "Exercises WorkerCloudflareD1Saver, which talks to D1 "
                    "through the native Worker binding instead of the REST API."
                ),
                "endpoints": {
                    "/health": "GET - D1 binding + checkpoint table health check",
                    "/checkpointer-put": "POST - save a checkpoint",
                    "/checkpointer-put-writes": "POST - save writes for a checkpoint",
                    "/checkpointer-get-tuple": "POST - get a checkpoint for a thread",
                    "/checkpointer-list": "POST - list checkpoints for a thread",
                    "/checkpointer-delete-thread": (
                        "POST - delete a thread's checkpoints"
                    ),
                    "/checkpointer-graph": (
                        "POST - compile and run a StateGraph using the "
                        "checkpointer (graph.ainvoke; graph.invoke can't run "
                        "in a Worker with any checkpointer, see "
                        "WorkerCloudflareD1Saver's docstring)"
                    ),
                },
                "sync_bridge": (
                    "Every /checkpointer-* CRUD endpoint above accepts an "
                    'optional "sync": true field in its JSON body, routing '
                    "through the saver's sync methods "
                    "(put/get_tuple/put_writes/list/delete_thread, bridged "
                    "via pyodide.ffi.run_sync()) instead of the async ones. "
                    "There is no sync equivalent of /checkpointer-graph."
                ),
            }
        )

    def _get_saver(self) -> WorkerCloudflareD1Saver:
        """Build a WorkerCloudflareD1Saver bound to this Worker's D1 binding."""
        if not hasattr(self.env, "DB"):
            raise RuntimeError(
                "DB binding not configured. Add a d1_databases section to "
                "wrangler.jsonc"
            )
        return WorkerCloudflareD1Saver(self.env.DB)

    # MARK: - Health

    async def handle_health(self) -> Response:
        """Confirm the D1 binding is reachable and the checkpoint tables exist."""
        saver = self._get_saver()
        await saver.setup()
        return Response.json({"status": "healthy", "is_setup": saver.is_setup})

    # MARK: - Checkpoint CRUD

    async def handle_put(self, request) -> Response:
        """Save a fresh checkpoint for a thread.

        Pass `"sync": true` in the JSON body to route through the saver's
        sync `put()` (bridged via `pyodide.ffi.run_sync()`) instead of
        `aput()`.
        """
        data = await request.json()
        thread_id = data.get("thread_id") or str(uuid.uuid4())
        checkpoint_ns = data.get("checkpoint_ns", "")
        metadata = data.get("metadata") or {
            "source": "input",
            "step": 1,
            "parents": {},
        }

        saver = self._get_saver()
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }
        checkpoint = empty_checkpoint()
        if data.get("sync"):
            new_config = saver.put(config, checkpoint, metadata, {})
        else:
            new_config = await saver.aput(config, checkpoint, metadata, {})

        return Response.json(
            {
                "success": True,
                "thread_id": thread_id,
                "checkpoint_id": new_config["configurable"]["checkpoint_id"],
            }
        )

    async def handle_put_writes(self, request) -> Response:
        """Save intermediate writes linked to an existing checkpoint.

        Pass `"sync": true` in the JSON body to route through the saver's
        sync `put_writes()` instead of `aput_writes()`.
        """
        data = await request.json()
        thread_id = data["thread_id"]
        checkpoint_id = data["checkpoint_id"]
        checkpoint_ns = data.get("checkpoint_ns", "")
        task_id = data.get("task_id", "task-1")
        writes = [(w["channel"], w["value"]) for w in data.get("writes", [])]

        saver = self._get_saver()
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        if data.get("sync"):
            saver.put_writes(config, writes, task_id)
        else:
            await saver.aput_writes(config, writes, task_id)

        return Response.json({"success": True, "count": len(writes)})

    async def handle_get_tuple(self, request) -> Response:
        """Get the latest (or a specific) checkpoint for a thread.

        Pass `"sync": true` in the JSON body to route through the saver's
        sync `get_tuple()` instead of `aget_tuple()`.
        """
        data = await request.json()
        thread_id = data["thread_id"]
        configurable = {
            "thread_id": thread_id,
            "checkpoint_ns": data.get("checkpoint_ns", ""),
        }
        if data.get("checkpoint_id"):
            configurable["checkpoint_id"] = data["checkpoint_id"]

        saver = self._get_saver()
        if data.get("sync"):
            tup = saver.get_tuple({"configurable": configurable})
        else:
            tup = await saver.aget_tuple({"configurable": configurable})

        if tup is None:
            return Response.json({"found": False})

        parent_configurable = (tup.parent_config or {}).get("configurable", {})
        return Response.json(
            {
                "found": True,
                "thread_id": thread_id,
                "checkpoint_id": tup.checkpoint["id"],
                "metadata": tup.metadata,
                "parent_checkpoint_id": parent_configurable.get("checkpoint_id"),
                "pending_writes": [
                    [task_id, channel, value]
                    for task_id, channel, value in (tup.pending_writes or [])
                ],
            }
        )

    async def handle_list(self, request) -> Response:
        """List checkpoints for a thread, newest first.

        Pass `"sync": true` in the JSON body to route through the saver's
        sync `list()` instead of `alist()`.
        """
        data = await request.json()
        thread_id = data["thread_id"]
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": data.get("checkpoint_ns", ""),
            }
        }

        saver = self._get_saver()
        checkpoints = []
        if data.get("sync"):
            for tup in saver.list(config, limit=data.get("limit")):
                checkpoints.append(
                    {"checkpoint_id": tup.checkpoint["id"], "metadata": tup.metadata}
                )
        else:
            async for tup in saver.alist(config, limit=data.get("limit")):
                checkpoints.append(
                    {"checkpoint_id": tup.checkpoint["id"], "metadata": tup.metadata}
                )

        return Response.json(
            {"success": True, "count": len(checkpoints), "checkpoints": checkpoints}
        )

    async def handle_delete_thread(self, request) -> Response:
        """Delete all checkpoints and writes for a thread.

        Pass `"sync": true` in the JSON body to route through the saver's
        sync `delete_thread()` instead of `adelete_thread()`.
        """
        data = await request.json()
        thread_id = data["thread_id"]

        saver = self._get_saver()
        if data.get("sync"):
            saver.delete_thread(thread_id)
        else:
            await saver.adelete_thread(thread_id)

        return Response.json({"success": True, "thread_id": thread_id})

    # MARK: - Full Graph Demo

    async def handle_graph(self, request) -> Response:
        """Compile a tiny StateGraph with WorkerCloudflareD1Saver and run it.

        This is the end-to-end path the checkpointer is actually built for:
        `builder.compile(checkpointer=saver)` followed by `graph.ainvoke(...)`.
        There is no sync equivalent here (`graph.invoke()`, not this saver's
        sync bridge): LangGraph's synchronous Pregel loop always submits
        checkpoint writes to a real `concurrent.futures.ThreadPoolExecutor`,
        which Workers/Pyodide can't create, regardless of which checkpointer
        is attached. See `WorkerCloudflareD1Saver`'s docstring.
        """
        data = await request.json() if request.method == "POST" else {}
        thread_id = data.get("thread_id") or str(uuid.uuid4())
        start_value = data.get("value", 1)

        def add_one(state: int) -> int:
            return state + 1

        builder = StateGraph(int)
        builder.add_node("add_one", add_one)
        builder.add_edge(START, "add_one")
        builder.add_edge("add_one", END)

        saver = self._get_saver()
        graph = builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(start_value, config)
        state = await graph.aget_state(config)

        return Response.json(
            {
                "success": True,
                "thread_id": thread_id,
                "input": start_value,
                "result": result,
                "checkpoint_id": state.config["configurable"]["checkpoint_id"],
            }
        )
