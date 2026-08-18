"""Example Python Worker using CloudflareVectorizeBaseStore.

This Worker exercises the Vectorize/Workers-AI-binding-backed LangGraph
BaseStore end to end: put/get/delete/search, plus a StateGraph compiled
with the store attached, exercising it from inside a node (the store's
real intended usage pattern) -- all through the `VECTORIZE`/`AI` D1
bindings, with no Cloudflare REST API calls or API token involved.

Note: Python Workers are currently in beta.
"""

# MARK: - Imports
import uuid

from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langgraph.graph import END, START, StateGraph
from workers import Response, WorkerEntrypoint

from langmem_cloudflare_vectorize import CloudflareVectorizeBaseStore

EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5"

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
            elif path == "store-put":
                return await self.handle_put(request)
            elif path == "store-get":
                return await self.handle_get(request)
            elif path == "store-delete":
                return await self.handle_delete(request)
            elif path == "store-search":
                return await self.handle_search(request)
            elif path == "store-graph":
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
                "name": "langmem-cloudflare-vectorize Worker example",
                "description": (
                    "Exercises CloudflareVectorizeBaseStore, which talks to "
                    "Vectorize and Workers AI through native Worker bindings "
                    "instead of the REST API."
                ),
                "endpoints": {
                    "/health": "GET - Vectorize/AI binding health check",
                    "/store-put": "POST - save an item",
                    "/store-get": "POST - get an item",
                    "/store-delete": "POST - delete an item",
                    "/store-search": "POST - search items within a namespace prefix",
                    "/store-graph": (
                        "POST - compile and run a StateGraph with the store "
                        "attached (store.aput/aget inside a node)"
                    ),
                },
                "sync_bridge": (
                    "Every /store-* CRUD endpoint above accepts an optional "
                    '"sync": true field in its JSON body, routing through the '
                    "store's sync methods (get/put/delete/search, bridged via "
                    "pyodide.ffi.run_sync()) instead of the async ones."
                ),
            }
        )

    def _get_store(self) -> CloudflareVectorizeBaseStore:
        """Build a CloudflareVectorizeBaseStore bound to this Worker's bindings."""
        if not hasattr(self.env, "VECTORIZE"):
            raise RuntimeError(
                "VECTORIZE binding not configured. Add a vectorize section "
                "to wrangler.jsonc"
            )
        if not hasattr(self.env, "AI"):
            raise RuntimeError(
                "AI binding not configured. Add an ai section to wrangler.jsonc"
            )
        embeddings = CloudflareWorkersAIEmbeddings(
            binding=self.env.AI, model_name=EMBEDDING_MODEL
        )
        return CloudflareVectorizeBaseStore(
            embedding_function=embeddings, binding=self.env.VECTORIZE
        )

    # MARK: - Health

    async def handle_health(self) -> Response:
        """Confirm the Vectorize/AI bindings are reachable."""
        store = self._get_store()
        await store.asearch((), limit=1)
        return Response.json({"status": "healthy"})

    # MARK: - Store CRUD

    async def handle_put(self, request) -> Response:
        """Save an item.

        Pass `"sync": true` in the JSON body to route through the store's
        sync `put()` (bridged via `pyodide.ffi.run_sync()`) instead of
        `aput()`.
        """
        data = await request.json()
        namespace = tuple(data.get("namespace") or ["docs"])
        key = data.get("key") or str(uuid.uuid4())
        value = data.get("value") or {"text": "hello"}

        store = self._get_store()
        if data.get("sync"):
            store.put(namespace, key, value)
        else:
            await store.aput(namespace, key, value)

        return Response.json(
            {"success": True, "namespace": list(namespace), "key": key}
        )

    async def handle_get(self, request) -> Response:
        """Get an item by namespace and key.

        Pass `"sync": true` in the JSON body to route through the store's
        sync `get()` instead of `aget()`.
        """
        data = await request.json()
        namespace = tuple(data.get("namespace") or ["docs"])
        key = data["key"]

        store = self._get_store()
        if data.get("sync"):
            item = store.get(namespace, key)
        else:
            item = await store.aget(namespace, key)

        if item is None:
            return Response.json({"found": False})

        return Response.json(
            {
                "found": True,
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
            }
        )

    async def handle_delete(self, request) -> Response:
        """Delete an item by namespace and key.

        Pass `"sync": true` in the JSON body to route through the store's
        sync `delete()` instead of `adelete()`.
        """
        data = await request.json()
        namespace = tuple(data.get("namespace") or ["docs"])
        key = data["key"]

        store = self._get_store()
        if data.get("sync"):
            store.delete(namespace, key)
        else:
            await store.adelete(namespace, key)

        return Response.json({"success": True})

    async def handle_search(self, request) -> Response:
        """Search items within a namespace prefix.

        Pass `"sync": true` in the JSON body to route through the store's
        sync `search()` instead of `asearch()`.
        """
        data = await request.json()
        namespace_prefix = tuple(data.get("namespace_prefix") or [])
        query = data.get("query")
        limit = data.get("limit", 20)

        store = self._get_store()
        if data.get("sync"):
            results = store.search(namespace_prefix, query=query, limit=limit)
        else:
            results = await store.asearch(namespace_prefix, query=query, limit=limit)

        return Response.json(
            {
                "success": True,
                "count": len(results),
                "results": [
                    {
                        "namespace": list(r.namespace),
                        "key": r.key,
                        "value": r.value,
                        "score": r.score,
                    }
                    for r in results
                ],
            }
        )

    # MARK: - Full Graph Demo

    async def handle_graph(self, request) -> Response:
        """Compile a tiny StateGraph with the store attached and exercise it
        from inside a node -- store.put()/get() inside a compiled graph is
        the store's real intended usage pattern, not just standalone CRUD.

        Pass `"recall": true` in the JSON body to read a previously-written
        memory back instead of writing a new one. These are deliberately two
        separate requests/graph runs, not a write immediately followed by a
        read in the same node: Vectorize inserts are processed asynchronously
        (a mutation queue, not synchronous indexing), so a get() right after
        a put() for the same id can legitimately miss for anywhere from a
        few seconds up to tens of seconds under load -- reading back in a
        later request is the realistic usage pattern for a memory store, and
        avoids a demo that flakes on an immediate re-read.
        """
        data = await request.json() if request.method == "POST" else {}
        thread_id = data.get("thread_id") or str(uuid.uuid4())
        text = data.get("text", "graph memory example")
        recall = bool(data.get("recall"))

        store = self._get_store()

        async def remember(state: dict) -> dict:
            await store.aput(("memories", thread_id), "note", {"text": state["text"]})
            return {"text": state["text"], "remembered": None}

        async def recall_fn(state: dict) -> dict:
            found = await store.aget(("memories", thread_id), "note")
            return {
                "text": state.get("text", ""),
                "remembered": found.value if found else None,
            }

        builder = StateGraph(dict)
        node_name = "recall" if recall else "remember"
        builder.add_node(node_name, recall_fn if recall else remember)
        builder.add_edge(START, node_name)
        builder.add_edge(node_name, END)
        graph = builder.compile(store=store)

        result = await graph.ainvoke(
            {"text": text}, {"configurable": {"thread_id": thread_id}}
        )

        return Response.json(
            {
                "success": True,
                "thread_id": thread_id,
                "recall": recall,
                "result": result,
            }
        )
