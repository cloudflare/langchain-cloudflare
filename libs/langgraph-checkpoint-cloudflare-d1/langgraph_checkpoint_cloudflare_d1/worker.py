"""Checkpoint saver for Cloudflare D1 using the native Python Worker binding.

This module only works inside a Cloudflare Python Worker, where a D1 binding
(e.g. ``env.DB``) is available. Unlike
:class:`~langgraph_checkpoint_cloudflare_d1.CloudflareD1Saver` and
:class:`~langgraph_checkpoint_cloudflare_d1.aio.AsyncCloudflareD1Saver`, which
talk to D1 over the Cloudflare REST API, :class:`WorkerCloudflareD1Saver`
talks to D1 directly through the binding via
``sqlalchemy_cloudflare_d1.WorkerConnection`` -- no network round-trip to the
Cloudflare API, no API token required.

Requires the optional ``worker`` extra:

    pip install 'langgraph-checkpoint-cloudflare-d1[worker]'
"""

# MARK: - Imports
import base64
import json
import logging
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from .models import D1QueryResult, D1Response
from .utils import decode_metadata_blob, search_where

logger = logging.getLogger(__name__)

T = TypeVar("T")

# MARK: - Schema

_CHECKPOINTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)
"""

_WRITES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
)
"""


# MARK: - Worker Checkpoint Saver


class WorkerCloudflareD1Saver(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores checkpoints in D1 via a Worker binding.

    The async methods (`aget_tuple`, `alist`, `aput`, `aput_writes`,
    `adelete_thread`) are the native path -- use them with `graph.ainvoke(...)`
    / `graph.astream(...)`.

    The synchronous `BaseCheckpointSaver` methods (`get_tuple`, `list`, `put`,
    `put_writes`, `delete_thread`) are also implemented, for callers that want
    to use this saver directly (outside of a compiled graph) without `await`,
    by bridging to their async counterparts with `pyodide.ffi.run_sync()` --
    the same mechanism `sqlalchemy_cloudflare_d1.SyncWorkerConnection` uses so
    SQLAlchemy's sync engine can work in a Worker without `greenlet`. This only
    works while called from inside a live Worker request (e.g. a `fetch()`
    handler); calling a sync method outside a Worker runtime raises
    `NotImplementedError`.

    That sync bridge does **not** make `graph.invoke()` (sync) usable inside a
    Worker, though: LangGraph's synchronous `Pregel` loop submits checkpoint
    writes to a real `concurrent.futures.ThreadPoolExecutor`
    (`langgraph.pregel._executor.BackgroundExecutor`) regardless of which
    checkpointer is attached, and Workers/Pyodide cannot spawn real OS threads
    -- `graph.invoke()` fails with `RuntimeError: can't start new thread` with
    *any* checkpointer plugged in, not just this one. `graph.ainvoke(...)` /
    `graph.astream(...)` are the only graph-level entry points that work here;
    the sync bridge is for calling this saver's own methods directly.

    Args:
        d1_binding: The D1 database binding from the Worker environment
            (e.g. `self.env.DB` inside a `WorkerEntrypoint.fetch` handler).
        serde (Optional[SerializerProtocol]): The serializer to use for serializing and
            deserializing checkpoints.
        enable_logging (bool): Whether to enable logging. Defaults to False.

    Examples:
        >>> from workers import WorkerEntrypoint, Response
        >>> from langgraph_checkpoint_cloudflare_d1.worker import (
        ...     WorkerCloudflareD1Saver,
        ... )
        >>>
        >>> class Default(WorkerEntrypoint):
        ...     async def fetch(self, request):
        ...         checkpointer = WorkerCloudflareD1Saver(self.env.DB)
        ...         graph = builder.compile(checkpointer=checkpointer)
        ...         config = {"configurable": {"thread_id": "1"}}
        ...         result = await graph.ainvoke(3, config)
        ...         return Response.json(result)
    """

    is_setup: bool
    enable_logging: bool

    # MARK: - Initialization

    def __init__(
        self,
        d1_binding: Any,
        *,
        serde: Optional[SerializerProtocol] = None,
        enable_logging: bool = False,
    ) -> None:
        super().__init__(serde=serde)
        self.enable_logging = enable_logging

        try:
            from sqlalchemy_cloudflare_d1 import WorkerConnection
        except ImportError as e:
            raise ImportError(
                "WorkerCloudflareD1Saver requires the sqlalchemy-cloudflare-d1 "
                "package's WorkerConnection, which is only usable inside a "
                "Cloudflare Python Worker. Install it with: "
                "pip install 'langgraph-checkpoint-cloudflare-d1[worker]'"
            ) from e

        self._connection = WorkerConnection(d1_binding)
        self.is_setup = False

    # MARK: - Sync Bridge

    def _run_sync(self, make_coro: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Run an async checkpoint-saver coroutine synchronously.

        Bridges to async the same way `sqlalchemy_cloudflare_d1.SyncWorkerConnection`
        does for SQLAlchemy's sync engine: `pyodide.ffi.run_sync()` drives the
        Worker's single-threaded event loop to completion rather than blocking
        a thread. It's only importable -- and only works -- while called from
        inside a live Cloudflare Python Worker request, not a plain CPython
        process.

        Takes a zero-arg thunk rather than an already-built coroutine so we
        never construct (and leave dangling/unawaited) a coroutine object when
        `run_sync()` turns out not to be available.
        """
        try:
            from pyodide.ffi import run_sync  # type: ignore[import-not-found]
        except ImportError as e:
            raise NotImplementedError(
                "Synchronous WorkerCloudflareD1Saver methods require Pyodide's "
                "run_sync(), which is only available inside a Cloudflare Python "
                "Worker. Use the async methods (aget_tuple, alist, aput, "
                "aput_writes, adelete_thread) instead, or call this saver "
                "through graph.ainvoke(...) / graph.astream(...)."
            ) from e
        return run_sync(make_coro())  # type: ignore[no-any-return]

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from D1 synchronously.

        Bridges to `aget_tuple` via `pyodide.ffi.run_sync()`; only callable
        from inside a live Worker request.
        """
        return self._run_sync(lambda: self.aget_tuple(config))

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from D1 synchronously, newest first.

        Bridges to `alist` via `pyodide.ffi.run_sync()`; only callable from
        inside a live Worker request. Unlike `alist`, this fully materializes
        the results before returning -- an async generator can't be driven
        lazily through `run_sync()`, only awaited to completion.
        """

        async def _collect() -> List[CheckpointTuple]:
            return [
                tup
                async for tup in self.alist(
                    config, filter=filter, before=before, limit=limit
                )
            ]

        return iter(self._run_sync(_collect))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to D1 synchronously.

        Bridges to `aput` via `pyodide.ffi.run_sync()`; only callable from
        inside a live Worker request.
        """
        return self._run_sync(
            lambda: self.aput(config, checkpoint, metadata, new_versions)
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint synchronously.

        Bridges to `aput_writes` via `pyodide.ffi.run_sync()`; only callable
        from inside a live Worker request.
        """
        self._run_sync(lambda: self.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread synchronously.

        Bridges to `adelete_thread` via `pyodide.ffi.run_sync()`; only
        callable from inside a live Worker request.
        """
        self._run_sync(lambda: self.adelete_thread(thread_id))

    # MARK: - Setup

    async def setup(self) -> None:
        """Create the checkpoint tables in D1 if they don't already exist.

        Called automatically when needed; users should not call this directly.
        """
        if self.is_setup:
            return

        # D1's binding `prepare()` compiles a single statement, unlike the REST
        # `/query` endpoint which accepts a multi-statement batch -- so each
        # CREATE TABLE is issued as its own call.
        await self._execute_query(_CHECKPOINTS_TABLE_SQL)
        await self._execute_query(_WRITES_TABLE_SQL)
        self.is_setup = True

    # MARK: - Query Execution

    async def _execute_query(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> D1Response:
        """Execute a SQL statement against D1 through the Worker binding.

        Returns the same `D1Response` shape as the REST-based savers so the
        row-parsing logic in `aget_tuple`/`alist` can be shared verbatim.
        """
        formatted_params: List[Any] = []
        if params:
            for p in params:
                if isinstance(p, (dict, list)):
                    formatted_params.append(json.dumps(p, separators=(",", ":")))
                elif isinstance(p, bytes):
                    # Encode byte objects as base64 strings, matching the REST
                    # savers -- D1's binding doesn't reliably round-trip raw
                    # Python `bytes` through the Pyodide/JS boundary for BLOB
                    # columns, so checkpoints/writes are stored as base64 text.
                    formatted_params.append(base64.b64encode(p).decode("utf-8"))
                else:
                    formatted_params.append(p)

        cursor = self._connection.cursor()
        try:
            await cursor.execute_async(query, formatted_params if params else None)
            columns = [desc[0] for desc in (cursor.description or [])]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return D1Response(
                success=True,
                result=[D1QueryResult(results=rows, success=True)],
            )
        except Exception as e:
            if self.enable_logging:
                logger.error(
                    f"D1 Worker binding query failed: {type(e).__name__}: {e}\n"
                    f"Query: {query[:200]}..."
                )
            return D1Response(success=False)
        finally:
            cursor.close()

    # MARK: - Read Operations

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from D1 asynchronously.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if
            no matching checkpoint was found.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        thread_id = config["configurable"]["thread_id"]
        if not thread_id:
            return None

        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            query = (
                "SELECT * FROM checkpoints WHERE thread_id = ? "
                "AND checkpoint_ns = ? AND checkpoint_id = ?"
            )
            params = [thread_id, checkpoint_ns, checkpoint_id]
        else:
            query = (
                "SELECT * FROM checkpoints WHERE thread_id = ? "
                "AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1"
            )
            params = [thread_id, checkpoint_ns]

        result = await self._execute_query(query, params)

        if not result.success:
            return None

        rows = result.get_rows()
        if not rows:
            return None

        row = rows[0]

        thread_id = row.get("thread_id")
        checkpoint_id = row.get("checkpoint_id")
        parent_checkpoint_id = row.get("parent_checkpoint_id")
        type_ = row.get("type")
        checkpoint = row.get("checkpoint")
        metadata = row.get("metadata")

        if not get_checkpoint_id(config):
            config = {
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": checkpoint_id,
                }
            }

        writes_query = (
            "SELECT task_id, channel, type, value FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
            "ORDER BY task_id, idx"
        )
        writes_params = [thread_id, checkpoint_ns, checkpoint_id]

        writes_result = await self._execute_query(writes_query, writes_params)
        writes = []

        for write_row in writes_result.get_rows():
            task_id = write_row.get("task_id")
            channel = write_row.get("channel")
            write_type = write_row.get("type")
            value = write_row.get("value")

            if not value:
                continue

            if isinstance(value, str):
                try:
                    value = base64.b64decode(value)
                except Exception:
                    continue

            write_type_str = cast(str, write_type)
            value_bytes = cast(bytes, value)
            writes.append(
                (
                    task_id,
                    channel,
                    self.serde.loads_typed((write_type_str, value_bytes)),
                )
            )

        try:
            if isinstance(checkpoint, str):
                checkpoint = base64.b64decode(checkpoint)

            type_str = cast(str, type_)
            checkpoint_bytes = cast(bytes, checkpoint)
            deserialized_checkpoint = self.serde.loads_typed(
                (type_str, checkpoint_bytes)
            )

            metadata_dict = decode_metadata_blob(metadata, default={"step": -2})
            if "step" not in metadata_dict:
                metadata_dict["step"] = -2

            checkpoint_metadata: CheckpointMetadata = cast(
                CheckpointMetadata, metadata_dict
            )
            typed_writes = cast(Optional[List[Tuple[str, str, Any]]], writes)

            return CheckpointTuple(
                config,
                deserialized_checkpoint,
                checkpoint_metadata,
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                typed_writes,
            )
        except Exception:
            return None

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from D1 asynchronously, newest first.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before this checkpoint are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching
            checkpoint tuples.
        """
        await self.setup()
        where, params = search_where(config, filter, before)

        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id,
        parent_checkpoint_id, type, checkpoint, metadata
        FROM checkpoints
        {where}
        ORDER BY checkpoint_id DESC"""

        if limit:
            query += f" LIMIT {limit}"

        result = await self._execute_query(query, params)

        if not result.success:
            return

        rows = result.get_rows()
        if not rows:
            return

        for row in rows:
            thread_id = row.get("thread_id")
            checkpoint_ns = row.get("checkpoint_ns")
            checkpoint_id = row.get("checkpoint_id")
            parent_checkpoint_id = row.get("parent_checkpoint_id")
            type_ = row.get("type")
            checkpoint = row.get("checkpoint")
            metadata = row.get("metadata")

            writes_query = (
                "SELECT task_id, channel, type, value FROM writes "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
                "AND checkpoint_id = ? ORDER BY task_id, idx"
            )
            writes_params = [thread_id, checkpoint_ns, checkpoint_id]

            writes_result = await self._execute_query(writes_query, writes_params)
            writes = []

            for write_row in writes_result.get_rows():
                task_id = write_row.get("task_id")
                channel = write_row.get("channel")
                write_type = write_row.get("type")
                value = write_row.get("value")

                if not value:
                    continue

                if isinstance(value, str):
                    try:
                        value = base64.b64decode(value)
                    except Exception:
                        pass

                if not isinstance(value, bytes):
                    continue

                write_type_str = cast(str, write_type)
                value_bytes = cast(bytes, value)
                writes.append(
                    (
                        task_id,
                        channel,
                        self.serde.loads_typed((write_type_str, value_bytes)),
                    )
                )

            if checkpoint and isinstance(checkpoint, str):
                try:
                    checkpoint = base64.b64decode(checkpoint)
                except Exception:
                    pass

            if not checkpoint or not isinstance(checkpoint, bytes):
                continue

            type_str = cast(str, type_)
            checkpoint_bytes = cast(bytes, checkpoint)

            metadata_dict = decode_metadata_blob(metadata)
            checkpoint_metadata: CheckpointMetadata = cast(
                CheckpointMetadata, metadata_dict
            )
            typed_writes = cast(Optional[List[Tuple[str, str, Any]]], writes)

            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                self.serde.loads_typed((type_str, checkpoint_bytes)),
                checkpoint_metadata,
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                typed_writes,
            )

    # MARK: - Write Operations

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to D1 asynchronously.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)

        processed_metadata = get_checkpoint_metadata(config, metadata)
        if "step" not in processed_metadata:
            processed_metadata["step"] = -2

        serialized_metadata = json.dumps(processed_metadata, ensure_ascii=False).encode(
            "utf-8", "ignore"
        )

        if not isinstance(serialized_checkpoint, bytes) and isinstance(
            serialized_checkpoint, str
        ):
            try:
                serialized_checkpoint = serialized_checkpoint.encode("utf-8")
            except Exception:
                pass

        query = (
            "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, "
            "checkpoint_id, parent_checkpoint_id, type, checkpoint, "
            "metadata) VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        params = [
            str(config["configurable"]["thread_id"]),
            checkpoint_ns,
            checkpoint["id"],
            config["configurable"].get("checkpoint_id"),
            type_,
            serialized_checkpoint,
            serialized_metadata,
        ]

        try:
            result = await self._execute_query(query, params)
            if not result.success and self.enable_logging:
                logger.error(
                    f"Failed to save checkpoint for thread_id={thread_id}, "
                    f"checkpoint_id={checkpoint['id']}: D1 query returned success=False"
                )
        except Exception as e:
            if self.enable_logging:
                logger.error(
                    f"Exception saving checkpoint for thread_id={thread_id}, "
                    f"checkpoint_id={checkpoint['id']}: {type(e).__name__}: {e}",
                    exc_info=True,
                )
            # Don't raise - allow the graph to continue but log the failure

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        await self.setup()

        query = (
            (
                "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, "
                "checkpoint_id, task_id, idx, channel, type, value) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else (
                "INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, "
                "checkpoint_id, task_id, idx, channel, type, value) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
        )

        for idx, (channel, value) in enumerate(writes):
            type_, serialized_value = self.serde.dumps_typed(value)

            if not isinstance(serialized_value, bytes) and isinstance(
                serialized_value, str
            ):
                try:
                    serialized_value = serialized_value.encode("utf-8")
                except Exception:
                    pass

            params = [
                str(config["configurable"]["thread_id"]),
                str(config["configurable"].get("checkpoint_ns", "")),
                str(config["configurable"]["checkpoint_id"]),
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                type_,
                serialized_value,
            ]

            try:
                result = await self._execute_query(query, params)
                if not result.success and self.enable_logging:
                    logger.warning(
                        f"Failed to save write for "
                        f"thread_id={config['configurable']['thread_id']}, "
                        f"checkpoint_id="
                        f"{config['configurable']['checkpoint_id']}, "
                        f"channel={channel}: D1 query returned "
                        f"success=False"
                    )
            except Exception as e:
                if self.enable_logging:
                    logger.error(
                        f"Exception saving write for "
                        f"thread_id={config['configurable']['thread_id']}, "
                        f"checkpoint_id="
                        f"{config['configurable']['checkpoint_id']}, "
                        f"channel={channel}: {type(e).__name__}: {e}"
                    )
                # Continue to next write even if this one fails

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID asynchronously.

        Args:
            thread_id: The thread ID to delete.
        """
        await self.setup()

        await self._execute_query(
            "DELETE FROM checkpoints WHERE thread_id = ?", [str(thread_id)]
        )
        await self._execute_query(
            "DELETE FROM writes WHERE thread_id = ?", [str(thread_id)]
        )

    # MARK: - Versioning

    def get_next_version(self, current: Optional[str], channel: None) -> str:
        """Generate the next version ID for a channel.

        Args:
            current (Optional[str]): The current version identifier of the channel.
            channel: Deprecated argument, kept for backwards compatibility.

        Returns:
            str: The next version identifier, which is guaranteed to be monotonically
            increasing.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"


__all__ = ["WorkerCloudflareD1Saver"]
