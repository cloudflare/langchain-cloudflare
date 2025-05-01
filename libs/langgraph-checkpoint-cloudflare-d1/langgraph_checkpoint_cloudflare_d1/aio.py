import asyncio
import base64
import json
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, TypeVar, cast

import httpx
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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ChannelProtocol

from .models import D1Response
from .utils import search_where

T = TypeVar("T")


# Helper for Python < 3.10 compatibility
async def _anext(aiterator: AsyncIterator[T]) -> T:
    """Compatibility function for Python < 3.10"""
    return await aiterator.__anext__()


class AsyncCloudflareD1Saver(BaseCheckpointSaver[str]):
    """An asynchronous checkpoint saver that stores checkpoints in a Cloudflare D1 database.

    This class provides an asynchronous interface for saving and retrieving checkpoints
    using Cloudflare's D1 database. It's designed for use in asynchronous environments and
    offers better performance for I/O-bound operations compared to synchronous alternatives.

    Attributes:
        account_id (str): Your Cloudflare account ID.
        database_id (str): The ID of your D1 database.
        api_token (str): Your Cloudflare API token with D1 permissions.
        serde (SerializerProtocol): The serializer used for encoding/decoding checkpoints.

    Note:
        Requires the [httpx](https://pypi.org/project/httpx/) package.
        Install it with `pip install httpx`.

    Examples:
        Usage within StateGraph:

        ```pycon
        >>> import asyncio
        >>>
        >>> from langgraph_checkpoint_cloudflare_d1.aio import AsyncCloudflareD1Saver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> async def main():
        >>>     builder = StateGraph(int)
        >>>     builder.add_node("add_one", lambda x: x + 1)
        >>>     builder.set_entry_point("add_one")
        >>>     builder.set_finish_point("add_one")
        >>>     async with AsyncCloudflareD1Saver.from_connection_params(
        >>>         account_id="account_id",
        >>>         database_id="database_id",
        >>>         api_token="api_token"
        >>>     ) as memory:
        >>>         graph = builder.compile(checkpointer=memory)
        >>>         coro = graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
        >>>         print(await asyncio.gather(coro))
        >>>
        >>> asyncio.run(main())
        Output: [2]
        ```
    """

    base_url: str
    headers: Dict[str, str]
    is_setup: bool
    client: httpx.AsyncClient

    def __init__(
        self,
        account_id: str,
        database_id: str,
        api_token: str,
        *,
        serde: Optional[SerializerProtocol] = None,
    ):
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.account_id = account_id
        self.database_id = database_id
        self.api_token = api_token
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.client = httpx.AsyncClient(headers=self.headers)
        self.is_setup = False

    @classmethod
    @asynccontextmanager
    async def from_connection_params(
        cls,
        account_id: str,
        database_id: str,
        api_token: str,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> AsyncIterator["AsyncCloudflareD1Saver"]:
        """Create a new AsyncCloudflareD1Saver instance from connection parameters.

        Args:
            account_id: Your Cloudflare account ID.
            database_id: The ID of your D1 database.
            api_token: Your Cloudflare API token with D1 permissions.
            serde: Optional serializer for encoding/decoding checkpoints.

        Yields:
            AsyncCloudflareD1Saver: A new AsyncCloudflareD1Saver instance.
        """
        saver = cls(
            account_id=account_id,
            database_id=database_id,
            api_token=api_token,
            serde=serde,
        )
        try:
            yield saver
        finally:
            await saver.client.aclose()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database synchronously.

        This method retrieves a checkpoint tuple from the D1 database based on the
        provided config.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Raises:
            asyncio.InvalidStateError: If called from the main thread.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCloudflareD1Saver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database synchronously.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before this checkpoint are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.

        Raises:
            asyncio.InvalidStateError: If called from the main thread.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCloudflareD1Saver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    _anext(aiter_),
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database synchronously.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint synchronously."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID synchronously.

        Args:
            thread_id: The thread ID to delete.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCloudflareD1Saver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the D1 database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        async with self.lock:
            if self.is_setup:
                return

            setup_query = """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
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
            );
            """

            await self._execute_query(setup_query)
            self.is_setup = True

    async def _execute_query(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> D1Response:
        """Execute a SQL query against the D1 database asynchronously."""
        endpoint = f"{self.base_url}/query"

        # Format params for D1 API
        formatted_params = []
        if params:
            for p in params:
                if isinstance(p, (dict, list)):
                    formatted_params.append(json.dumps(p, separators=(",", ":")))
                elif isinstance(p, bytes):
                    # Encode byte objects as base64 strings
                    formatted_params.append(base64.b64encode(p).decode("utf-8"))
                else:
                    # Pass None values directly (don't convert to "NULL")
                    formatted_params.append(p)

        data = {"sql": query, "params": formatted_params if params else []}

        try:
            response = await self.client.post(endpoint, json=data)
            response.raise_for_status()
            raw_data = response.json()

            # Parse response into Pydantic model
            try:
                # Use model_validate method in Pydantic v2, fallback to parse_obj in v1
                if hasattr(D1Response, "model_validate"):
                    return D1Response.model_validate(raw_data)
                else:
                    return D1Response(**raw_data)
            except Exception:
                # Direct dictionary to bypass model validation during debugging
                return D1Response(
                    success=raw_data.get("success", False),
                    result=raw_data.get("result"),
                )
        except Exception:
            # Fallback to a basic response if parsing fails
            return D1Response(success=False)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the D1 database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        thread_id = config["configurable"]["thread_id"]
        if not thread_id:
            return None

        # If a specific checkpoint ID is requested, get that one
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            query = "SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
            params = [thread_id, checkpoint_ns, checkpoint_id]
        else:
            # Otherwise get the most recent checkpoint for this thread
            query = "SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1"
            params = [thread_id, checkpoint_ns]

        result = await self._execute_query(query, params)

        if not result.success:
            return None

        # D1 API returns a list of results, each with their own "results" array
        # Get the first query result and extract rows
        if (
            not result.result
            or not isinstance(result.result, list)
            or len(result.result) == 0
            or not result.result[0].results
        ):
            return None

        rows = result.result[0].results
        if not rows:
            return None

        row = rows[0]

        # Row will be a dictionary with our standardized model
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

        # Get all writes associated with this checkpoint
        writes_query = "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx"
        writes_params = [thread_id, checkpoint_ns, checkpoint_id]

        writes_result = await self._execute_query(writes_query, writes_params)
        writes = []

        if (
            writes_result.success
            and writes_result.result
            and isinstance(writes_result.result, list)
            and len(writes_result.result) > 0
        ):
            writes_rows = writes_result.result[0].results
            for write_row in writes_rows:
                task_id = write_row.get("task_id")
                channel = write_row.get("channel")
                type_ = write_row.get("type")
                value = write_row.get("value")

                # Skip NULL/None values
                if not value:
                    continue

                # Decode value if it's a string (base64-encoded)
                if value and isinstance(value, str):
                    try:
                        value = base64.b64decode(value)
                    except Exception:
                        pass

                # Cast types to satisfy type checker
                type_str = cast(str, type_)
                value_bytes = cast(bytes, value)
                writes.append(
                    (task_id, channel, self.serde.loads_typed((type_str, value_bytes)))
                )

        # Deserialize checkpoint
        try:
            # Ensure checkpoint is in bytes format
            if isinstance(checkpoint, str):
                checkpoint = base64.b64decode(checkpoint)

            # Cast types to satisfy type checker
            type_str = cast(str, type_)
            checkpoint_bytes = cast(bytes, checkpoint)
            deserialized_checkpoint = self.serde.loads_typed(
                (type_str, checkpoint_bytes)
            )

            # Deserialize metadata safely
            try:
                if metadata is not None and metadata != "":
                    metadata_dict = self.jsonplus_serde.loads(metadata)
                else:
                    metadata_dict = {"step": -2}  # Default initial metadata

                # Ensure required fields are present
                if "step" not in metadata_dict:
                    metadata_dict["step"] = -2  # Default initial step value
            except Exception:
                metadata_dict = {"step": -2}  # Default with required field

            # Properly cast metadata and writes to expected types
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
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the D1 database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        await self.setup()
        where, params = search_where(config, filter, before)

        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
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

            # Get writes for this checkpoint
            writes_query = "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx"
            writes_params = [thread_id, checkpoint_ns, checkpoint_id]

            writes_result = await self._execute_query(writes_query, writes_params)
            writes = []

            writes_rows = writes_result.get_rows()
            if writes_rows:
                for write_row in writes_rows:
                    task_id = write_row.get("task_id")
                    channel = write_row.get("channel")
                    type_ = write_row.get("type")
                    value = write_row.get("value")

                    # Skip NULL/None values
                    if not value:
                        continue

                    # Decode value if it's a string (base64-encoded)
                    if value and isinstance(value, str):
                        try:
                            value = base64.b64decode(value)
                        except Exception:
                            pass

                    # Ensure value is bytes
                    if not isinstance(value, bytes):
                        continue

                    # Cast types to satisfy type checker
                    type_str = cast(str, type_)
                    value_bytes = cast(bytes, value)
                    writes.append(
                        (
                            task_id,
                            channel,
                            self.serde.loads_typed((type_str, value_bytes)),
                        )
                    )

            # Decode checkpoint if it's a string (base64-encoded)
            if checkpoint and isinstance(checkpoint, str):
                try:
                    checkpoint = base64.b64decode(checkpoint)
                except Exception:
                    pass

            # Handle None or empty checkpoint
            if not checkpoint:
                continue

            # Ensure checkpoint is bytes for the deserializer
            if not isinstance(checkpoint, bytes):
                continue

            # Cast types to satisfy type checker
            type_str = cast(str, type_)
            checkpoint_bytes = cast(bytes, checkpoint)

            # Cast metadata dict to CheckpointMetadata
            metadata_dict = (
                self.jsonplus_serde.loads(metadata) if metadata is not None else {}
            )
            checkpoint_metadata: CheckpointMetadata = cast(
                CheckpointMetadata, metadata_dict
            )

            # Properly cast writes list to the expected type
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

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the D1 database. The checkpoint is associated
        with the provided config and its parent config (if any).

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

        # Serialize data
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)

        # Ensure metadata has required fields
        processed_metadata = get_checkpoint_metadata(config, metadata)
        if "step" not in processed_metadata:
            processed_metadata["step"] = -2  # Set default value if missing

        serialized_metadata = self.jsonplus_serde.dumps(processed_metadata)

        # Ensure serialized data is bytes for BLOB columns
        if not isinstance(serialized_checkpoint, bytes):
            # Convert to bytes if needed (should not happen with proper serialization)
            if isinstance(serialized_checkpoint, str):
                try:
                    serialized_checkpoint = serialized_checkpoint.encode("utf-8")
                except Exception:
                    pass

        if not isinstance(serialized_metadata, bytes):
            # Convert to bytes if needed
            if isinstance(serialized_metadata, str):
                try:
                    serialized_metadata = serialized_metadata.encode("utf-8")
                except Exception:
                    pass

        query = "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)"
        params = [
            str(config["configurable"]["thread_id"]),
            checkpoint_ns,
            checkpoint["id"],
            config["configurable"].get("checkpoint_id"),
            type_,
            serialized_checkpoint,
            serialized_metadata,
        ]

        await self._execute_query(query, params)

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

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        await self.setup()

        query = (
            "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else "INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )

        # Execute with separate queries for each write
        for idx, (channel, value) in enumerate(writes):
            type_, serialized_value = self.serde.dumps_typed(value)

            # Ensure serialized value is bytes for BLOB column
            if not isinstance(serialized_value, bytes):
                # Convert to bytes if needed
                if isinstance(serialized_value, str):
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

            await self._execute_query(query, params)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID asynchronously.

        Args:
            thread_id: The thread ID to delete.
        """
        await self.setup()

        # Delete checkpoints
        checkpoints_query = "DELETE FROM checkpoints WHERE thread_id = ?"
        await self._execute_query(checkpoints_query, [str(thread_id)])

        # Delete writes
        writes_query = "DELETE FROM writes WHERE thread_id = ?"
        await self._execute_query(writes_query, [str(thread_id)])

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """Generate the next version ID for a channel.

        This method creates a new version identifier for a channel based on its current version.

        Args:
            current (Optional[str]): The current version identifier of the channel.
            channel (BaseChannel): The channel being versioned.

        Returns:
            str: The next version identifier, which is guaranteed to be monotonically increasing.
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
