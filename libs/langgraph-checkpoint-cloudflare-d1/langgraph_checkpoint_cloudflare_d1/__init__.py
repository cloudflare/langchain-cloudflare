# Import the async implementation at the top of file
import base64
import json
import random
import threading
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import requests
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

from .aio import AsyncCloudflareD1Saver
from .models import D1Response
from .utils import _metadata_predicate, search_where


class CloudflareD1Saver(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores checkpoints in a Cloudflare D1 database.

    This class provides a way to store and retrieve checkpoints using Cloudflare's D1
    database service through their REST API.

    Args:
        account_id (str): Your Cloudflare account ID.
        database_id (str): The ID of your D1 database.
        api_token (str): Your Cloudflare API token with D1 permissions.
        serde (Optional[SerializerProtocol]): The serializer to use for serializing and deserializing checkpoints.

    Examples:
        >>> from langgraph_checkpoint_cloudflare_d1 import CloudflareD1Saver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> # Create a new CloudflareD1Saver instance
        >>> checkpointer = CloudflareD1Saver(
        ...     account_id="account_id",
        ...     database_id="database_id",
        ...     api_token="api_token"
        ... )
        >>> graph = builder.compile(checkpointer=checkpointer)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> result = graph.invoke(3, config)
    """

    base_url: str
    headers: Dict[str, str]
    is_setup: bool

    def __init__(
        self,
        account_id: str,
        database_id: str,
        api_token: str,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
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
        self.is_setup = False
        self.lock = threading.Lock()

    def setup(self) -> None:
        """Set up the checkpoint database.

        Creates the necessary tables in the D1 database if they don't already exist.
        """
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

        # Execute the setup queries in a single batch
        self._execute_query(setup_query)
        self.is_setup = True

    def _execute_query(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> D1Response:
        """Execute a SQL query against the D1 database."""
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
            response = requests.post(endpoint, headers=self.headers, json=data)
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

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the D1 database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        with self.lock:
            self.setup()
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

            result = self._execute_query(query, params)

            if not result.success:
                return None

            # D1 API returns a list of results, each with their own "results" array
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

            writes_result = self._execute_query(writes_query, writes_params)

            # Initialize write collection
            all_writes: List[Tuple[str, str, Any]] = []

            if not writes_result.success:
                # Keep all_writes empty if the response failed
                pass
            else:
                result_data = writes_result.result

                # Handle result which is a list of D1QueryResult objects
                if isinstance(result_data, list) and len(result_data) > 0:
                    results = result_data[0].results
                else:
                    results = []

                for write_row in results:
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
                            continue

                    # Cast types to satisfy type checker
                    type_str = cast(str, type_)
                    value_bytes = cast(bytes, value)

                    # First get the serialized value
                    serialized_value = self.serde.loads_typed((type_str, value_bytes))

                    # Properly cast the task_id and channel to ensure type correctness
                    task_id_str = cast(str, task_id)
                    channel_str = cast(str, channel)

                    all_writes.append((task_id_str, channel_str, serialized_value))

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

                # Get checkpoint metadata
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

                # Cast to correct types for type checker
                checkpoint_metadata: CheckpointMetadata = cast(
                    CheckpointMetadata, metadata_dict
                )
                typed_writes: Optional[List[Tuple[str, str, Any]]] = cast(
                    Optional[List[Tuple[str, str, Any]]], all_writes
                )

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

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        Args:
            config: The config to use for listing checkpoints.
            filter: Filtering criteria.
            before: List checkpoints before this config.
            limit: The maximum number of checkpoints to return.

        Returns:
            Iterator[CheckpointTuple]: Iterator over checkpoint tuples.
        """
        filter = filter or {}
        thread_id = None

        # Extract thread ID from config or filter
        if (
            config
            and "configurable" in config
            and "thread_id" in config["configurable"]
        ):
            thread_id = config["configurable"]["thread_id"]
        elif filter and "thread_id" in filter:
            thread_id = filter["thread_id"]

        # Do not filter results - simply iterate and yield checkpoints
        with self.lock:
            if not thread_id:
                return

            self.setup()

            # Determine what to filter by
            filter_by = []
            params = []

            if filter:
                # Support exact match filtering on thread_id and checkpoint_id
                for key, value in filter.items():
                    if key in ["thread_id", "checkpoint_id"]:
                        filter_by.append(f"{key} = ?")
                        params.append(value)
                    else:
                        # For complex filtering, would need metadata query support
                        continue

            filter_clause = f"WHERE {' AND '.join(filter_by)}" if filter_by else ""
            limit_clause = f"LIMIT {limit}" if limit else ""

            query = f"SELECT * FROM checkpoints {filter_clause} ORDER BY checkpoint_id DESC {limit_clause}"

            response = self._execute_query(query, params)

            # Clean up the response format
            if not response.success:
                return

            # D1 API returns a list of results, each with their own "results" array
            if (
                not response.result
                or not isinstance(response.result, list)
                or len(response.result) == 0
                or not response.result[0].results
            ):
                return

            rows = response.result[0].results
            if not rows:
                return

            for row in rows:
                try:
                    # Extract standard fields
                    thread_id = row.get("thread_id")
                    checkpoint_ns = row.get("checkpoint_ns", "")
                    checkpoint_id = row.get("checkpoint_id")
                    parent_checkpoint_id = row.get("parent_checkpoint_id")
                    type_ = row.get("type")
                    checkpoint = row.get("checkpoint")
                    metadata = row.get("metadata")

                    # Skip if no checkpoint data
                    if not checkpoint:
                        continue

                    # Ensure checkpoint is bytes
                    if isinstance(checkpoint, str):
                        try:
                            checkpoint = base64.b64decode(checkpoint)
                        except Exception:
                            continue

                    if not isinstance(checkpoint, bytes):
                        continue

                    # Get writes for this checkpoint
                    writes_query = "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx"
                    writes_params = [thread_id, checkpoint_ns, checkpoint_id]

                    writes_response = self._execute_query(writes_query, writes_params)

                    # Initialize write collection
                    all_writes: List[Tuple[str, str, Any]] = []

                    if not writes_response.success:
                        # Keep all_writes empty if the response failed
                        pass
                    else:
                        result_data = writes_response.result

                        # Handle result which is a list of D1QueryResult objects
                        if isinstance(result_data, list) and len(result_data) > 0:
                            results = result_data[0].results
                        else:
                            results = []

                        for write_row in results:
                            task_id = write_row.get("task_id")
                            channel = write_row.get("channel")
                            write_type = write_row.get("type")
                            value = write_row.get("value")

                            # Skip NULL/None values
                            if not value:
                                continue

                            # Decode value if it's a string (base64-encoded)
                            if value and isinstance(value, str):
                                try:
                                    value = base64.b64decode(value)
                                except Exception:
                                    continue

                            # Cast types to properly satisfy type checker
                            task_id_str = cast(str, task_id)
                            channel_str = cast(str, channel)
                            write_type_str = cast(str, write_type)
                            value_bytes = cast(bytes, value)

                            try:
                                serialized_value = self.serde.loads_typed(
                                    (write_type_str, value_bytes)
                                )
                                all_writes.append(
                                    (task_id_str, channel_str, serialized_value)
                                )
                            except Exception:
                                continue

                    # Deserialize metadata safely
                    try:
                        metadata_dict = {}
                        if (
                            metadata is not None
                            and metadata != ""
                            and isinstance(metadata, (str, bytes))
                        ):
                            metadata_dict = self.jsonplus_serde.loads(metadata)
                    except Exception:
                        metadata_dict = {}

                    # Create and yield checkpoint tuple
                    type_str = cast(str, type_)
                    checkpoint_bytes = cast(bytes, checkpoint)

                    # Ensure metadata_dict is properly typed as CheckpointMetadata
                    checkpoint_metadata: CheckpointMetadata = cast(
                        CheckpointMetadata, metadata_dict
                    )

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
                        cast(Optional[List[Tuple[str, str, Any]]], all_writes),
                    )
                except Exception:
                    continue

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

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
        with self.lock:
            self.setup()
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
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

            try:
                self._execute_query(query, params)
            except Exception:
                pass

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                }
            }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the D1 database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        with self.lock:
            self.setup()

            query = (
                "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                if all(w[0] in WRITES_IDX_MAP for w in writes)
                else "INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )

            # Execute with many batches due to D1 limitations
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

                self._execute_query(query, params)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        with self.lock:
            self.setup()

            # Delete checkpoints
            checkpoints_query = "DELETE FROM checkpoints WHERE thread_id = ?"
            self._execute_query(checkpoints_query, [str(thread_id)])

            # Delete writes
            writes_query = "DELETE FROM writes WHERE thread_id = ?"
            self._execute_query(writes_query, [str(thread_id)])

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        Note:
            This async method is not supported by CloudflareD1Saver.
            Use get_tuple() instead, or consider using AsyncCloudflareD1Saver.
        """
        raise NotImplementedError(
            "The CloudflareD1Saver does not support async methods. "
            "Consider using AsyncCloudflareD1Saver instead."
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        Note:
            This async method is not supported by CloudflareD1Saver.
            Use list() instead, or consider using AsyncCloudflareD1Saver.
        """
        raise NotImplementedError(
            "The CloudflareD1Saver does not support async methods. "
            "Consider using AsyncCloudflareD1Saver instead."
        )
        # This is to satisfy the type signature, it will never be executed
        if False:
            yield await self.aget_tuple(cast(RunnableConfig, config))

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        Note:
            This async method is not supported by CloudflareD1Saver.
            Use put() instead, or consider using AsyncCloudflareD1Saver.
        """
        raise NotImplementedError(
            "The CloudflareD1Saver does not support async methods. "
            "Consider using AsyncCloudflareD1Saver instead."
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        Note:
            This async method is not supported by CloudflareD1Saver.
            Use put_writes() instead, or consider using AsyncCloudflareD1Saver.
        """
        raise NotImplementedError(
            "The CloudflareD1Saver does not support async methods. "
            "Consider using AsyncCloudflareD1Saver instead."
        )

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


# Export the main classes and utils
__all__ = [
    "CloudflareD1Saver",
    "AsyncCloudflareD1Saver",
    "search_where",
    "_metadata_predicate",
]
