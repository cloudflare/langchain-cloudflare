import base64
import json
from typing import Any, Dict, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import get_checkpoint_id


def decode_metadata_blob(
    metadata: Any, default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Decode a `checkpoints.metadata` BLOB value into a metadata dict.

    The column is written as base64-encoded JSON bytes (see the bytes-encoding
    branch in each saver's query-param formatting), and both the D1 REST API
    and the Worker binding return BLOB columns as base64 strings -- so callers
    must base64-decode before parsing as JSON.
    """
    if default is None:
        default = {}
    if metadata is None or metadata == "":
        return dict(default)
    try:
        if isinstance(metadata, str):
            try:
                metadata = base64.b64decode(metadata)
            except Exception:
                pass
        if isinstance(metadata, bytes):
            metadata = metadata.decode("utf-8")
        return json.loads(metadata)
    except Exception:
        return dict(default)


def search_where(
    config: Optional[RunnableConfig],
    filter: Optional[Dict[str, Any]],
    before: Optional[RunnableConfig] = None,
) -> Tuple[str, Sequence[Any]]:
    """Return WHERE clause predicates for search() given metadata filter
    and `before` config."""
    wheres = []
    param_values = []

    # construct predicate for config filter
    if config is not None:
        wheres.append("thread_id = ?")
        param_values.append(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns")
        if checkpoint_ns is not None:
            wheres.append("checkpoint_ns = ?")
            param_values.append(checkpoint_ns)

        if checkpoint_id := get_checkpoint_id(config):
            wheres.append("checkpoint_id = ?")
            param_values.append(checkpoint_id)

    # construct predicate for metadata filter
    if filter:
        metadata_predicates, metadata_values = _metadata_predicate(filter)
        wheres.extend(metadata_predicates)
        param_values.extend(metadata_values)

    # construct predicate for `before`
    if before is not None:
        wheres.append("checkpoint_id < ?")
        param_values.append(get_checkpoint_id(before))

    return ("WHERE " + " AND ".join(wheres) if wheres else "", param_values)


def _metadata_predicate(
    metadata_filter: Dict[str, Any],
) -> Tuple[Sequence[str], Sequence[Any]]:
    """Return WHERE clause predicates for search() given metadata filter."""

    def _where_value(query_value: Any) -> Tuple[str, Any]:
        """Return tuple of operator and value for WHERE clause predicate."""
        if query_value is None:
            return ("IS ?", None)
        elif isinstance(query_value, bool):
            # Must be checked before the int branch: bool is a subclass of int
            # in Python, so `isinstance(True, int)` is also True.
            return ("= ?", 1 if query_value else 0)
        elif (
            isinstance(query_value, str)
            or isinstance(query_value, int)
            or isinstance(query_value, float)
        ):
            return ("= ?", query_value)
        elif isinstance(query_value, dict) or isinstance(query_value, list):
            # query value for JSON object cannot have trailing space after
            # separators (, :)
            # SQLite json_extract() returns JSON string without whitespace
            return ("= ?", json.dumps(query_value, separators=(",", ":")))
        else:
            return ("= ?", str(query_value))

    predicates = []
    param_values = []

    # process metadata query
    for query_key, query_value in metadata_filter.items():
        operator, param_value = _where_value(query_value)
        predicates.append(
            f"json_extract(CAST(metadata AS TEXT), '$.{query_key}') {operator}"
        )
        param_values.append(param_value)

    return (predicates, param_values)
