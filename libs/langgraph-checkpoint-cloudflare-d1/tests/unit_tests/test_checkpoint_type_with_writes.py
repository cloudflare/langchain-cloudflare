"""Regression tests: a checkpoint must be decoded with its own `type`, not the
`type` of the last pending write read for it.

get_tuple(), aget_tuple() and alist() reused the `type_` variable inside the
loop over pending writes, so the checkpoint was decoded with the last write's
type. LangGraph writes `None` values (type "null") for branch triggers such as
`branch:to:<node>`, so e.g. the input checkpoint of every run came back with
`checkpoint=None`.
"""

import base64
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_checkpoint_cloudflare_d1 import AsyncCloudflareD1Saver, CloudflareD1Saver
from langgraph_checkpoint_cloudflare_d1.models import D1Response

CHECKPOINT = empty_checkpoint()
CONFIG = {
    "configurable": {
        "thread_id": "t1",
        "checkpoint_ns": "",
        "checkpoint_id": CHECKPOINT["id"],
    }
}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _response(rows: List[Dict[str, Any]]) -> D1Response:
    return D1Response.model_validate(
        {"success": True, "result": [{"results": rows, "success": True, "meta": {}}]}
    )


def _checkpoint_rows() -> List[Dict[str, Any]]:
    type_, data = JsonPlusSerializer().dumps_typed(CHECKPOINT)
    assert type_ == "msgpack"
    metadata = json.dumps({"source": "input", "step": -1}).encode()
    return [
        {
            "thread_id": "t1",
            "checkpoint_ns": "",
            "checkpoint_id": CHECKPOINT["id"],
            "parent_checkpoint_id": None,
            "type": type_,
            "checkpoint": _b64(data),
            "metadata": _b64(metadata),
        }
    ]


def _write_rows() -> List[Dict[str, Any]]:
    # Same shape LangGraph stores for the input step: the input value, then a
    # `None` branch trigger, which the serializer types as "null".
    serde = JsonPlusSerializer()
    rows = []
    for channel, value in [("n", 1), ("branch:to:inc", None)]:
        type_, data = serde.dumps_typed(value)
        rows.append(
            {
                "task_id": "task-1",
                "channel": channel,
                "type": type_,
                "value": _b64(data),
            }
        )
    assert rows[-1]["type"] == "null"
    return rows


def _responses() -> List[D1Response]:
    return [_response(_checkpoint_rows()), _response(_write_rows())]


def test_get_tuple_decodes_checkpoint_with_its_own_type() -> None:
    saver = CloudflareD1Saver(account_id="acct", database_id="db", api_token="token")
    saver.is_setup = True

    with patch.object(saver, "_execute_query", side_effect=_responses()):
        tup = saver.get_tuple(CONFIG)

    assert tup is not None
    assert tup.checkpoint == CHECKPOINT


async def test_aget_tuple_decodes_checkpoint_with_its_own_type() -> None:
    saver = AsyncCloudflareD1Saver(
        account_id="acct", database_id="db", api_token="token"
    )
    saver.is_setup = True

    with patch.object(saver, "_execute_query", AsyncMock(side_effect=_responses())):
        tup = await saver.aget_tuple(CONFIG)

    assert tup is not None
    assert tup.checkpoint == CHECKPOINT


async def test_alist_decodes_checkpoint_with_its_own_type() -> None:
    saver = AsyncCloudflareD1Saver(
        account_id="acct", database_id="db", api_token="token"
    )
    saver.is_setup = True

    with patch.object(saver, "_execute_query", AsyncMock(side_effect=_responses())):
        tuples = [
            tup async for tup in saver.alist({"configurable": {"thread_id": "t1"}})
        ]

    assert len(tuples) == 1
    assert tuples[0].checkpoint == CHECKPOINT
