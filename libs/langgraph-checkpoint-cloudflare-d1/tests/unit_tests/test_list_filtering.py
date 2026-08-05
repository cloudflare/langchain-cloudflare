"""Regression tests: sync list() must apply the same filtering as async alist().

Before the fix, CloudflareD1Saver.list() built its own WHERE clause that only
supported exact-match filtering on `thread_id`/`checkpoint_id` and silently
dropped every other filter key plus the `before` argument entirely, unlike
alist() (which correctly delegates to search_where()). A caller filtering by
metadata, or paginating with `before`, got back unfiltered results with no
error.
"""

from unittest.mock import patch

from langgraph_checkpoint_cloudflare_d1 import CloudflareD1Saver
from langgraph_checkpoint_cloudflare_d1.models import D1Response


def _make_saver() -> CloudflareD1Saver:
    saver = CloudflareD1Saver(
        account_id="acct", database_id="db", api_token="token"
    )
    saver.is_setup = True  # skip the CREATE TABLE round-trip
    return saver


def test_list_passes_metadata_filter_into_the_query() -> None:
    saver = _make_saver()
    config = {"configurable": {"thread_id": "t1"}}

    with patch.object(
        saver, "_execute_query", return_value=D1Response(success=True, result=[])
    ) as mock_execute:
        list(saver.list(config, filter={"source": "loop"}))

    query, params = mock_execute.call_args[0]
    assert "json_extract" in query
    assert "loop" in params


def test_list_passes_before_into_the_query() -> None:
    saver = _make_saver()
    config = {"configurable": {"thread_id": "t1"}}
    before = {"configurable": {"thread_id": "t1", "checkpoint_id": "c9"}}

    with patch.object(
        saver, "_execute_query", return_value=D1Response(success=True, result=[])
    ) as mock_execute:
        list(saver.list(config, before=before))

    query, params = mock_execute.call_args[0]
    assert "checkpoint_id < ?" in query
    assert "c9" in params
