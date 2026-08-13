"""Live integration tests for the D1 REST API checkpoint savers.

Requires CF_ACCOUNT_ID, CF_D1_DATABASE_ID, and CF_D1_API_TOKEN (or
TEST_CF_API_TOKEN / CF_API_TOKEN) in the repo-root .env file -- see
tests/integration_tests/conftest.py.

These exercise CloudflareD1Saver (sync) and AsyncCloudflareD1Saver (async)
against the real Cloudflare D1 REST API. tests/worker_tests/ covers the
equivalent surface for WorkerCloudflareD1Saver (the D1 Worker-binding saver)
against a pywrangler dev server.
"""

from typing import Tuple

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph

from langgraph_checkpoint_cloudflare_d1 import AsyncCloudflareD1Saver, CloudflareD1Saver

# MARK: - Sync REST Saver


class TestCloudflareD1Saver:
    """Tests for the synchronous, REST API-backed CloudflareD1Saver."""

    @pytest.fixture
    def saver(self, d1_credentials: Tuple[str, str, str]) -> CloudflareD1Saver:
        account_id, database_id, api_token = d1_credentials
        return CloudflareD1Saver(
            account_id=account_id, database_id=database_id, api_token=api_token
        )

    def test_put_and_get_tuple_round_trips_checkpoint(
        self, saver: CloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()

        new_config = saver.put(
            config, checkpoint, {"source": "input", "step": 1, "parents": {}}, {}
        )
        assert new_config["configurable"]["checkpoint_id"] == checkpoint["id"]

        try:
            tup = saver.get_tuple(new_config)
            assert tup is not None
            assert tup.checkpoint["id"] == checkpoint["id"]
            assert tup.metadata["step"] == 1
        finally:
            saver.delete_thread(thread_id)

    def test_put_writes_are_returned_as_pending_writes(
        self, saver: CloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()
        new_config = saver.put(config, checkpoint, {"step": 1}, {})

        saver.put_writes(new_config, [("channel1", {"foo": "bar"})], task_id="task-1")

        try:
            tup = saver.get_tuple(new_config)
            assert tup is not None
            assert tup.pending_writes == [("task-1", "channel1", {"foo": "bar"})]
        finally:
            saver.delete_thread(thread_id)

    def test_list_decodes_metadata_and_respects_limit(
        self, saver: CloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        for step in range(3):
            config = saver.put(
                config,
                empty_checkpoint(),
                {"source": "loop", "step": step, "parents": {}},
                {},
            )

        try:
            list_config = {"configurable": {"thread_id": thread_id}}
            results = list(saver.list(list_config, limit=2))
            assert len(results) == 2
            assert results[0].metadata["step"] == 2
            assert results[1].metadata["step"] == 1
        finally:
            saver.delete_thread(thread_id)

    def test_delete_thread_removes_checkpoints(
        self, saver: CloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        new_config = saver.put(config, empty_checkpoint(), {"step": 1}, {})

        saver.delete_thread(thread_id)

        assert saver.get_tuple(new_config) is None

    def test_graph_round_trip_with_sync_saver(
        self, saver: CloudflareD1Saver, thread_id: str
    ) -> None:
        def add_one(state: int) -> int:
            return state + 1

        builder = StateGraph(int)
        builder.add_node("add_one", add_one)
        builder.add_edge(START, "add_one")
        builder.add_edge("add_one", END)
        graph = builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = graph.invoke(3, config)
            assert result == 4
        finally:
            saver.delete_thread(thread_id)


# MARK: - Async REST Saver


class TestAsyncCloudflareD1Saver:
    """Tests for the async, REST API-backed AsyncCloudflareD1Saver."""

    @pytest.fixture
    async def saver(self, d1_credentials: Tuple[str, str, str]):
        account_id, database_id, api_token = d1_credentials
        async with AsyncCloudflareD1Saver.from_connection_params(
            account_id=account_id, database_id=database_id, api_token=api_token
        ) as saver:
            yield saver

    async def test_aput_and_aget_tuple_round_trips_checkpoint(
        self, saver: AsyncCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()

        new_config = await saver.aput(
            config, checkpoint, {"source": "input", "step": 1, "parents": {}}, {}
        )
        assert new_config["configurable"]["checkpoint_id"] == checkpoint["id"]

        try:
            tup = await saver.aget_tuple(new_config)
            assert tup is not None
            assert tup.checkpoint["id"] == checkpoint["id"]
            assert tup.metadata["step"] == 1
        finally:
            await saver.adelete_thread(thread_id)

    async def test_aput_writes_are_returned_as_pending_writes(
        self, saver: AsyncCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()
        new_config = await saver.aput(config, checkpoint, {"step": 1}, {})

        await saver.aput_writes(
            new_config, [("channel1", {"foo": "bar"})], task_id="task-1"
        )

        try:
            tup = await saver.aget_tuple(new_config)
            assert tup is not None
            assert tup.pending_writes == [("task-1", "channel1", {"foo": "bar"})]
        finally:
            await saver.adelete_thread(thread_id)

    async def test_alist_decodes_metadata_and_respects_limit(
        self, saver: AsyncCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        for step in range(3):
            config = await saver.aput(
                config,
                empty_checkpoint(),
                {"source": "loop", "step": step, "parents": {}},
                {},
            )

        try:
            list_config = {"configurable": {"thread_id": thread_id}}
            results = [tup async for tup in saver.alist(list_config, limit=2)]
            assert len(results) == 2
            assert results[0].metadata["step"] == 2
            assert results[1].metadata["step"] == 1
        finally:
            await saver.adelete_thread(thread_id)

    async def test_adelete_thread_removes_checkpoints(
        self, saver: AsyncCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        new_config = await saver.aput(config, empty_checkpoint(), {"step": 1}, {})

        await saver.adelete_thread(thread_id)

        assert await saver.aget_tuple(new_config) is None

    async def test_graph_round_trip_with_async_saver(
        self, saver: AsyncCloudflareD1Saver, thread_id: str
    ) -> None:
        def add_one(state: int) -> int:
            return state + 1

        builder = StateGraph(int)
        builder.add_node("add_one", add_one)
        builder.add_edge(START, "add_one")
        builder.add_edge("add_one", END)
        graph = builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = await graph.ainvoke(3, config)
            assert result == 4
        finally:
            await saver.adelete_thread(thread_id)
