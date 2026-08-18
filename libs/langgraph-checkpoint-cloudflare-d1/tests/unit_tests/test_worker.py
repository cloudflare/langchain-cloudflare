"""Unit tests for WorkerCloudflareD1Saver.

These tests exercise the Worker-binding checkpoint saver against an in-memory
fake D1 binding backed by sqlite3, so they run without Pyodide, wrangler, or
network access -- unlike tests/worker_tests/, which drive the real thing
through a pywrangler dev server.

The fake binding mimics the async `prepare(sql).bind(*params).all()` /
`raw({"columnNames": True})` surface that `sqlalchemy_cloudflare_d1.WorkerConnection`
calls against the real D1 binding.
"""

import sqlite3
import uuid
from typing import Any, Optional, Sequence

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langgraph_checkpoint_cloudflare_d1 import WorkerCloudflareD1Saver

# MARK: - Fake D1 Binding


class _FakeD1Statement:
    def __init__(self, conn: sqlite3.Connection, sql: str) -> None:
        self._conn = conn
        self._sql = sql
        self._params: Sequence[Any] = ()

    def bind(self, *params: Any) -> "_FakeD1Statement":
        self._params = params
        return self

    async def all(self) -> dict:
        cursor = self._conn.execute(self._sql, self._params)
        upper = self._sql.strip().upper()
        if upper.startswith(("SELECT", "PRAGMA", "WITH")) or "RETURNING" in upper:
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        else:
            rows = []
        self._conn.commit()
        changes = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        return {"results": rows, "success": True, "meta": {"changes": changes}}

    async def raw(self, opts: Optional[dict] = None) -> list:
        cursor = self._conn.execute(self._sql, self._params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return [columns] if (opts or {}).get("columnNames") else []


class FakeD1Binding:
    """In-memory stand-in for a Cloudflare D1 Worker binding (`env.DB`)."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")

    def prepare(self, sql: str) -> _FakeD1Statement:
        return _FakeD1Statement(self._conn, sql)


# MARK: - Fixtures


@pytest.fixture
def saver() -> WorkerCloudflareD1Saver:
    return WorkerCloudflareD1Saver(FakeD1Binding(), enable_logging=True)


@pytest.fixture
def thread_id() -> str:
    return str(uuid.uuid4())


# MARK: - Tests


class TestWorkerCloudflareD1Saver:
    async def test_put_and_get_tuple_round_trips_checkpoint(
        self, saver: WorkerCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()

        new_config = await saver.aput(
            config, checkpoint, {"source": "input", "step": 1, "parents": {}}, {}
        )

        assert new_config["configurable"]["thread_id"] == thread_id
        assert new_config["configurable"]["checkpoint_id"] == checkpoint["id"]

        tup = await saver.aget_tuple(new_config)
        assert tup is not None
        assert tup.checkpoint["id"] == checkpoint["id"]
        assert tup.metadata["step"] == 1

    async def test_put_writes_are_returned_as_pending_writes(
        self, saver: WorkerCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()
        new_config = await saver.aput(config, checkpoint, {"step": 1}, {})

        await saver.aput_writes(
            new_config, [("channel1", {"foo": "bar"})], task_id="task-1"
        )

        tup = await saver.aget_tuple(new_config)
        assert tup is not None
        assert tup.pending_writes == [("task-1", "channel1", {"foo": "bar"})]

    async def test_aget_tuple_returns_none_when_missing(
        self, saver: WorkerCloudflareD1Saver
    ) -> None:
        config = {"configurable": {"thread_id": "does-not-exist"}}
        assert await saver.aget_tuple(config) is None

    async def test_alist_decodes_metadata_and_respects_limit(
        self, saver: WorkerCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        for step in range(3):
            checkpoint = empty_checkpoint()
            config = await saver.aput(
                config, checkpoint, {"source": "loop", "step": step, "parents": {}}, {}
            )

        # List against the bare thread config (no checkpoint_id), otherwise
        # search_where() scopes the query down to the single checkpoint above.
        list_config = {"configurable": {"thread_id": thread_id}}
        results = [tup async for tup in saver.alist(list_config, limit=2)]
        assert len(results) == 2
        # newest first
        assert results[0].metadata["step"] == 2
        assert results[1].metadata["step"] == 1

    async def test_adelete_thread_removes_checkpoints_and_writes(
        self, saver: WorkerCloudflareD1Saver, thread_id: str
    ) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()
        new_config = await saver.aput(config, checkpoint, {"step": 1}, {})
        await saver.aput_writes(new_config, [("channel1", "value")], task_id="task-1")

        await saver.adelete_thread(thread_id)

        assert await saver.aget_tuple(new_config) is None
        assert [tup async for tup in saver.alist(config)] == []

    def test_sync_methods_require_pyodide_run_sync(
        self, saver: WorkerCloudflareD1Saver, thread_id: str
    ) -> None:
        """Sync methods bridge via pyodide.ffi.run_sync(), unavailable outside a Worker.

        This test runs under plain CPython (no Pyodide), so every sync method
        should raise NotImplementedError from the _run_sync() import guard --
        not because the sync API is unsupported, but because run_sync() only
        exists inside a live Cloudflare Python Worker request. See
        test_worker_integration.py for these same methods actually working
        against a real pywrangler dev server.
        """
        config = {"configurable": {"thread_id": thread_id}}
        with pytest.raises(NotImplementedError, match="run_sync"):
            saver.get_tuple(config)
        with pytest.raises(NotImplementedError, match="run_sync"):
            list(saver.list(config))
        with pytest.raises(NotImplementedError, match="run_sync"):
            saver.put(config, empty_checkpoint(), {"step": 1}, {})
        with pytest.raises(NotImplementedError, match="run_sync"):
            saver.put_writes(config, [("channel1", "value")], "task-1")
        with pytest.raises(NotImplementedError, match="run_sync"):
            saver.delete_thread(thread_id)


def test_worker_saver_requires_sqlalchemy_cloudflare_d1(monkeypatch) -> None:
    """A missing `sqlalchemy_cloudflare_d1` install should raise a clear ImportError."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sqlalchemy_cloudflare_d1":
            raise ImportError("No module named 'sqlalchemy_cloudflare_d1'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(
        ImportError, match="langgraph-checkpoint-cloudflare-d1\\[worker\\]"
    ):
        WorkerCloudflareD1Saver(FakeD1Binding())
