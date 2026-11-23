"""Integration tests for WorkerCloudflareD1Saver inside a Cloudflare Python Worker.

These tests start the Worker in examples/workers using `pywrangler dev` and
make HTTP requests to verify the D1 binding-backed checkpointer works end to
end against a real (remote) D1 database. See tests/integration_tests/ for the
equivalent suite against the REST API savers.

Note: These tests require:
1. The examples/workers directory to be set up
2. Wrangler OAuth login (`npx wrangler login`) -- not just an API token
3. pywrangler installed (uv add workers-py)
"""

import uuid

import pytest
import requests
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

# MARK: - Health


class TestWorkerIndex:
    def test_index_returns_documentation(self, dev_server) -> None:
        port = dev_server
        response = requests.get(f"http://localhost:{port}/")

        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "/checkpointer-put" in data["endpoints"]
        assert "/checkpointer-graph" in data["endpoints"]

    def test_health_reports_table_setup(self, dev_server) -> None:
        port = dev_server
        response = requests.get(f"http://localhost:{port}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["is_setup"] is True


# MARK: - Checkpoint CRUD

# Shared test bodies for both the async (native) and sync-bridge Worker
# paths -- every /checkpointer-* CRUD endpoint accepts a "sync" field that
# routes through WorkerCloudflareD1Saver's sync methods (put/get_tuple/
# put_writes/list/delete_thread, bridged via pyodide.ffi.run_sync()) instead
# of the async ones (aput/aget_tuple/...). Subclasses just set `sync`, giving
# both paths the same coverage REST already has via CloudflareD1Saver vs.
# AsyncCloudflareD1Saver being separate classes.


class _WorkerCheckpointerCrudBase:
    sync: bool = False

    @pytest.fixture
    def thread_id(self) -> str:
        return f"worker-test-{uuid.uuid4().hex}"

    def _post(self, port: int, path: str, payload: dict) -> requests.Response:
        return requests.post(
            f"http://localhost:{port}{path}",
            json={**payload, "sync": self.sync},
            headers={"Content-Type": "application/json"},
        )

    def test_put_and_get_tuple_round_trips_checkpoint(
        self, dev_server, thread_id: str
    ) -> None:
        port = dev_server

        put_res = self._post(
            port,
            "/checkpointer-put",
            {
                "thread_id": thread_id,
                "metadata": {"source": "input", "step": 1, "parents": {}},
            },
        )
        assert put_res.status_code == 200
        put_data = put_res.json()
        assert put_data["success"] is True
        checkpoint_id = put_data["checkpoint_id"]

        try:
            get_res = self._post(
                port, "/checkpointer-get-tuple", {"thread_id": thread_id}
            )
            assert get_res.status_code == 200
            get_data = get_res.json()
            assert get_data["found"] is True
            assert get_data["checkpoint_id"] == checkpoint_id
            assert get_data["metadata"]["step"] == 1
        finally:
            self._post(port, "/checkpointer-delete-thread", {"thread_id": thread_id})

    def test_put_writes_are_returned_as_pending_writes(
        self, dev_server, thread_id: str
    ) -> None:
        port = dev_server

        put_res = self._post(port, "/checkpointer-put", {"thread_id": thread_id})
        checkpoint_id = put_res.json()["checkpoint_id"]

        writes_res = self._post(
            port,
            "/checkpointer-put-writes",
            {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "task_id": "task-1",
                "writes": [{"channel": "channel1", "value": {"foo": "bar"}}],
            },
        )
        assert writes_res.status_code == 200
        assert writes_res.json()["count"] == 1

        try:
            get_res = self._post(
                port, "/checkpointer-get-tuple", {"thread_id": thread_id}
            )
            data = get_res.json()
            assert data["pending_writes"] == [["task-1", "channel1", {"foo": "bar"}]]
        finally:
            self._post(port, "/checkpointer-delete-thread", {"thread_id": thread_id})

    def test_list_decodes_metadata_and_respects_limit(
        self, dev_server, thread_id: str
    ) -> None:
        port = dev_server

        for step in range(3):
            self._post(
                port,
                "/checkpointer-put",
                {
                    "thread_id": thread_id,
                    "metadata": {"source": "loop", "step": step, "parents": {}},
                },
            )

        try:
            list_res = self._post(
                port, "/checkpointer-list", {"thread_id": thread_id, "limit": 2}
            )
            assert list_res.status_code == 200
            data = list_res.json()
            assert data["count"] == 2
            assert data["checkpoints"][0]["metadata"]["step"] == 2
            assert data["checkpoints"][1]["metadata"]["step"] == 1
        finally:
            self._post(port, "/checkpointer-delete-thread", {"thread_id": thread_id})

    def test_delete_thread_removes_checkpoints(
        self, dev_server, thread_id: str
    ) -> None:
        port = dev_server

        self._post(port, "/checkpointer-put", {"thread_id": thread_id})
        del_res = self._post(
            port, "/checkpointer-delete-thread", {"thread_id": thread_id}
        )
        assert del_res.status_code == 200

        get_res = self._post(port, "/checkpointer-get-tuple", {"thread_id": thread_id})
        assert get_res.json()["found"] is False

    def test_get_tuple_and_list_decode_checkpoint_after_null_write(
        self, dev_server, thread_id: str
    ) -> None:
        # The input value, then a None branch trigger (type "null"), as
        # LangGraph stores them for a run's input step. The checkpoint must
        # still decode with its own type.
        port = dev_server

        put_res = self._post(
            port,
            "/checkpointer-put",
            {"thread_id": thread_id, "metadata": {"source": "input", "step": -1}},
        )
        checkpoint_id = put_res.json()["checkpoint_id"]
        self._post(
            port,
            "/checkpointer-put-writes",
            {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "task_id": "task-1",
                "writes": [
                    {"channel": "n", "value": 1},
                    {"channel": "branch:to:add_one", "value": None},
                ],
            },
        )

        try:
            get_res = self._post(
                port, "/checkpointer-get-tuple", {"thread_id": thread_id}
            )
            assert get_res.status_code == 200, get_res.text
            assert get_res.json()["checkpoint_id"] == checkpoint_id

            list_res = self._post(port, "/checkpointer-list", {"thread_id": thread_id})
            assert list_res.status_code == 200, list_res.text
            assert [c["checkpoint_id"] for c in list_res.json()["checkpoints"]] == [
                checkpoint_id
            ]
        finally:
            self._post(port, "/checkpointer-delete-thread", {"thread_id": thread_id})


class TestWorkerCheckpointerCrud(_WorkerCheckpointerCrudBase):
    """Async path: aput/aget_tuple/aput_writes/alist/adelete_thread."""

    sync = False


class TestWorkerCheckpointerCrudSync(_WorkerCheckpointerCrudBase):
    """Sync-bridge path: put/get_tuple/put_writes/list/delete_thread, via
    pyodide.ffi.run_sync() -- the same mechanism
    sqlalchemy_cloudflare_d1.SyncWorkerConnection uses for SQLAlchemy's sync
    engine. See WorkerCloudflareD1Saver's docstring for what this does and
    doesn't cover (notably: not graph.invoke()).
    """

    sync = True


# MARK: - Full Graph Demo


def _expected_history(start_value: int) -> list:
    """State history of the Worker's add_one graph under InMemorySaver."""

    def add_one(state: int) -> int:
        return state + 1

    builder = StateGraph(int)
    builder.add_node("add_one", add_one)
    builder.add_edge(START, "add_one")
    builder.add_edge("add_one", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "expected"}}
    graph.invoke(start_value, config)
    return [
        {
            "source": s.metadata["source"],
            "step": s.metadata["step"],
            "values": s.values,
            "next": list(s.next),
        }
        for s in graph.get_state_history(config)
    ]


class TestWorkerCheckpointerGraph:
    def test_graph_compiles_and_runs_with_worker_saver(self, dev_server) -> None:
        port = dev_server
        thread_id = f"worker-test-graph-{uuid.uuid4().hex}"

        response = requests.post(
            f"http://localhost:{port}/checkpointer-graph",
            json={"thread_id": thread_id, "value": 3},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["input"] == 3
        assert data["result"] == 4
        assert data["checkpoint_id"]

        # Same history InMemorySaver gives for the same graph and input.
        assert data["history"] == _expected_history(3)
        assert data["input_next"] == ["__start__"]
        assert data["replay_result"] == 4

        requests.post(
            f"http://localhost:{port}/checkpointer-delete-thread",
            json={"thread_id": thread_id},
            headers={"Content-Type": "application/json"},
        )
