"""Integration tests for CloudflareVectorizeBaseStore inside a Cloudflare Python Worker.

These tests start the Worker in examples/workers using `pywrangler dev` and
make HTTP requests to verify the Vectorize/AI binding-backed store works end
to end against a real (remote) Vectorize index. See tests/integration_tests/
for the equivalent suite against the REST API.

Note: These tests require:
1. The examples/workers directory to be set up
2. Wrangler OAuth login (`npx wrangler login`) -- not just an API token
3. pywrangler installed (uv add workers-py)

Vectorize inserts are processed by an asynchronous mutation queue rather than
indexed synchronously, so a get()/search() right after a put() can legitimately
miss for anywhere from a few seconds up to a couple of minutes under load
(shared index, concurrent test runs, etc.) -- these tests poll rather than
expect immediate consistency. See examples/workers/README.md's note on
/store-graph for the same characteristic.
"""

import time
import uuid

import pytest
import requests

_POLL_TIMEOUT_SECONDS = 120
_SEARCH_POLL_TIMEOUT_SECONDS = 240
_POLL_INTERVAL_SECONDS = 3


def _poll_until(predicate, timeout_seconds: int = _POLL_TIMEOUT_SECONDS):
    """Call `predicate()` until it returns a truthy value or time runs out.

    Returns the last (falsy) result if the timeout is reached, rather than
    raising, so callers can assert against a clear final value/message.
    """
    deadline = time.time() + timeout_seconds
    result = predicate()
    while not result and time.time() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)
        result = predicate()
    return result


# MARK: - Health


class TestWorkerIndex:
    def test_index_returns_documentation(self, dev_server) -> None:
        port = dev_server
        response = requests.get(f"http://localhost:{port}/")

        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "/store-put" in data["endpoints"]
        assert "/store-graph" in data["endpoints"]

    def test_health_reports_bindings_reachable(self, dev_server) -> None:
        port = dev_server
        response = requests.get(f"http://localhost:{port}/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# MARK: - Store CRUD

# Shared test bodies for both the async (native) and sync-bridge Worker
# paths -- every /store-* CRUD endpoint accepts a "sync" field that routes
# through CloudflareVectorizeBaseStore's sync methods (get/put/delete/search,
# bridged via pyodide.ffi.run_sync()) instead of the async ones
# (aget/aput/adelete/asearch).


class _WorkerStoreCrudBase:
    sync: bool = False

    @pytest.fixture
    def namespace(self) -> list:
        return ["worker-test", uuid.uuid4().hex]

    def _post(self, port: int, path: str, payload: dict) -> requests.Response:
        return requests.post(
            f"http://localhost:{port}{path}",
            json={**payload, "sync": self.sync},
            headers={"Content-Type": "application/json"},
        )

    def test_put_and_get_round_trips_item(self, dev_server, namespace: list) -> None:
        port = dev_server
        key = "note"

        put_res = self._post(
            port,
            "/store-put",
            {"namespace": namespace, "key": key, "value": {"text": "hello world"}},
        )
        assert put_res.status_code == 200
        assert put_res.json()["success"] is True

        try:

            def _get():
                data = self._post(
                    port, "/store-get", {"namespace": namespace, "key": key}
                ).json()
                return data if data.get("found") else None

            found = _poll_until(_get)
            assert found is not None, "get never found the item"
            assert found["value"] == {"text": "hello world"}
        finally:
            self._post(port, "/store-delete", {"namespace": namespace, "key": key})

    def test_get_missing_item_returns_not_found(
        self, dev_server, namespace: list
    ) -> None:
        port = dev_server
        res = self._post(port, "/store-get", {"namespace": namespace, "key": "nope"})
        assert res.status_code == 200
        assert res.json()["found"] is False

    def test_search_finds_item_within_namespace_prefix(
        self, dev_server, namespace: list
    ) -> None:
        port = dev_server
        key = "note"

        self._post(
            port,
            "/store-put",
            {
                "namespace": namespace,
                "key": key,
                "value": {"text": "searchable memory"},
            },
        )

        try:

            def _search():
                res = self._post(
                    port,
                    "/store-search",
                    {"namespace_prefix": namespace, "limit": 10},
                )
                data = res.json()
                return data if data.get("count", 0) > 0 else None

            # Vectorize's ANN search index lags further behind a raw insert
            # than direct getByIds lookups do, so this needs a longer window
            # than the plain get()-based tests above.
            data = _poll_until(_search, timeout_seconds=_SEARCH_POLL_TIMEOUT_SECONDS)
            assert data is not None, "search never found the item"
            assert any(r["key"] == key for r in data["results"])
        finally:
            self._post(port, "/store-delete", {"namespace": namespace, "key": key})

    def test_delete_removes_item(self, dev_server, namespace: list) -> None:
        port = dev_server
        key = "note"

        self._post(
            port,
            "/store-put",
            {"namespace": namespace, "key": key, "value": {"text": "to be deleted"}},
        )
        _poll_until(
            lambda: (
                self._post(port, "/store-get", {"namespace": namespace, "key": key})
                .json()
                .get("found")
            )
        )

        del_res = self._post(
            port, "/store-delete", {"namespace": namespace, "key": key}
        )
        assert del_res.status_code == 200
        assert del_res.json()["success"] is True

        gone = _poll_until(
            lambda: (
                None
                if self._post(port, "/store-get", {"namespace": namespace, "key": key})
                .json()
                .get("found")
                else True
            )
        )
        assert gone is True


class TestWorkerStoreCrud(_WorkerStoreCrudBase):
    """Async path: aget/aput/adelete/asearch."""

    sync = False


class TestWorkerStoreCrudSync(_WorkerStoreCrudBase):
    """Sync-bridge path: get/put/delete/search, via pyodide.ffi.run_sync() --
    the same mechanism sqlalchemy_cloudflare_d1.SyncWorkerConnection uses for
    SQLAlchemy's sync engine, and WorkerCloudflareD1Saver uses in
    langgraph-checkpoint-cloudflare-d1.
    """

    sync = True


# MARK: - Full Graph Demo


class TestWorkerStoreGraph:
    def test_graph_writes_then_recalls_memory_across_runs(self, dev_server) -> None:
        port = dev_server
        thread_id = f"worker-test-graph-{uuid.uuid4().hex}"

        write_res = requests.post(
            f"http://localhost:{port}/store-graph",
            json={"thread_id": thread_id, "text": "graph memory example"},
            headers={"Content-Type": "application/json"},
        )
        assert write_res.status_code == 200
        write_data = write_res.json()
        assert write_data["success"] is True
        assert write_data["recall"] is False

        def _recall():
            res = requests.post(
                f"http://localhost:{port}/store-graph",
                json={"thread_id": thread_id, "recall": True},
                headers={"Content-Type": "application/json"},
            )
            data = res.json()
            return data if data["result"].get("remembered") else None

        recall_data = _poll_until(_recall)
        assert recall_data is not None, "graph never recalled the written memory"
        assert recall_data["result"]["remembered"] == {"text": "graph memory example"}
