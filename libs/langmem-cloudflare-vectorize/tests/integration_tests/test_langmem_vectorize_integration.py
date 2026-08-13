"""Integration tests for CloudflareVectorizeBaseStore against the live
Cloudflare Vectorize/Workers AI REST APIs.

In order to run this test, you need to:
1. Have a Cloudflare account
2. Set up API tokens with access to Workers AI and Vectorize
3. Set environment variables (in a repo-root .env file):
   CF_ACCOUNT_ID
   CF_AI_API_TOKEN
   CF_VECTORIZE_API_TOKEN

Vectorize inserts are processed by an asynchronous mutation queue rather than
indexed synchronously, so a get()/search() right after a put() can legitimately
miss for a few seconds -- these tests poll rather than expect immediate
consistency. search() polls longer than get(): Vectorize's ANN search index
lags further behind a raw insert than direct ID lookups do. See
examples/workers/README.md's note on /store-graph for the same
characteristic on the Worker-binding side.
"""

import time
import uuid
from typing import Generator

import pytest

from langmem_cloudflare_vectorize import CloudflareVectorizeBaseStore

_POLL_TIMEOUT_SECONDS = 60
_SEARCH_POLL_TIMEOUT_SECONDS = 90
_POLL_INTERVAL_SECONDS = 2


def _poll_until(predicate, timeout_seconds: int = _POLL_TIMEOUT_SECONDS):
    """Call `predicate()` until it returns a truthy value or time runs out."""
    deadline = time.time() + timeout_seconds
    result = predicate()
    while not result and time.time() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)
        result = predicate()
    return result


async def _apoll_until(predicate, timeout_seconds: int = _POLL_TIMEOUT_SECONDS):
    """Async version of _poll_until: `predicate` is an async callable."""
    deadline = time.time() + timeout_seconds
    result = await predicate()
    while not result and time.time() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)
        result = await predicate()
    return result


@pytest.fixture(scope="class")
def store(
    vectorize_credentials: tuple, index_name_class_scoped: str
) -> Generator[CloudflareVectorizeBaseStore, None, None]:
    account_id, ai_api_token, vectorize_api_token = vectorize_credentials

    store = CloudflareVectorizeBaseStore.with_cloudflare_embeddings(
        account_id=account_id,
        index_name=index_name_class_scoped,
        workers_ai_token=ai_api_token,
        vectorize_api_token=vectorize_api_token,
    )

    yield store

    store.vectorstore.delete_index()


@pytest.fixture(scope="class")
def index_name_class_scoped() -> str:
    return f"lgmem-vec-integration-{uuid.uuid4().hex[:16]}"


# MARK: - Sync CRUD


class TestCloudflareVectorizeBaseStoreSync:
    """REST API sync path: get/put/delete/search."""

    def test_put_and_get_round_trips_item(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        namespace = ("integration-sync", uuid.uuid4().hex)
        key = "note"

        store.put(namespace, key, {"text": "hello world"})
        try:
            item = _poll_until(lambda: store.get(namespace, key))
            assert item is not None
            assert item.value == {"text": "hello world"}
        finally:
            store.delete(namespace, key)

    def test_get_missing_item_returns_none(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        namespace = ("integration-sync", uuid.uuid4().hex)
        assert store.get(namespace, "nope") is None

    def test_search_finds_item_within_namespace_prefix(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        namespace = ("integration-sync-search", uuid.uuid4().hex)
        key = "note"

        store.put(namespace, key, {"text": "searchable memory"})
        try:
            results = _poll_until(
                lambda: store.search(namespace) or None,
                timeout_seconds=_SEARCH_POLL_TIMEOUT_SECONDS,
            )
            assert results
            assert any(r.key == key for r in results)
        finally:
            store.delete(namespace, key)

    def test_delete_removes_item(self, store: CloudflareVectorizeBaseStore) -> None:
        namespace = ("integration-sync-delete", uuid.uuid4().hex)
        key = "note"

        store.put(namespace, key, {"text": "to be deleted"})
        _poll_until(lambda: store.get(namespace, key))

        store.delete(namespace, key)

        gone = _poll_until(lambda: store.get(namespace, key) is None)
        assert gone is True


# MARK: - Async CRUD


class TestCloudflareVectorizeBaseStoreAsync:
    """REST API async path: aget/aput/adelete/asearch."""

    async def test_aput_and_aget_round_trips_item(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        namespace = ("integration-async", uuid.uuid4().hex)
        key = "note"

        await store.aput(namespace, key, {"text": "hello async world"})
        try:
            item = await _apoll_until(lambda: store.aget(namespace, key))
            assert item is not None
            assert item.value == {"text": "hello async world"}
        finally:
            await store.adelete(namespace, key)

    async def test_aget_missing_item_returns_none(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        namespace = ("integration-async", uuid.uuid4().hex)
        assert await store.aget(namespace, "nope") is None

    async def test_asearch_finds_item_within_namespace_prefix(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        namespace = ("integration-async-search", uuid.uuid4().hex)
        key = "note"

        await store.aput(namespace, key, {"text": "async searchable memory"})
        try:
            results = await _apoll_until(
                lambda: store.asearch(namespace),
                timeout_seconds=_SEARCH_POLL_TIMEOUT_SECONDS,
            )
            assert results
            assert any(r.key == key for r in results)
        finally:
            await store.adelete(namespace, key)

    async def test_adelete_removes_item(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        namespace = ("integration-async-delete", uuid.uuid4().hex)
        key = "note"

        await store.aput(namespace, key, {"text": "to be deleted async"})
        await _apoll_until(lambda: store.aget(namespace, key))

        await store.adelete(namespace, key)

        async def _check_gone():
            return (await store.aget(namespace, key)) is None

        gone = await _apoll_until(_check_gone)
        assert gone is True
