"""Unit tests for CloudflareVectorizeBaseStore's Worker binding support.

These exercise the real async methods (aget/aput/adelete/asearch/abatch)
against an in-memory fake Vectorize binding, plus the sync bridge's
behavior outside of a Worker runtime -- all without Pyodide, wrangler, or
network access. See tests/worker_tests/ (added separately) for the
equivalent suite against a real pywrangler dev server.

The fake binding mimics the `upsert`/`insert`/`query`/`getByIds`/`deleteByIds`
async surface that `langchain_cloudflare.vectorstores.CloudflareVectorize`'s
`_binding_*` helpers call against the real Vectorize binding.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import GetOp, PutOp

from langmem_cloudflare_vectorize import CloudflareVectorizeBaseStore

DUMMY_EMBEDDING = [0.1, 0.2, 0.3]


# MARK: - Fakes


class FakeEmbeddings(Embeddings):
    """Minimal Embeddings stand-in -- avoids a real REST/Workers AI call."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [DUMMY_EMBEDDING for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return DUMMY_EMBEDDING

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class FakeVectorizeBinding:
    """In-memory stand-in for a Cloudflare Vectorize Worker binding
    (`env.VECTORIZE`)."""

    def __init__(self) -> None:
        self._vectors: Dict[str, Dict[str, Any]] = {}

    async def upsert(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        for v in vectors:
            self._vectors[v["id"]] = v
        return {"mutationId": "fake", "count": len(vectors)}

    async def insert(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.upsert(vectors)

    async def query(self, vector: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        top_k = options.get("topK", 20)
        matches = []
        for v in list(self._vectors.values())[:top_k]:
            match: Dict[str, Any] = {"id": v["id"], "score": 0.9}
            if options.get("returnMetadata"):
                match["metadata"] = v.get("metadata", {})
            matches.append(match)
        return {"matches": matches}

    async def getByIds(self, ids: List[str]) -> List[Dict[str, Any]]:
        return [self._vectors[i] for i in ids if i in self._vectors]

    async def deleteByIds(self, ids: List[str]) -> Dict[str, Any]:
        for i in ids:
            self._vectors.pop(i, None)
        return {"mutationId": "fake", "count": len(ids)}


# MARK: - Fixtures


@pytest.fixture
def store() -> CloudflareVectorizeBaseStore:
    return CloudflareVectorizeBaseStore(
        embedding_function=FakeEmbeddings(),
        binding=FakeVectorizeBinding(),
        index_name="test-index",
    )


# MARK: - Tests


class TestCloudflareVectorizeBaseStoreBinding:
    async def test_aput_and_aget_round_trip(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        await store.aput(("docs", "a"), "k1", {"text": "hello world"})

        item = await store.aget(("docs", "a"), "k1")

        assert item is not None
        assert item.value == {"text": "hello world"}

    async def test_aget_returns_none_when_missing(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        assert await store.aget(("docs",), "does-not-exist") is None

    async def test_asearch_finds_put_items(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        await store.aput(("docs", "a"), "k1", {"text": "hello"})
        await store.aput(("docs", "b"), "k2", {"text": "world"})

        results = await store.asearch(())
        assert {item.key for item in results} == {"k1", "k2"}

        scoped = await store.asearch(("docs", "a"))
        assert [item.key for item in scoped] == ["k1"]

    async def test_adelete_removes_item(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        await store.aput(("docs", "a"), "k1", {"text": "hello"})
        await store.adelete(("docs", "a"), "k1")

        assert await store.aget(("docs", "a"), "k1") is None

    async def test_abatch_handles_put_and_get(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        await store.abatch(
            [PutOp(namespace=("docs", "b"), key="k2", value={"n": 1}, index=None)]
        )
        results = await store.abatch([GetOp(namespace=("docs", "b"), key="k2")])

        assert results[0].value == {"n": 1}  # type: ignore[union-attr]

    def test_sync_methods_require_pyodide_run_sync(
        self, store: CloudflareVectorizeBaseStore
    ) -> None:
        """Sync methods bridge via pyodide.ffi.run_sync() when a binding is
        set, unavailable outside a Worker. This test runs under plain
        CPython, so every sync method should raise NotImplementedError from
        the _run_sync() import guard -- not because sync usage with a
        binding is unsupported, but because run_sync() only exists inside a
        live Cloudflare Python Worker request.
        """
        with pytest.raises(NotImplementedError, match="run_sync"):
            store.get(("docs",), "k1")
        with pytest.raises(NotImplementedError, match="run_sync"):
            store.put(("docs",), "k1", {"n": 1})
        with pytest.raises(NotImplementedError, match="run_sync"):
            store.delete(("docs",), "k1")
        with pytest.raises(NotImplementedError, match="run_sync"):
            store.search(())
        with pytest.raises(NotImplementedError, match="run_sync"):
            store.batch([GetOp(namespace=("docs",), key="k1")])


def test_init_requires_embedding_function() -> None:
    with pytest.raises(ValueError, match="embedding_function is required"):
        CloudflareVectorizeBaseStore(binding=FakeVectorizeBinding())


def test_init_reuses_already_configured_vectorstore() -> None:
    """cf_vectorize, when given, is what operations actually use -- not
    silently discarded in favor of a second, separately-constructed
    instance (the pre-existing bug this fixes)."""
    from langchain_cloudflare.vectorstores import CloudflareVectorize

    binding = FakeVectorizeBinding()
    cf_vectorize = CloudflareVectorize(
        embedding=FakeEmbeddings(),  # type: ignore[arg-type]
        binding=binding,
        index_name="test-index",
    )

    store = CloudflareVectorizeBaseStore(
        embedding_function=FakeEmbeddings(),
        cf_vectorize=cf_vectorize,
    )

    assert store.vectorstore is cf_vectorize
    assert store.cf_vectorize is cf_vectorize
