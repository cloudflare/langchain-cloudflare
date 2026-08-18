"""Regression test: search() must not crash on an empty namespace_prefix with no query.

Before the fix, `store.search(())` (list everything, no query -- a documented
valid LangGraph BaseStore usage) raised UnboundLocalError because
`docs_and_scores` was only assigned when `namespace_prefix_str` was truthy.
"""

from dataclasses import dataclass
from typing import Any

from langmem_cloudflare_vectorize import CloudflareVectorizeBaseStore


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, Any]


class FakeVectorstore:
    def __init__(self, docs_with_scores: list[tuple[FakeDocument, float]]):
        self._docs_with_scores = docs_with_scores
        self.calls: list[dict[str, Any]] = []

    def similarity_search_with_score(self, *, query, k, md_filter):
        self.calls.append({"query": query, "k": k, "md_filter": md_filter})
        return self._docs_with_scores


def _make_store(vectorstore: FakeVectorstore) -> CloudflareVectorizeBaseStore:
    store = CloudflareVectorizeBaseStore.__new__(CloudflareVectorizeBaseStore)
    store.vectorstore = vectorstore
    store.binding = None
    return store


def _doc(namespace: str, key: str) -> FakeDocument:
    return FakeDocument(
        page_content="",
        metadata={
            "namespace": namespace,
            "key": key,
            "data": "{}",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        },
    )


def test_search_with_empty_namespace_prefix_and_no_query_does_not_crash() -> None:
    vectorstore = FakeVectorstore([(_doc("a/b", "k1"), 0.9), (_doc("c", "k2"), 0.8)])
    store = _make_store(vectorstore)

    results = store.search(())

    assert len(vectorstore.calls) == 1
    assert {item.key for item in results} == {"k1", "k2"}


def test_search_with_nonempty_namespace_prefix_and_no_query_still_works() -> None:
    vectorstore = FakeVectorstore([(_doc("a/b", "k1"), 0.9), (_doc("c", "k2"), 0.8)])
    store = _make_store(vectorstore)

    results = store.search(("a",))

    assert [item.key for item in results] == ["k1"]
