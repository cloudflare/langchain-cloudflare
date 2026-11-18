"""search()/asearch() must match namespace prefixes by whole namespace labels.

LangGraph's BaseStore treats ``namespace_prefix`` as a tuple prefix:
``("memories", "1")`` covers ``("memories", "1")`` and ``("memories", "1", "x")``
but not ``("memories", "12")``. Comparing the joined strings with
``str.startswith`` let a search scoped to user "1" return user "12"'s items.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from langmem_cloudflare_vectorize import CloudflareVectorizeBaseStore


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, Any]


class FakeVectorstore:
    def __init__(self, docs_with_scores: list[tuple[FakeDocument, float]]):
        self._docs_with_scores = docs_with_scores

    def similarity_search_with_score(self, *, query, k, md_filter):
        return self._docs_with_scores

    async def asimilarity_search_with_score(
        self, *, query, k, md_filter, return_metadata
    ):
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


DOCS = [
    (_doc("memories/12", "other-user"), 0.95),
    (_doc("memories/1", "own"), 0.9),
    (_doc("memories/1/notes", "own-nested"), 0.85),
    (_doc("memories-archive/1", "other-namespace"), 0.8),
]


@pytest.mark.parametrize(
    "prefix,expected",
    [
        (("memories", "1"), ["own", "own-nested"]),
        (("memories",), ["other-user", "own", "own-nested"]),
        (("memories", "1", "notes"), ["own-nested"]),
        ((), ["other-user", "own", "own-nested", "other-namespace"]),
    ],
)
def test_search_matches_whole_namespace_labels(prefix, expected) -> None:
    store = _make_store(FakeVectorstore(DOCS))

    assert [item.key for item in store.search(prefix, query="q")] == expected
    assert [item.key for item in store.search(prefix)] == expected


@pytest.mark.parametrize(
    "prefix,expected",
    [
        (("memories", "1"), ["own", "own-nested"]),
        (("memories",), ["other-user", "own", "own-nested"]),
    ],
)
async def test_asearch_matches_whole_namespace_labels(prefix, expected) -> None:
    store = _make_store(FakeVectorstore(DOCS))

    results = await store.asearch(prefix, query="q")

    assert [item.key for item in results] == expected
