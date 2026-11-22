"""Shared cases and assertions for REST and Worker embedding results."""

# MARK: - IMPORTS
import math
from typing import List, Optional

import pytest

# MARK: - CASES
# (model, requested dimensions, expected vector length, query, documents).
# EmbeddingGemma 2 takes caller-supplied task prefixes; legacy models take
# plain text.
_GEMMA_QUERY = "task: search result | query: waterproof running shoes"
_GEMMA_DOCUMENTS = [
    "title: Trail shoes | text: Waterproof shoes built for running on wet trails.",
    "title: Chocolate cake | text: Bake the cake for forty minutes.",
]
_LEGACY_QUERY = "waterproof running shoes"
_LEGACY_DOCUMENTS = [
    "Waterproof shoes built for running on wet trails.",
    "Bake the cake for forty minutes.",
]

EMBEDDING_CASES = [
    pytest.param(
        "@cf/baai/bge-base-en-v1.5",
        None,
        768,
        _LEGACY_QUERY,
        _LEGACY_DOCUMENTS,
        id="bge-base",
    ),
    pytest.param(
        "@cf/google/embeddinggemma-2",
        None,
        768,
        _GEMMA_QUERY,
        _GEMMA_DOCUMENTS,
        id="embeddinggemma-2-default",
    ),
    pytest.param(
        "@cf/google/embeddinggemma-2",
        256,
        256,
        _GEMMA_QUERY,
        _GEMMA_DOCUMENTS,
        id="embeddinggemma-2-256",
    ),
]


# MARK: - ASSERTIONS
def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.hypot(*a) * math.hypot(*b))


def assert_embeddings(
    query_vector: List[float],
    document_vectors: List[List[float]],
    expected_dimensions: int,
    documents: Optional[List[str]] = None,
) -> None:
    """Check vector shape and that the query ranks the relevant document first.

    The first document in each case answers the query; the second does not.
    """
    if documents is not None:
        assert len(document_vectors) == len(documents)
    for vector in [query_vector, *document_vectors]:
        assert len(vector) == expected_dimensions
        assert all(isinstance(value, float) for value in vector)
    relevant, unrelated = document_vectors
    assert _cosine(query_vector, relevant) > _cosine(query_vector, unrelated)
