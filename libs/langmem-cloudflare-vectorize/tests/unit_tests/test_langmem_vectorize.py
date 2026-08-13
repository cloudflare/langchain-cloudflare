"""Test CloudflareVectorizeBaseStore implementation."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from langmem_cloudflare_vectorize import CloudflareVectorizeBaseStore


@pytest.mark.parametrize(
    ("namespace", "key", "expected_combined"),
    [
        (("documents", "user_123"), "document_456", "documents/user_123:document_456"),
        (("files",), "readme.txt", "files:readme.txt"),
        (("users", "admin", "settings"), "config", "users/admin/settings:config"),
    ],
)
def test_generate_vector_id(
    namespace: tuple[str, ...], key: str, expected_combined: str
) -> None:
    """Test that _generate_vector_id creates correct SHA256 hashes."""
    # Test the static method directly without needing dependencies
    store = CloudflareVectorizeBaseStore.__new__(CloudflareVectorizeBaseStore)
    vector_id = store._generate_vector_id(namespace, key)

    # Verify it matches expected SHA256 hash
    expected_hash = hashlib.sha256(expected_combined.encode()).hexdigest()

    assert vector_id == expected_hash


def test_with_cloudflare_embeddings_sets_index_name_on_vectorstore() -> None:
    """Regression test: with_cloudflare_embeddings() must build its
    CloudflareVectorize instance with index_name set.

    Before the fix, index_name was only ever passed to create_index() (a
    one-off call-time argument) and to the CloudflareVectorizeBaseStore
    constructor (which sets it on the *store*, not on the reused
    cf_vectorize instance) -- so store.vectorstore.index_name stayed None,
    and any CloudflareVectorize method relying on the instance default
    (aget_by_ids, similarity_search, etc.) failed with "index_name must be
    provided" the moment it wasn't passed explicitly per call.
    """
    with (
        patch(
            "langchain_cloudflare.vectorstores.CloudflareVectorize.list_indexes",
            return_value=[{"name": "existing-index"}],
        ),
        patch(
            "langchain_cloudflare.vectorstores.CloudflareVectorize.create_index"
        ) as mock_create_index,
        patch(
            "langchain_cloudflare.embeddings.CloudflareWorkersAIEmbeddings",
            return_value=MagicMock(),
        ),
    ):
        store = CloudflareVectorizeBaseStore.with_cloudflare_embeddings(
            account_id="acct",
            index_name="existing-index",
            workers_ai_token="ai-token",
            vectorize_api_token="vec-token",
        )

    mock_create_index.assert_not_called()  # index already exists per list_indexes
    assert store.vectorstore.index_name == "existing-index"
