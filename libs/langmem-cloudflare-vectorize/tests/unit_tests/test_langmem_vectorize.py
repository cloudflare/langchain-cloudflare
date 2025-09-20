"""Test CloudflareVectorizeBaseStore implementation."""

import hashlib

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
