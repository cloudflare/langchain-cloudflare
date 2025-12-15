"""Unit tests for CloudflareVectorize vectorstores.

These tests verify the D1 SQLAlchemy integration and helper methods.
SQL injection prevention is handled by SQLAlchemy's parameterized queries.
"""

import json
from unittest.mock import MagicMock

import pytest

from langchain_cloudflare.vectorstores import CloudflareVectorize, VectorizeRecord

# Dummy embedding values for test records
DUMMY_EMBEDDING = [0.0] * 10


class TestVectorizeRecord:
    """Test VectorizeRecord class."""

    def test_record_creation(self) -> None:
        """Test basic record creation."""
        record = VectorizeRecord(
            id="doc-1",
            text="Hello world",
            values=DUMMY_EMBEDDING,
            namespace="test",
            metadata={"author": "Test Author"},
        )

        assert record.id == "doc-1"
        assert record.text == "Hello world"
        assert record.namespace == "test"
        assert record.metadata == {"author": "Test Author"}

    def test_record_to_dict(self) -> None:
        """Test record serialization to dict."""
        record = VectorizeRecord(
            id="doc-1",
            text="Test",
            values=DUMMY_EMBEDDING,
            namespace="ns",
            metadata={"key": "value"},
        )

        record_dict = record.to_dict()
        assert record_dict["id"] == "doc-1"
        assert record_dict["text"] == "Test"
        assert record_dict["namespace"] == "ns"
        assert record_dict["metadata"] == {"key": "value"}

    def test_record_with_none_metadata(self) -> None:
        """Test record with None metadata."""
        record = VectorizeRecord(
            id="doc-1",
            text="Test",
            values=DUMMY_EMBEDDING,
            namespace="",
            metadata=None,
        )

        # Should not raise
        record_dict = record.to_dict()
        assert record_dict["id"] == "doc-1"

    def test_record_with_nested_metadata(self) -> None:
        """Test record with complex nested metadata."""
        nested_metadata = {
            "level1": {"level2": {"level3": "deep value"}},
            "tags": ["tag1", "tag2"],
        }
        record = VectorizeRecord(
            id="doc-1",
            text="Test",
            values=DUMMY_EMBEDDING,
            namespace="",
            metadata=nested_metadata,
        )

        record_dict = record.to_dict()
        assert record_dict["metadata"]["level1"]["level2"]["level3"] == "deep value"
        assert record_dict["metadata"]["tags"] == ["tag1", "tag2"]


class TestD1EngineHelpers:
    """Test D1 engine and table helper methods."""

    def test_get_d1_table_structure(self) -> None:
        """Test that _get_d1_table returns correct table structure."""
        # Create a mock instance with required attributes
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize._get_d1_table = CloudflareVectorize._get_d1_table.__get__(
            mock_vectorize, CloudflareVectorize
        )

        table = mock_vectorize._get_d1_table("test_table")

        # Verify table structure
        assert table.name == "test_table"
        column_names = [col.name for col in table.columns]
        assert "id" in column_names
        assert "text" in column_names
        assert "namespace" in column_names
        assert "metadata" in column_names

    def test_get_d1_engine_requires_database_id(self) -> None:
        """Test that _get_d1_engine raises error without database ID."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_database_id = None
        mock_vectorize._get_d1_engine = CloudflareVectorize._get_d1_engine.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="D1 database ID is required"):
            mock_vectorize._get_d1_engine()

    def test_get_d1_engine_requires_api_token(self) -> None:
        """Test that _get_d1_engine raises error without API token."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_database_id = "test-db-id"
        mock_vectorize.d1_api_token = None
        mock_vectorize.api_token = None
        mock_vectorize._get_d1_engine = CloudflareVectorize._get_d1_engine.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="D1 API token is required"):
            mock_vectorize._get_d1_engine()


class TestD1MethodValidation:
    """Test D1 method input validation."""

    def test_d1_create_table_requires_table_name(self) -> None:
        """Test that d1_create_table validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_create_table = CloudflareVectorize.d1_create_table.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_create_table("")

    def test_d1_drop_table_requires_table_name(self) -> None:
        """Test that d1_drop_table validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_drop_table = CloudflareVectorize.d1_drop_table.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_drop_table("")

    def test_d1_upsert_texts_requires_table_name(self) -> None:
        """Test that d1_upsert_texts validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_upsert_texts = CloudflareVectorize.d1_upsert_texts.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_upsert_texts("", [])

    def test_d1_upsert_texts_empty_data_returns_success(self) -> None:
        """Test that d1_upsert_texts handles empty data."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_upsert_texts = CloudflareVectorize.d1_upsert_texts.__get__(
            mock_vectorize, CloudflareVectorize
        )

        result = mock_vectorize.d1_upsert_texts("test_table", [])
        assert result == {"success": True, "changes": 0}

    def test_d1_get_by_ids_requires_table_name(self) -> None:
        """Test that d1_get_by_ids validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_get_by_ids = CloudflareVectorize.d1_get_by_ids.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_get_by_ids("", ["id1"])

    def test_d1_get_by_ids_empty_ids_returns_empty(self) -> None:
        """Test that d1_get_by_ids handles empty IDs list."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_get_by_ids = CloudflareVectorize.d1_get_by_ids.__get__(
            mock_vectorize, CloudflareVectorize
        )

        result = mock_vectorize.d1_get_by_ids("test_table", [])
        assert result == []

    def test_d1_delete_requires_table_name(self) -> None:
        """Test that d1_delete validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_delete = CloudflareVectorize.d1_delete.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_delete("", ["id1"])

    def test_d1_delete_empty_ids_returns_success(self) -> None:
        """Test that d1_delete handles empty IDs list."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_delete = CloudflareVectorize.d1_delete.__get__(
            mock_vectorize, CloudflareVectorize
        )

        result = mock_vectorize.d1_delete("test_table", [])
        assert result == {"success": True, "changes": 0}

    def test_d1_metadata_query_requires_table_name(self) -> None:
        """Test that d1_metadata_query validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_metadata_query = (
            CloudflareVectorize.d1_metadata_query.__get__(
                mock_vectorize, CloudflareVectorize
            )
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_metadata_query("", {})

    def test_d1_metadata_query_empty_filters_returns_empty(self) -> None:
        """Test that d1_metadata_query handles empty filters."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        mock_vectorize.d1_metadata_query = (
            CloudflareVectorize.d1_metadata_query.__get__(
                mock_vectorize, CloudflareVectorize
            )
        )

        result = mock_vectorize.d1_metadata_query("test_table", {})
        assert result == []


class TestSQLAlchemyIntegration:
    """Test SQLAlchemy integration for SQL injection safety.

    These tests verify that the SQLAlchemy-based implementation properly
    handles potentially malicious input through parameterized queries.
    The actual SQL injection prevention is handled by SQLAlchemy's
    parameterized query system - these tests verify the data flows correctly.
    """

    def test_metadata_with_sql_injection_payload_serialized_safely(self) -> None:
        """Test that malicious metadata is safely serialized to JSON.

        SQLAlchemy handles the parameterization - we just verify the
        metadata is properly JSON-serialized before being passed to
        the database.
        """
        malicious_metadata = {"info": {"note": "x'); DROP TABLE docs; --"}}

        # The metadata should serialize to JSON without issues
        serialized = json.dumps(malicious_metadata)
        assert "DROP TABLE" in serialized  # Payload is preserved in JSON
        # But it's just a string value, not executable SQL

        # Verify it deserializes correctly
        deserialized = json.loads(serialized)
        assert deserialized["info"]["note"] == "x'); DROP TABLE docs; --"

    def test_nested_list_metadata_serialization(self) -> None:
        """Test that nested list metadata is safely serialized."""
        metadata = {"tags": ["safe", "x'); DELETE FROM users; --"]}

        serialized = json.dumps(metadata)
        deserialized = json.loads(serialized)

        assert deserialized["tags"][1] == "x'); DELETE FROM users; --"

    def test_record_with_special_characters_in_all_fields(self) -> None:
        """Test VectorizeRecord handles special characters in all fields."""
        record = VectorizeRecord(
            id="doc-'; DROP TABLE docs;--",
            text="SELECT * FROM users WHERE name = 'admin'; --",
            values=DUMMY_EMBEDDING,
            namespace="test'; DROP TABLE ns;--",
            metadata={
                "key": "value'; DROP TABLE meta;--",
                "nested": {"inner": "'); INSERT INTO hackers VALUES ('pwned');--"},
            },
        )

        record_dict = record.to_dict()

        # All values should be preserved as-is (SQLAlchemy handles safety)
        assert "DROP TABLE docs" in record_dict["id"]
        assert "SELECT * FROM users" in record_dict["text"]
        assert "DROP TABLE ns" in record_dict["namespace"]
        assert "DROP TABLE meta" in record_dict["metadata"]["key"]
        assert "INSERT INTO hackers" in record_dict["metadata"]["nested"]["inner"]

        # JSON serialization should work
        metadata_json = json.dumps(record_dict["metadata"])
        assert isinstance(metadata_json, str)
