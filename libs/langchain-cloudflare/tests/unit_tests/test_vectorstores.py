"""Unit tests for CloudflareVectorize vectorstores.

These tests verify the D1 SQLAlchemy integration and helper methods.
SQL injection prevention is handled by SQLAlchemy's parameterized queries.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from langchain_cloudflare.vectorstores import (
    CloudflareVectorize,
    VectorizeRecord,
    _index_is_ready,
)

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


def _setup_mock_with_validators(mock_vectorize: MagicMock) -> None:
    """Add static validator methods to a mock CloudflareVectorize instance."""
    mock_vectorize._validate_table_name = CloudflareVectorize._validate_table_name
    mock_vectorize._validate_operation = CloudflareVectorize._validate_operation
    mock_vectorize._validate_metadata_key = CloudflareVectorize._validate_metadata_key
    mock_vectorize._build_metadata_filter_query = (
        CloudflareVectorize._build_metadata_filter_query
    )
    mock_vectorize._rows_to_dicts = CloudflareVectorize._rows_to_dicts
    mock_vectorize._prepare_records_for_insert = (
        CloudflareVectorize._prepare_records_for_insert
    )


class TestD1MethodValidation:
    """Test D1 method input validation."""

    def test_d1_create_table_requires_table_name(self) -> None:
        """Test that d1_create_table validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_create_table = CloudflareVectorize.d1_create_table.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_create_table("")

    def test_d1_create_table_rejects_unsafe_table_name(self) -> None:
        """Test that d1_create_table rejects SQL injection in table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_create_table = CloudflareVectorize.d1_create_table.__get__(
            mock_vectorize, CloudflareVectorize
        )

        # Various SQL injection attempts
        injection_names = [
            "users; DROP TABLE users; --",
            "table'; DELETE FROM users; --",
            "test`injection",
            "table name with spaces",
            "table.with.dots",
        ]

        for bad_name in injection_names:
            with pytest.raises(ValueError, match="Invalid table_name"):
                mock_vectorize.d1_create_table(bad_name)

    def test_d1_create_table_accepts_valid_table_names(self) -> None:
        """Test that d1_create_table accepts valid table names."""
        # Valid names should not raise during validation
        # (they'll fail at engine creation which we're not testing here)
        valid_names = [
            "users",
            "my_table",
            "table123",
            "Test_Table_2",
            "lang-chain-docs",
        ]

        for name in valid_names:
            # Should not raise ValueError for valid names
            CloudflareVectorize._validate_table_name(name)

    def test_d1_drop_table_requires_table_name(self) -> None:
        """Test that d1_drop_table validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_drop_table = CloudflareVectorize.d1_drop_table.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_drop_table("")

    def test_d1_upsert_texts_requires_table_name(self) -> None:
        """Test that d1_upsert_texts validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_upsert_texts = CloudflareVectorize.d1_upsert_texts.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_upsert_texts("", [])

    def test_d1_upsert_texts_empty_data_returns_success(self) -> None:
        """Test that d1_upsert_texts handles empty data."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_upsert_texts = CloudflareVectorize.d1_upsert_texts.__get__(
            mock_vectorize, CloudflareVectorize
        )

        result = mock_vectorize.d1_upsert_texts("test_table", [])
        assert result == {"success": True, "changes": 0}

    def test_d1_get_by_ids_requires_table_name(self) -> None:
        """Test that d1_get_by_ids validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_get_by_ids = CloudflareVectorize.d1_get_by_ids.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_get_by_ids("", ["id1"])

    def test_d1_get_by_ids_empty_ids_returns_empty(self) -> None:
        """Test that d1_get_by_ids handles empty IDs list."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_get_by_ids = CloudflareVectorize.d1_get_by_ids.__get__(
            mock_vectorize, CloudflareVectorize
        )

        result = mock_vectorize.d1_get_by_ids("test_table", [])
        assert result == []

    def test_d1_delete_requires_table_name(self) -> None:
        """Test that d1_delete validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_delete = CloudflareVectorize.d1_delete.__get__(
            mock_vectorize, CloudflareVectorize
        )

        with pytest.raises(ValueError, match="table_name must be provided"):
            mock_vectorize.d1_delete("", ["id1"])

    def test_d1_delete_empty_ids_returns_success(self) -> None:
        """Test that d1_delete handles empty IDs list."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_delete = CloudflareVectorize.d1_delete.__get__(
            mock_vectorize, CloudflareVectorize
        )

        result = mock_vectorize.d1_delete("test_table", [])
        assert result == {"success": True, "changes": 0}

    def test_d1_metadata_query_requires_table_name(self) -> None:
        """Test that d1_metadata_query validates table_name."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
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
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize.d1_metadata_query = (
            CloudflareVectorize.d1_metadata_query.__get__(
                mock_vectorize, CloudflareVectorize
            )
        )

        result = mock_vectorize.d1_metadata_query("test_table", {})
        assert result == []


class TestMetadataQuerySQLInjectionPrevention:
    """Test SQL injection prevention in d1_metadata_query methods."""

    def test_d1_metadata_query_rejects_invalid_operation(self) -> None:
        """Test that d1_metadata_query rejects SQL injection in operation param."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize._get_d1_engine = MagicMock()
        mock_vectorize._get_d1_table = CloudflareVectorize._get_d1_table.__get__(
            mock_vectorize, CloudflareVectorize
        )
        mock_vectorize.d1_metadata_query = (
            CloudflareVectorize.d1_metadata_query.__get__(
                mock_vectorize, CloudflareVectorize
            )
        )

        # Try SQL injection through operation parameter
        with pytest.raises(ValueError, match="operation must be 'AND' or 'OR'"):
            mock_vectorize.d1_metadata_query(
                "test_table",
                {"key": ["value"]},
                operation="AND; DROP TABLE users; --",
            )

    def test_d1_metadata_query_rejects_invalid_metadata_key(self) -> None:
        """Test that d1_metadata_query rejects SQL injection in metadata keys."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize._get_d1_engine = MagicMock()
        mock_vectorize._get_d1_table = CloudflareVectorize._get_d1_table.__get__(
            mock_vectorize, CloudflareVectorize
        )
        mock_vectorize.d1_metadata_query = (
            CloudflareVectorize.d1_metadata_query.__get__(
                mock_vectorize, CloudflareVectorize
            )
        )

        # Try SQL injection through metadata key
        with pytest.raises(ValueError, match="Invalid metadata key"):
            mock_vectorize.d1_metadata_query(
                "test_table",
                {"key'); DROP TABLE users; --": ["value"]},
                operation="AND",
            )

    def test_d1_metadata_query_accepts_valid_alphanumeric_keys(self) -> None:
        """Test that d1_metadata_query accepts valid alphanumeric keys."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_engine.connect.return_value = mock_conn

        mock_vectorize._get_d1_engine = MagicMock(return_value=mock_engine)
        mock_vectorize._get_d1_table = CloudflareVectorize._get_d1_table.__get__(
            mock_vectorize, CloudflareVectorize
        )
        mock_vectorize.d1_metadata_query = (
            CloudflareVectorize.d1_metadata_query.__get__(
                mock_vectorize, CloudflareVectorize
            )
        )

        # Should not raise for valid keys
        result = mock_vectorize.d1_metadata_query(
            "test_table",
            {
                "author_name": ["John"],
                "category2": ["books"],
                "tag_123": ["fiction"],
            },
            operation="AND",
        )
        assert result == []

    def test_d1_metadata_query_rejects_special_chars_in_key(self) -> None:
        """Test that metadata keys with special characters are rejected."""
        mock_vectorize = MagicMock(spec=CloudflareVectorize)
        _setup_mock_with_validators(mock_vectorize)
        mock_vectorize._get_d1_engine = MagicMock()
        mock_vectorize._get_d1_table = CloudflareVectorize._get_d1_table.__get__(
            mock_vectorize, CloudflareVectorize
        )
        mock_vectorize.d1_metadata_query = (
            CloudflareVectorize.d1_metadata_query.__get__(
                mock_vectorize, CloudflareVectorize
            )
        )

        # Various injection attempts through metadata keys
        injection_keys = [
            "key.nested",  # dot
            "key-name",  # hyphen
            "key name",  # space
            'key"; DROP TABLE',  # quote
            "key' OR '1'='1",  # quote
            "key`test",  # backtick
            "key;test",  # semicolon
        ]

        for bad_key in injection_keys:
            with pytest.raises(ValueError, match="Invalid metadata key"):
                mock_vectorize.d1_metadata_query(
                    "test_table",
                    {bad_key: ["value"]},
                    operation="AND",
                )


class _FakeEmbeddings:
    """Minimal Embeddings stand-in -- avoids a real REST/Workers AI call."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [DUMMY_EMBEDDING for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return DUMMY_EMBEDDING

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class _FakeVectorizeBinding:
    """Minimal env.VECTORIZE binding stand-in, recording what it's given."""

    def __init__(self) -> None:
        self.upserted: list[dict] = []
        self.inserted: list[dict] = []

    async def upsert(self, vectors: list[dict]) -> dict:
        self.upserted.extend(vectors)
        return {"mutationId": "fake", "count": len(vectors)}

    async def insert(self, vectors: list[dict]) -> dict:
        self.inserted.extend(vectors)
        return {"mutationId": "fake", "count": len(vectors)}


class TestAaddDocumentsHonorsExplicitIds:
    """Regression test: aadd_documents(documents, ids=[...]) must use those ids.

    Before the fix, `ids` was accepted as an explicit named parameter but the
    body checked `"ids" not in kwargs` to decide whether to generate random
    UUIDs instead -- since `ids` was consumed by the named parameter, it
    never appeared in kwargs, so that check was always true and any
    caller-supplied ids were silently discarded. The sync add_documents()
    doesn't have this bug: it never took `ids` as a named parameter, so its
    identical-looking kwargs check is actually correct there.
    """

    async def test_aadd_documents_uses_given_ids_not_random_uuids(self) -> None:
        from langchain_core.documents import Document

        binding = _FakeVectorizeBinding()
        cf_vectorize = CloudflareVectorize(
            embedding=_FakeEmbeddings(),  # type: ignore[arg-type]
            binding=binding,
            index_name="test-index",
        )

        doc = Document(page_content="hello world", metadata={"k": "v"})
        returned_ids = await cf_vectorize.aadd_documents([doc], ids=["my-explicit-id"])

        assert returned_ids == ["my-explicit-id"]
        assert len(binding.inserted) == 1
        assert binding.inserted[0]["id"] == "my-explicit-id"

    async def test_aadd_documents_generates_id_when_none_given(self) -> None:
        from langchain_core.documents import Document

        binding = _FakeVectorizeBinding()
        cf_vectorize = CloudflareVectorize(
            embedding=_FakeEmbeddings(),  # type: ignore[arg-type]
            binding=binding,
            index_name="test-index",
        )

        doc = Document(page_content="hello world", metadata={"k": "v"})
        returned_ids = await cf_vectorize.aadd_documents([doc])

        assert len(returned_ids) == 1
        assert returned_ids[0]  # non-empty, auto-generated
        assert binding.inserted[0]["id"] == returned_ids[0]


class TestIndexIsReady:
    """Regression test: _index_is_ready must match the actual `/info` shape.

    Before the fix it checked `index_info.get("name") and
    index_info.get("config")`, fields that belong to the `/indexes/{name}`
    endpoint's response -- but _index_is_ready is only ever called with the
    output of get_index_info()/aget_index_info(), which hit `/info` instead.
    `/info` never returns `name` or `config`, so this always returned False,
    meaning create_index(wait=True) could never detect "ready" and would
    poll until _poll_mutation_status's timeout fired (previously: it hung
    forever, since that loop had no timeout of its own either).
    """

    def test_ready_for_a_fresh_empty_index(self) -> None:
        # Actual /info response for a just-created, zero-vector index.
        assert _index_is_ready({"dimensions": 768, "vectorCount": 0}) is True

    def test_ready_for_an_index_with_processed_mutations(self) -> None:
        # Actual /info response shape once at least one mutation has landed.
        assert (
            _index_is_ready(
                {
                    "dimensions": 768,
                    "vectorCount": 1,
                    "processedUpToDatetime": "2026-08-13T04:31:31.713Z",
                    "processedUpToMutation": "f3346a16-5900-44aa-917c-9204b2cedda6",
                }
            )
            is True
        )

    def test_not_ready_for_an_empty_response(self) -> None:
        assert _index_is_ready({}) is False

    def test_not_ready_for_the_indexes_endpoint_shape(self) -> None:
        # /indexes/{name}'s response shape -- not what get_index_info() ever
        # actually returns, but worth pinning that this alone isn't "ready"
        # since it lacks `dimensions`.
        assert (
            _index_is_ready(
                {"name": "my-index", "config": {"dimensions": 768, "metric": "cosine"}}
            )
            is False
        )


class TestPollMutationStatusTimeout:
    """Regression test: _poll_mutation_status/_apoll_mutation_status must not
    hang forever.

    Before the fix, both were `while True:` loops with no maximum duration,
    and got_index_info()'s underlying requests.get() had no timeout either --
    so a single stuck/slow API response left the loop spinning silently with
    no error, ever. Observed in practice as a 45+ minute hang on
    create_index(wait=True) during integration test development.
    """

    def test_poll_mutation_status_raises_timeout_error_past_max_wait(self) -> None:
        cf_vectorize = CloudflareVectorize(
            embedding=_FakeEmbeddings(),  # type: ignore[arg-type]
            account_id="acct",
            api_token="tok",
            index_name="test-index",
        )
        # Never looks "ready" and never matches a mutation_id, so the only
        # way out of the loop is the max_wait_seconds deadline.
        cf_vectorize.get_index_info = MagicMock(return_value={})  # type: ignore[method-assign]

        clock = {"now": 0.0}

        def fake_monotonic() -> float:
            clock["now"] += 5
            return clock["now"]

        with (
            patch("langchain_cloudflare.vectorstores.time.sleep"),
            patch("langchain_cloudflare.vectorstores.time.monotonic", fake_monotonic),
        ):
            with pytest.raises(TimeoutError, match="Timed out waiting"):
                cf_vectorize._poll_mutation_status(
                    index_name="test-index", max_wait_seconds=10
                )

    async def test_apoll_mutation_status_raises_timeout_error_past_max_wait(
        self,
    ) -> None:
        cf_vectorize = CloudflareVectorize(
            embedding=_FakeEmbeddings(),  # type: ignore[arg-type]
            account_id="acct",
            api_token="tok",
            index_name="test-index",
        )

        async def fake_aget_index_info(*args: object, **kwargs: object) -> dict:
            return {}

        cf_vectorize.aget_index_info = fake_aget_index_info  # type: ignore[method-assign]

        clock = {"now": 0.0}

        def fake_monotonic() -> float:
            clock["now"] += 5
            return clock["now"]

        async def fake_sleep(*args: object, **kwargs: object) -> None:
            return None

        with (
            patch("langchain_cloudflare.vectorstores.asyncio.sleep", fake_sleep),
            patch("langchain_cloudflare.vectorstores.time.monotonic", fake_monotonic),
        ):
            with pytest.raises(TimeoutError, match="Timed out waiting"):
                await cf_vectorize._apoll_mutation_status(
                    index_name="test-index", max_wait_seconds=10
                )


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
