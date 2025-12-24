"""Test Cloudflare Vectorize.

In order to run this test, you need to:
1. Have a Cloudflare account
2. Set up API tokens with access to:
   - Workers AI
   - Vectorize
   - D1 (optional, for raw value storage)
3. Set environment variables in .env file:
   CF_ACCOUNT_ID
   CF_AI_API_TOKEN
   CF_VECTORIZE_API_TOKEN
   CF_D1_API_TOKEN
   CF_D1_DATABASE_ID
"""

import json
import os
import uuid
from pathlib import Path
from typing import Generator, List

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_cloudflare.embeddings import (
    CloudflareWorkersAIEmbeddings,
)
from langchain_cloudflare.vectorstores import CloudflareVectorize, VectorizeRecord

# Load environment variables from .env file in the integration_tests directory
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

MODEL_WORKERSAI = "@cf/baai/bge-large-en-v1.5"


@pytest.fixture(scope="class")
def embeddings() -> CloudflareWorkersAIEmbeddings:
    """Get embeddings model."""
    return CloudflareWorkersAIEmbeddings(
        account_id=os.getenv("CF_ACCOUNT_ID"),
        api_token=os.getenv("CF_AI_API_TOKEN"),
        model_name=MODEL_WORKERSAI,
    )


@pytest.fixture(scope="class")
def store(embeddings: CloudflareWorkersAIEmbeddings) -> Generator:
    index_name = f"test-langchain-{uuid.uuid4().hex}"

    store = CloudflareVectorize(
        embedding=embeddings,
        account_id=os.getenv("CF_ACCOUNT_ID", ""),
        d1_database_id=os.getenv("CF_D1_DATABASE_ID"),
        vectorize_api_token=os.getenv("CF_VECTORIZE_API_TOKEN"),
        d1_api_token=os.getenv("CF_D1_API_TOKEN"),
        index_name=index_name,
    )

    # Create the index
    store.create_index(wait=True)
    store.create_metadata_index(property_name="section", index_type="string", wait=True)
    store.add_documents(documents=TestCloudflareVectorize.documents, wait=True)

    yield store

    # Cleanup
    store.delete_index()


class TestCloudflareVectorize:
    """Test Cloudflare Vectorize functionality."""

    index_name = f"test-langchain-{uuid.uuid4().hex}"
    documents: List[Document] = [
        Document(
            page_content="Cloudflare's headquarters are in San Francisco, California.",
            metadata={"section": "Introduction"},
        ),
        Document(
            page_content="Cloudflare launched Workers AI, an AI inference platform.",
            metadata={"section": "Products"},
        ),
        Document(
            page_content="Cloudflare provides edge computing and CDN services.",
            metadata={"section": "Products"},
        ),
        Document(
            page_content="Cloudflare offers SASE and Zero Trust solutions.",
            metadata={"section": "Security"},
        ),
    ]

    def test_similarity_search(self, store: CloudflareVectorize) -> None:
        """Test similarity search."""
        docs = store.similarity_search(query="AI platform", k=2)
        assert len(docs) > 0
        assert any("Workers AI" in doc.page_content for doc in docs)

    def test_similarity_search_with_score(self, store: CloudflareVectorize) -> None:
        """Test similarity search with scores."""
        docs_and_scores = store.similarity_search_with_score(query="AI platform", k=2)
        assert len(docs_and_scores) > 0
        assert all(isinstance(item, tuple) for item in docs_and_scores)
        assert all(
            isinstance(item[0], Document) and isinstance(item[1], float)
            for item in docs_and_scores
        )
        assert any("Workers AI" in item[0].page_content for item in docs_and_scores)

    def test_similarity_search_with_metadata_filter(
        self, store: CloudflareVectorize
    ) -> None:
        """Test similarity search with metadata filtering."""
        docs = store.similarity_search(
            query="Cloudflare services",
            k=2,
            md_filter={"section": "Products"},
            return_metadata="all",
        )
        assert len(docs) > 0
        assert all(doc.metadata["section"] == "Products" for doc in docs)

    def test_get_by_ids(self, store: CloudflareVectorize) -> None:
        """Test retrieving documents by IDs."""
        # First get some IDs via search
        docs = store.similarity_search(query="California", k=1)
        doc_ids = list(set(doc.id for doc in docs if doc.id is not None))

        retrieved_docs = store.get_by_ids(doc_ids)
        retrieved_ids = set(doc.id for doc in retrieved_docs)

        assert retrieved_ids == set(doc_ids), (
            f"Retrieved IDs {retrieved_ids} don't match original IDs {doc_ids}"
        )

    def test_add_duplicates_and_upsert(self, store: CloudflareVectorize) -> None:
        """Test adding duplicate documents and upserting documents."""

        # Initial documents
        docs = [
            Document(
                id="test-id-1",
                page_content="This is a test document",
                metadata={"section": "Introduction"},
            ),
            Document(
                id="test-id-2",
                page_content="Another test document",
                metadata={"section": "Introduction"},
            ),
        ]

        # Add initial documents
        store.add_documents(documents=docs, wait=True)

        # Try adding same docs again with upsert=True (no duplicates)
        store.add_documents(documents=docs, upsert=True, wait=True)

        # Search to verify no duplicates
        results = store.get_by_ids(["test-id-1", "test-id-2"])

        assert len(results) == 2, "Should only have 2 documents despite adding twice"

        # Update document content
        updated_doc = Document(
            id="test-id-1",
            page_content="Updated: This is a test document",
            metadata={"section": "Introduction"},
        )

        # Upsert the updated document
        store.add_documents(documents=[updated_doc], upsert=True, wait=True)

        # Verify update
        results = store.get_by_ids(["test-id-1"])
        assert len(results) == 1
        assert results[0].page_content.startswith("Updated:")

    def test_delete_and_verify(self, store: CloudflareVectorize) -> None:
        """Test deleting documents and verifying they're gone."""
        test_id = uuid.uuid4().hex[:8]

        # Initial documents with unique IDs
        docs = [
            Document(
                id=f"{test_id}-delete-test-1",
                page_content="Document to delete 1",
                metadata={"section": "Test"},
            ),
            Document(
                id=f"{test_id}-delete-test-2",
                page_content="Document to delete 2",
                metadata={"section": "Test"},
            ),
            Document(
                id=f"{test_id}-keep-test-1",
                page_content="Document to keep",
                metadata={"section": "Test"},
            ),
        ]

        # Add documents
        store.add_documents(documents=docs, wait=True)

        # Verify initial state
        results = store.get_by_ids(
            [
                f"{test_id}-delete-test-1",
                f"{test_id}-delete-test-2",
                f"{test_id}-keep-test-1",
            ]
        )
        assert len(results) == 3, "Should have all 3 documents initially"

        # Delete specific documents
        ids_to_delete = [f"{test_id}-delete-test-1", f"{test_id}-delete-test-2"]
        store.delete(ids=ids_to_delete, wait=True)

        # Verify deletion
        results = store.get_by_ids(ids_to_delete)
        assert len(results) == 0

        # Verify remaining document
        results = store.get_by_ids([f"{test_id}-keep-test-1"])
        assert len(results) == 1
        assert results[0].id == f"{test_id}-keep-test-1"

    def test_similarity_search_with_namespace(self, store: CloudflareVectorize) -> None:
        """Test similarity search with namespace filtering."""
        # Create unique namespace for this test
        test_namespace = f"test-namespace-{uuid.uuid4().hex[:8]}"

        # Documents to add with the namespace
        namespace_docs = [
            Document(
                page_content="Cloudflare R2 provides S3-compatible object storage.",
                metadata={"section": "Products"},
            ),
            Document(
                page_content="Cloudflare Pages is a platform for frontend developers.",
                metadata={"section": "Products"},
            ),
        ]

        # Add documents with namespace
        store.add_documents(
            documents=namespace_docs,
            namespaces=[test_namespace] * len(namespace_docs),
            wait=True,
        )

        # Search within the namespace
        results = store.similarity_search(
            query="storage solution", k=2, namespace=test_namespace
        )

        # Verify results
        assert len(results) > 0
        assert any("R2" in doc.page_content for doc in results)

        # Verify namespace filtering works by searching with a different namespace
        other_namespace = f"nonexistent-namespace-{uuid.uuid4().hex[:8]}"
        empty_results = store.similarity_search(
            query="storage solution", k=2, namespace=other_namespace
        )

        # Should find no results in the other namespace
        assert len(empty_results) == 0

    def test_from_documents(self, embeddings: CloudflareWorkersAIEmbeddings) -> None:
        """Test creating store from documents."""
        new_index = f"test-langchain-{uuid.uuid4().hex}"
        try:
            store = CloudflareVectorize.from_documents(
                documents=self.documents,
                embedding=embeddings,
                account_id=os.getenv("CF_ACCOUNT_ID"),
                index_name=new_index,
                d1_database_id=os.getenv("CF_D1_DATABASE_ID"),
                vectorize_api_token=os.getenv("CF_VECTORIZE_API_TOKEN"),
                d1_api_token=os.getenv("CF_D1_API_TOKEN"),
                wait=True,
            )

            docs = store.similarity_search(
                query="California", k=1, index_name=new_index
            )
            assert len(docs) > 0
            assert "California" in docs[0].page_content

        finally:
            # Cleanup
            store.delete_index(new_index)


# =============================================================================
# D1 Database Integration Tests
# =============================================================================


@pytest.fixture(scope="class")
def d1_store(embeddings: CloudflareWorkersAIEmbeddings) -> Generator:
    """Fixture for D1-specific tests with a dedicated table."""
    index_name = f"test-d1-{uuid.uuid4().hex[:12]}"

    store = CloudflareVectorize(
        embedding=embeddings,
        account_id=os.getenv("CF_ACCOUNT_ID", ""),
        d1_database_id=os.getenv("CF_D1_DATABASE_ID"),
        vectorize_api_token=os.getenv("CF_VECTORIZE_API_TOKEN"),
        d1_api_token=os.getenv("CF_D1_API_TOKEN"),
        index_name=index_name,
    )

    yield store

    # Cleanup - drop D1 table if it exists
    try:
        store.d1_drop_table(index_name)
    except Exception:
        pass


class TestD1DatabaseOperations:
    """Test D1 database CRUD operations directly."""

    def test_d1_create_and_drop_table(self, d1_store: CloudflareVectorize) -> None:
        """Test creating and dropping a D1 table."""
        table_name = f"test_table_{uuid.uuid4().hex[:8]}"

        # Create table
        result = d1_store.d1_create_table(table_name)
        assert result["success"] is True

        # Drop table
        result = d1_store.d1_drop_table(table_name)
        assert result["success"] is True

    def test_d1_upsert_and_retrieve(self, d1_store: CloudflareVectorize) -> None:
        """Test inserting and retrieving records from D1."""
        table_name = f"test_upsert_{uuid.uuid4().hex[:8]}"

        try:
            # Create table
            d1_store.d1_create_table(table_name)

            # Create test records
            records = [
                VectorizeRecord(
                    id="doc-1",
                    text="First document content",
                    values=[0.1] * 10,
                    namespace="test",
                    metadata={"author": "Alice", "category": "tech"},
                ),
                VectorizeRecord(
                    id="doc-2",
                    text="Second document content",
                    values=[0.2] * 10,
                    namespace="test",
                    metadata={"author": "Bob", "category": "science"},
                ),
            ]

            # Insert records
            result = d1_store.d1_upsert_texts(table_name, records)
            assert result["success"] is True
            assert result["changes"] == 2

            # Retrieve by IDs
            retrieved = d1_store.d1_get_by_ids(table_name, ["doc-1", "doc-2"])
            assert len(retrieved) == 2

            # Verify content
            doc1 = next(r for r in retrieved if r["id"] == "doc-1")
            assert doc1["text"] == "First document content"
            assert doc1["namespace"] == "test"

            # Test upsert (update existing)
            updated_record = VectorizeRecord(
                id="doc-1",
                text="Updated first document",
                values=[0.3] * 10,
                namespace="test",
                metadata={"author": "Alice", "category": "updated"},
            )
            result = d1_store.d1_upsert_texts(table_name, [updated_record], upsert=True)
            assert result["success"] is True

            # Verify update
            retrieved = d1_store.d1_get_by_ids(table_name, ["doc-1"])
            assert len(retrieved) == 1
            assert retrieved[0]["text"] == "Updated first document"

        finally:
            d1_store.d1_drop_table(table_name)

    def test_d1_delete_records(self, d1_store: CloudflareVectorize) -> None:
        """Test deleting records from D1."""
        table_name = f"test_delete_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Insert records
            records = [
                VectorizeRecord(
                    id=f"delete-{i}",
                    text=f"Document {i}",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={},
                )
                for i in range(5)
            ]
            d1_store.d1_upsert_texts(table_name, records)

            # Verify all records exist
            all_ids = [f"delete-{i}" for i in range(5)]
            retrieved = d1_store.d1_get_by_ids(table_name, all_ids)
            assert len(retrieved) == 5

            # Delete some records
            ids_to_delete = ["delete-1", "delete-3"]
            result = d1_store.d1_delete(table_name, ids_to_delete)
            assert result["success"] is True

            # Verify deletion
            retrieved = d1_store.d1_get_by_ids(table_name, all_ids)
            assert len(retrieved) == 3
            remaining_ids = {r["id"] for r in retrieved}
            assert remaining_ids == {"delete-0", "delete-2", "delete-4"}

        finally:
            d1_store.d1_drop_table(table_name)

    def test_d1_metadata_query(self, d1_store: CloudflareVectorize) -> None:
        """Test querying D1 by metadata filters."""
        table_name = f"test_metadata_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Insert records with varied metadata
            records = [
                VectorizeRecord(
                    id="meta-1",
                    text="Tech article by Alice",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"author": "Alice", "category": "tech"},
                ),
                VectorizeRecord(
                    id="meta-2",
                    text="Science article by Bob",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"author": "Bob", "category": "science"},
                ),
                VectorizeRecord(
                    id="meta-3",
                    text="Tech article by Bob",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"author": "Bob", "category": "tech"},
                ),
                VectorizeRecord(
                    id="meta-4",
                    text="Art article by Charlie",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"author": "Charlie", "category": "art"},
                ),
            ]
            d1_store.d1_upsert_texts(table_name, records)

            # Query by single metadata field (AND operation)
            results = d1_store.d1_metadata_query(
                table_name,
                {"category": ["tech"]},
                operation="AND",
            )
            assert len(results) == 2
            result_ids = {r["id"] for r in results}
            assert result_ids == {"meta-1", "meta-3"}

            # Query by multiple values (OR within field)
            results = d1_store.d1_metadata_query(
                table_name,
                {"author": ["Alice", "Charlie"]},
                operation="AND",
            )
            assert len(results) == 2
            result_ids = {r["id"] for r in results}
            assert result_ids == {"meta-1", "meta-4"}

            # Query by multiple fields (AND operation)
            results = d1_store.d1_metadata_query(
                table_name,
                {"author": ["Bob"], "category": ["tech"]},
                operation="AND",
            )
            assert len(results) == 1
            assert results[0]["id"] == "meta-3"

            # Query with OR operation across fields
            results = d1_store.d1_metadata_query(
                table_name,
                {"author": ["Alice"], "category": ["science"]},
                operation="OR",
            )
            assert len(results) == 2
            result_ids = {r["id"] for r in results}
            assert result_ids == {"meta-1", "meta-2"}

        finally:
            d1_store.d1_drop_table(table_name)


# =============================================================================
# SQL Injection Prevention Integration Tests
# =============================================================================


class TestSQLInjectionPrevention:
    """Integration tests to verify SQL injection prevention in D1 operations.

    These tests attempt various SQL injection payloads against a real D1 database
    to confirm that the protections work end-to-end.
    """

    def test_sql_injection_in_text_content(self, d1_store: CloudflareVectorize) -> None:
        """Test that SQL injection in text content is safely stored and retrieved."""
        table_name = f"test_sqli_text_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Malicious payloads in text content
            malicious_texts = [
                "Robert'); DROP TABLE students;--",
                "SELECT * FROM users WHERE '1'='1",
                "'; DELETE FROM documents; --",
                "1 OR 1=1; --",
                "admin'--",
                "' UNION SELECT * FROM passwords --",
            ]

            records = [
                VectorizeRecord(
                    id=f"sqli-text-{i}",
                    text=payload,
                    values=[0.1] * 10,
                    namespace="",
                    metadata={},
                )
                for i, payload in enumerate(malicious_texts)
            ]

            # Insert should succeed (payloads are just data)
            result = d1_store.d1_upsert_texts(table_name, records)
            assert result["success"] is True
            assert result["changes"] == len(malicious_texts)

            # Retrieve and verify payloads are stored verbatim
            ids = [f"sqli-text-{i}" for i in range(len(malicious_texts))]
            retrieved = d1_store.d1_get_by_ids(table_name, ids)
            assert len(retrieved) == len(malicious_texts)

            retrieved_texts = {r["text"] for r in retrieved}
            for payload in malicious_texts:
                assert payload in retrieved_texts, f"Payload not stored: {payload}"

        finally:
            d1_store.d1_drop_table(table_name)

    def test_sql_injection_in_metadata_values(
        self, d1_store: CloudflareVectorize
    ) -> None:
        """Test that SQL injection in metadata values is safely handled."""
        table_name = f"test_sqli_meta_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Malicious metadata values
            malicious_metadata = {
                "author": "'; DROP TABLE users; --",
                "title": "SELECT * FROM secrets",
                "nested": {
                    "value": "1'); DELETE FROM data WHERE ('1'='1",
                    "array": ["safe", "'; INSERT INTO hackers VALUES ('pwned'); --"],
                },
            }

            record = VectorizeRecord(
                id="sqli-meta-1",
                text="Test document",
                values=[0.1] * 10,
                namespace="",
                metadata=malicious_metadata,
            )

            # Insert should succeed
            result = d1_store.d1_upsert_texts(table_name, [record])
            assert result["success"] is True

            # Retrieve and verify metadata is stored correctly
            retrieved = d1_store.d1_get_by_ids(table_name, ["sqli-meta-1"])
            assert len(retrieved) == 1

            # Parse the stored metadata
            stored_meta = json.loads(retrieved[0]["metadata"])
            assert stored_meta["author"] == "'; DROP TABLE users; --"
            assert (
                stored_meta["nested"]["value"] == "1'); DELETE FROM data WHERE ('1'='1"
            )

        finally:
            d1_store.d1_drop_table(table_name)

    def test_sql_injection_in_document_ids(self, d1_store: CloudflareVectorize) -> None:
        """Test that SQL injection in document IDs is safely handled."""
        table_name = f"test_sqli_id_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Malicious IDs
            malicious_ids = [
                "id'; DROP TABLE test;--",
                "id' OR '1'='1",
                'id"; DELETE FROM data; --',
            ]

            records = [
                VectorizeRecord(
                    id=mal_id,
                    text=f"Document with malicious ID: {mal_id}",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={},
                )
                for mal_id in malicious_ids
            ]

            # Insert should succeed
            result = d1_store.d1_upsert_texts(table_name, records)
            assert result["success"] is True

            # Retrieve by malicious IDs should work
            retrieved = d1_store.d1_get_by_ids(table_name, malicious_ids)
            assert len(retrieved) == len(malicious_ids)

            # Delete by malicious IDs should work
            result = d1_store.d1_delete(table_name, malicious_ids[:1])
            assert result["success"] is True

            # Verify only targeted record deleted
            retrieved = d1_store.d1_get_by_ids(table_name, malicious_ids)
            assert len(retrieved) == len(malicious_ids) - 1

        finally:
            d1_store.d1_drop_table(table_name)

    def test_metadata_query_rejects_invalid_keys(
        self, d1_store: CloudflareVectorize
    ) -> None:
        """Test that metadata query rejects keys with special characters."""
        table_name = f"test_sqli_key_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Insert a valid record first
            record = VectorizeRecord(
                id="valid-1",
                text="Valid document",
                values=[0.1] * 10,
                namespace="",
                metadata={"valid_key": "value"},
            )
            d1_store.d1_upsert_texts(table_name, [record])

            # These malicious keys should be rejected
            injection_keys = [
                "key'); DROP TABLE test;--",
                "key' OR '1'='1",
                "key.nested",  # dots not allowed
                "key-name",  # hyphens not allowed
                "key name",  # spaces not allowed
            ]

            for bad_key in injection_keys:
                with pytest.raises(ValueError, match="Invalid metadata key"):
                    d1_store.d1_metadata_query(
                        table_name,
                        {bad_key: ["value"]},
                        operation="AND",
                    )

        finally:
            d1_store.d1_drop_table(table_name)

    def test_metadata_query_rejects_invalid_operation(
        self, d1_store: CloudflareVectorize
    ) -> None:
        """Test that metadata query rejects invalid SQL operations."""
        table_name = f"test_sqli_op_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Insert a valid record
            record = VectorizeRecord(
                id="valid-1",
                text="Valid document",
                values=[0.1] * 10,
                namespace="",
                metadata={"category": "test"},
            )
            d1_store.d1_upsert_texts(table_name, [record])

            # These malicious operations should be rejected
            invalid_operations = [
                "AND; DROP TABLE test; --",
                "OR 1=1; --",
                "UNION",
                "and",  # lowercase should fail
                "or",  # lowercase should fail
            ]

            for bad_op in invalid_operations:
                with pytest.raises(ValueError, match="operation must be 'AND' or 'OR'"):
                    d1_store.d1_metadata_query(
                        table_name,
                        {"category": ["test"]},
                        operation=bad_op,
                    )

        finally:
            d1_store.d1_drop_table(table_name)

    def test_metadata_query_with_malicious_values(
        self, d1_store: CloudflareVectorize
    ) -> None:
        """Test that metadata query safely handles malicious filter values."""
        table_name = f"test_sqli_val_{uuid.uuid4().hex[:8]}"

        try:
            d1_store.d1_create_table(table_name)

            # Insert records with normal metadata
            records = [
                VectorizeRecord(
                    id="normal-1",
                    text="Normal document",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"category": "tech", "author": "Alice"},
                ),
                VectorizeRecord(
                    id="normal-2",
                    text="Another document",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"category": "science", "author": "Bob"},
                ),
            ]
            d1_store.d1_upsert_texts(table_name, records)

            # Query with malicious filter values (should be treated as data)
            malicious_values = [
                "'; DROP TABLE test; --",
                "' OR '1'='1",
                "tech' UNION SELECT * FROM passwords --",
            ]

            # These should execute safely and return no results
            # (since no records have these exact metadata values)
            for mal_val in malicious_values:
                results = d1_store.d1_metadata_query(
                    table_name,
                    {"category": [mal_val]},
                    operation="AND",
                )
                # Should return empty (no match) rather than error or injection
                assert results == [], (
                    f"Unexpected results for malicious value: {mal_val}"
                )

            # Verify the table still exists and has correct data
            retrieved = d1_store.d1_get_by_ids(table_name, ["normal-1", "normal-2"])
            assert len(retrieved) == 2

        finally:
            d1_store.d1_drop_table(table_name)


# =============================================================================
# Async D1 Integration Tests
# =============================================================================


@pytest.fixture(scope="class")
def async_d1_store(embeddings: CloudflareWorkersAIEmbeddings) -> Generator:
    """Fixture for async D1 tests."""
    index_name = f"test-async-d1-{uuid.uuid4().hex[:12]}"

    store = CloudflareVectorize(
        embedding=embeddings,
        account_id=os.getenv("CF_ACCOUNT_ID", ""),
        d1_database_id=os.getenv("CF_D1_DATABASE_ID"),
        vectorize_api_token=os.getenv("CF_VECTORIZE_API_TOKEN"),
        d1_api_token=os.getenv("CF_D1_API_TOKEN"),
        index_name=index_name,
    )

    yield store

    # Cleanup
    try:
        store.d1_drop_table(index_name)
    except Exception:
        pass


class TestAsyncD1Operations:
    """Test async D1 database operations using SQLAlchemy create_async_engine()."""

    @pytest.mark.asyncio
    async def test_async_d1_create_and_drop_table(
        self, async_d1_store: CloudflareVectorize
    ) -> None:
        """Test async creating and dropping a D1 table."""
        table_name = f"test_async_table_{uuid.uuid4().hex[:8]}"

        # Create table
        result = await async_d1_store.ad1_create_table(table_name)
        assert result["success"] is True

        # Drop table
        result = await async_d1_store.ad1_drop_table(table_name)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_async_d1_upsert_and_retrieve(
        self, async_d1_store: CloudflareVectorize
    ) -> None:
        """Test async inserting and retrieving records from D1."""
        table_name = f"test_async_upsert_{uuid.uuid4().hex[:8]}"

        try:
            await async_d1_store.ad1_create_table(table_name)

            # Create test records
            records = [
                VectorizeRecord(
                    id="async-doc-1",
                    text="First async document",
                    values=[0.1] * 10,
                    namespace="async-test",
                    metadata={"source": "test"},
                ),
                VectorizeRecord(
                    id="async-doc-2",
                    text="Second async document",
                    values=[0.2] * 10,
                    namespace="async-test",
                    metadata={"source": "test"},
                ),
            ]

            # Insert records
            result = await async_d1_store.ad1_upsert_texts(table_name, records)
            assert result["success"] is True
            assert result["changes"] == 2

            # Retrieve by IDs
            retrieved = await async_d1_store.ad1_get_by_ids(
                table_name, ["async-doc-1", "async-doc-2"]
            )
            assert len(retrieved) == 2

            # Verify content
            doc1 = next(r for r in retrieved if r["id"] == "async-doc-1")
            assert doc1["text"] == "First async document"

        finally:
            await async_d1_store.ad1_drop_table(table_name)

    @pytest.mark.asyncio
    async def test_async_d1_delete(self, async_d1_store: CloudflareVectorize) -> None:
        """Test async deleting records from D1."""
        table_name = f"test_async_delete_{uuid.uuid4().hex[:8]}"

        try:
            await async_d1_store.ad1_create_table(table_name)

            records = [
                VectorizeRecord(
                    id=f"async-del-{i}",
                    text=f"Document {i}",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={},
                )
                for i in range(3)
            ]
            await async_d1_store.ad1_upsert_texts(table_name, records)

            # Delete one record
            result = await async_d1_store.ad1_delete(table_name, ["async-del-1"])
            assert result["success"] is True

            # Verify deletion
            retrieved = await async_d1_store.ad1_get_by_ids(
                table_name, ["async-del-0", "async-del-1", "async-del-2"]
            )
            assert len(retrieved) == 2
            remaining_ids = {r["id"] for r in retrieved}
            assert remaining_ids == {"async-del-0", "async-del-2"}

        finally:
            await async_d1_store.ad1_drop_table(table_name)

    @pytest.mark.asyncio
    async def test_async_d1_metadata_query(
        self, async_d1_store: CloudflareVectorize
    ) -> None:
        """Test async querying D1 by metadata filters."""
        table_name = f"test_async_meta_{uuid.uuid4().hex[:8]}"

        try:
            await async_d1_store.ad1_create_table(table_name)

            records = [
                VectorizeRecord(
                    id="async-meta-1",
                    text="Tech article",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"category": "tech"},
                ),
                VectorizeRecord(
                    id="async-meta-2",
                    text="Science article",
                    values=[0.1] * 10,
                    namespace="",
                    metadata={"category": "science"},
                ),
            ]
            await async_d1_store.ad1_upsert_texts(table_name, records)

            # Query by metadata
            results = await async_d1_store.ad1_metadata_query(
                table_name,
                {"category": ["tech"]},
                operation="AND",
            )
            assert len(results) == 1
            assert results[0]["id"] == "async-meta-1"

        finally:
            await async_d1_store.ad1_drop_table(table_name)
