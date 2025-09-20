"""Cloudflare Vectorize BaseStore implementation using langchain-cloudflare vector stores library."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langgraph.store.base import BaseStore, Item
from langchain_cloudflare.vectorstores import CloudflareVectorize
from langgraph.store.base import SearchItem

DEFAULT_TOP_K = 20


class CloudflareVectorizeBaseStore(BaseStore):
    """Cloudflare Vectorize implementation for LangGraph's BaseStore interface using langchain-cloudflare."""

    def __init__(
        self,
        account_id: str,
        index_name: str,
        api_token: str,
        embedding_function: callable,
        cf_vectorize: callable,
        dimensions: int = 768,
        base_url: str = "https://api.cloudflare.com/client/v4",
        timeout: float = 60.0,
    ) -> None:
        """Initialize CloudflareVectorizeBaseStore.

        Args:
            account_id: Cloudflare account ID
            index_name: Name of the Vectorize index
            api_token: Cloudflare API token
            embedding_function: Function to generate embeddings
            cf_vectorize: CloudflareVectorize instance
            dimensions: Number of dimensions for vectors (default: 1536)
            base_url: Base URL for Cloudflare API
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.account_id = account_id
        self.index_name = index_name
        self.api_token = api_token
        self.embedding_function = embedding_function
        self.dimensions = dimensions
        self.base_url = base_url
        self.timeout = timeout
        self.cf_vectorize = cf_vectorize

        self.vectorstore = CloudflareVectorize(
            account_id=account_id,
            index_name=index_name,
            api_token=api_token,
            embedding=embedding_function
        )

    @staticmethod
    def check_index_exists(index_list: List[Dict[str, Any]], index_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if an index with the given name exists in the list of indexes.

        Args:
            index_list: List of dictionaries containing index information
            index_name: Name of the index to search for

        Returns:
            Tuple of (exists: bool, index_info: dict or None)
        """
        for index in index_list:
            if index.get('name') == index_name:
                return True, index
        return False, None

    @classmethod
    def with_cloudflare_embeddings(
        cls,
        account_id: str,
        index_name: str,
        workers_ai_token: str,
        vectorize_api_token: str,
        embedding_model: str = "@cf/baai/bge-base-en-v1.5",
        dimensions: Optional[int] = None,
        description: Optional[str] = "A Test Vectorize Index",
        base_url: str = "https://api.cloudflare.com/client/v4",
        timeout: float = 60.0,
    ) -> CloudflareVectorizeBaseStore:
        """Convenience constructor that uses Cloudflare Workers AI embeddings and CloudflareVectorize.

        Args:
            account_id: Cloudflare account ID
            index_name: Name of the Vectorize index
            workers_ai_token: Token for Cloudflare Workers AI
            vectorize_api_token: Token for Vectorize service
            embedding_model: Model name for embeddings (default: "@cf/baai/bge-base-en-v1.5")
            dimensions: Number of dimensions (auto-detected if None)
            description: Index description
            base_url: Base URL for Cloudflare API
            timeout: Request timeout in seconds

        Returns:
            CloudflareVectorizeBaseStore instance

        Raises:
            ImportError: If langchain-cloudflare is not installed
        """
        try:
            from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
            from langchain_cloudflare import CloudflareVectorize
        except ImportError:
            raise ImportError(
                "langchain-cloudflare is required for Cloudflare embeddings. "
                "Install it with: pip install langchain-cloudflare"
            )

        # Auto-detect dimensions based on model if not provided
        if dimensions is None:
            model_dimensions = {
                "@cf/baai/bge-small-en-v1.5": 384,
                "@cf/baai/bge-base-en-v1.5": 768,
                "@cf/baai/bge-large-en-v1.5": 1024,
                "@cf/baai/bge-m3": 1024,
            }
            dimensions = model_dimensions.get(embedding_model, 768)

        # Initialize Cloudflare Workers AI embeddings
        embedder = CloudflareWorkersAIEmbeddings(
            account_id=account_id,
            api_token=workers_ai_token,
            model_name=embedding_model
        )

        # Initialize CloudflareVectorize
        cf_vectorize = CloudflareVectorize(
            embedding=embedder,
            account_id=account_id,
            vectorize_api_token=vectorize_api_token,  # Optional if using global token
        )

        indexes = cf_vectorize.list_indexes()
        index_exists = False
        for index in indexes:
            if index.get('name') == index_name:
                index_exists = True

        if not index_exists:
            cf_vectorize.create_index(
                index_name=index_name,
                description=description,
                dimensions=768,
                wait=True
            )

        return cls(
            account_id=account_id,
            index_name=index_name,
            api_token=vectorize_api_token,
            embedding_function=embedder,
            cf_vectorize=cf_vectorize,
            dimensions=dimensions,
            base_url=base_url,
            timeout=timeout
        )

    def _generate_vector_id(self, namespace: tuple, key: str) -> str:
        """Generate a consistent vector ID from namespace and key.

        Args:
            namespace: Namespace tuple
            key: Item key

        Returns:
            str: SHA256 hash of the combined namespace and key
        """
        namespace_str = "/".join(namespace)
        combined = f"{namespace_str}:{key}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _create_document(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None
    ) -> tuple[str, dict]:
        """Create a document for the vectorstore with proper metadata.

        Args:
            namespace: Namespace tuple
            key: Item key
            value: Item value
            index: Indexing configuration

        Returns:
            tuple: (text for embedding, metadata dict)
        """
        # Convert value to text for embedding
        if isinstance(value, dict):
            # Extract text for embedding based on index parameter
            if index is False:
                # Don't embed this item - use a placeholder
                text = f"Item: {key}"
            elif index is None or index is True:
                # Embed entire value
                text = json.dumps(value, ensure_ascii=False)
            elif isinstance(index, list):
                # Extract specified fields for embedding
                texts = []
                for field in index:
                    if field == "$":
                        texts.append(json.dumps(value, ensure_ascii=False))
                    elif field in value:
                        texts.append(str(value[field]))
                text = " ".join(texts) if texts else f"Item: {key}"
            else:
                text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)

        # Prepare metadata with namespace and key info
        namespace_str = "/".join(namespace)
        now = datetime.now().isoformat()

        metadata = {
            "namespace": namespace_str,
            "key": key,
            "data": json.dumps(value, ensure_ascii=False),
            "created_at": now,
            "updated_at": now,
            "vector_id": self._generate_vector_id(namespace, key)
        }

        return text, metadata

    def _extract_item_from_document(self, doc) -> tuple[tuple[str, ...], str, dict[str, Any], datetime, datetime]:
        """Extract BaseStore item components from a langchain document.

        Args:
            doc: Document from vectorstore

        Returns:
            tuple: (namespace, key, value, created_at, updated_at)
        """
        metadata = doc.metadata

        # Extract namespace and key
        namespace_str = metadata.get("namespace", "")
        namespace_tuple = tuple(namespace_str.split("/")) if namespace_str else ()
        key = metadata.get("key", "")

        # Extract value
        value_json = metadata.get("data", "{}")
        try:
            value = json.loads(value_json)
        except json.JSONDecodeError:
            value = {"content": value_json}

        # Extract timestamps
        created_at_str = metadata.get("created_at")
        updated_at_str = metadata.get("updated_at")

        try:
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
        except ValueError:
            created_at = datetime.now()

        try:
            updated_at = datetime.fromisoformat(updated_at_str) if updated_at_str else created_at
        except ValueError:
            updated_at = created_at

        return namespace_tuple, key, value, created_at, updated_at

    # LangGraph BaseStore interface methods
    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None
    ) -> Item | None:
        """Get a single item by namespace and key.

        Args:
            namespace: Namespace tuple
            key: Item key
            refresh_ttl: Whether to refresh TTL (not used in this implementation)

        Returns:
            Item if found, None otherwise
        """
        try:
            vector_id = self._generate_vector_id(namespace, key)

            # Search by vector_id in metadata
            docs = self.vectorstore.get_by_ids(
                [vector_id]
            )

            if docs:
                doc = docs[0]
                namespace_tuple, key_extracted, value, created_at, updated_at = self._extract_item_from_document(doc)

                return Item(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=created_at,
                    updated_at=updated_at
                )

            return None

        except Exception as e:
            return None

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None
    ) -> None:
        """Store or update a single item.

        Args:
            namespace: Namespace tuple
            key: Item key
            value: Item value
            index: Indexing configuration

        Raises:
            RuntimeError: If storage operation fails
        """
        try:
            # Check if item already exists
            existing_item = self.get(namespace, key)

            # Create document
            text, metadata = self._create_document(namespace, key, value, index)

            # Preserve created_at if item exists
            if existing_item:
                metadata["created_at"] = existing_item.created_at.isoformat()
                # Delete existing item first
                self.delete(namespace, key)

            # Generate vector ID
            vector_id = self._generate_vector_id(namespace, key)

            # Add the document to vectorstore
            from langchain_core.documents import Document
            doc = Document(page_content=text, metadata=metadata)

            self.vectorstore.add_documents([doc], ids=[vector_id])

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to store item: {str(e)}")

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete a single item.

        Args:
            namespace: Namespace tuple
            key: Item key
        """
        try:
            vector_id = self._generate_vector_id(namespace, key)
            self.vectorstore.delete([vector_id])
        except Exception as e:
            return None

    def search(
            self,
            namespace_prefix: tuple[str, ...],
            *,
            query: str | None = None,
            filter: dict[str, Any] | None = None,
            limit: int = DEFAULT_TOP_K,
            offset: int = 0,
            refresh_ttl: bool | None = None
    ) -> list[SearchItem]:
        """Search for items within a namespace prefix.

        Args:
            namespace_prefix: Namespace prefix tuple
            query: Search query string
            filter: Additional filters
            limit: Maximum number of results
            offset: Offset for pagination
            refresh_ttl: Whether to refresh TTL (not used in this implementation)

        Returns:
            List of SearchItem objects

        Raises:
            RuntimeError: If search operation fails
        """
        try:
            namespace_prefix_str = "/".join(namespace_prefix)

            # Build filter for namespace prefix
            search_filter = filter.copy() if filter else {}

            if query:
                # Semantic search with query
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=limit + offset,  # Get more to handle offset
                    md_filter=search_filter
                )
                docs_and_scores = [(doc, score) for doc, score in docs_with_scores]
            else:
                # Generate search data for vectorize
                if namespace_prefix_str:

                    docs_with_scores = self.vectorstore.similarity_search_with_score(
                        query=" ",  # Dummy query to get all results
                        k=limit + offset,
                        md_filter=search_filter
                    )
                    docs_and_scores = [(doc, score) for doc, score in docs_with_scores]

            # Filter by namespace prefix and apply pagination
            results = []
            processed_count = 0

            for doc, score in docs_and_scores:
                namespace_tuple, key, value, created_at, updated_at = self._extract_item_from_document(doc)

                # Check if namespace matches prefix
                namespace_str = "/".join(namespace_tuple)
                if namespace_str.startswith(namespace_prefix_str):
                    if processed_count >= offset:
                        results.append(SearchItem(
                            namespace=namespace_tuple,
                            key=key,
                            value=value,
                            created_at=created_at,
                            updated_at=updated_at,
                            score=score
                        ))

                        if len(results) >= limit:
                            break

                    processed_count += 1
            return results

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Search failed: {str(e)}")

    def batch(self, operations: list) -> list:
        """Execute multiple operations in a batch.

        Args:
            operations: List of operation dictionaries

        Returns:
            List of results corresponding to each operation
        """
        results = []

        for op in operations:
            op_type = op.get("type")

            try:
                if op_type == "get":
                    result = self.get(op["namespace"], op["key"])
                    results.append(result)
                elif op_type == "put":
                    self.put(
                        op["namespace"],
                        op["key"],
                        op["value"],
                        index=op.get("index")
                    )
                    results.append(None)
                elif op_type == "delete":
                    self.delete(op["namespace"], op["key"])
                    results.append(None)
                elif op_type == "search":
                    result = self.search(
                        op["namespace_prefix"],
                        query=op.get("query"),
                        filter=op.get("filter"),
                        limit=op.get("limit", DEFAULT_TOP_K),
                        offset=op.get("offset", 0)
                    )
                    results.append(result)
                else:
                    results.append(None)
            except Exception as e:
                results.append(None)

        return results

    # Async versions of the methods for compatibility
    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None
    ) -> Item | None:
        """Async version of get method.

        Args:
            namespace: Namespace tuple
            key: Item key
            refresh_ttl: Whether to refresh TTL (not used in this implementation)

        Returns:
            Item if found, None otherwise
        """
        return self.get(namespace, key, refresh_ttl=refresh_ttl)

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None
    ) -> None:
        """Async version of put method.

        Args:
            namespace: Namespace tuple
            key: Item key
            value: Item value
            index: Indexing configuration
        """
        self.put(namespace, key, value, index=index)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Async version of delete method.

        Args:
            namespace: Namespace tuple
            key: Item key
        """
        self.delete(namespace, key)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = DEFAULT_TOP_K,
        offset: int = 0,
        refresh_ttl: bool | None = None
    ) -> list[SearchItem]:
        """Async version of search method.

        Args:
            namespace_prefix: Namespace prefix tuple
            query: Search query string
            filter: Additional filters
            limit: Maximum number of results
            offset: Offset for pagination
            refresh_ttl: Whether to refresh TTL (not used in this implementation)

        Returns:
            List of SearchItem objects
        """
        return self.search(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl
        )

    async def abatch(self, operations: list) -> list:
        """Async version of batch method.

        Args:
            operations: List of operation dictionaries

        Returns:
            List of results corresponding to each operation
        """
        return self.batch(operations)