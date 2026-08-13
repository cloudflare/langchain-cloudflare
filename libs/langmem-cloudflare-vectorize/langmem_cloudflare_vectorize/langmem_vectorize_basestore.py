"""Cloudflare Vectorize BaseStore implementation using langchain-cloudflare vector stores library."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from langchain_cloudflare.vectorstores import CloudflareVectorize
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    NotProvided,
    PutOp,
    SearchItem,
    SearchOp,
)

DEFAULT_TOP_K = 20

T = TypeVar("T")


class CloudflareVectorizeBaseStore(BaseStore):
    """Cloudflare Vectorize implementation for LangGraph's BaseStore interface using langchain-cloudflare.

    Supports both the Cloudflare REST API (the default) and, inside a
    Cloudflare Python Worker, the native Vectorize/D1 bindings -- pass
    `binding` (e.g. `self.env.VECTORIZE`) and optionally `d1_binding`
    (e.g. `self.env.D1`) instead of `account_id`/`api_token`. The
    `embedding_function` you pass should itself be binding-aware too (e.g.
    `CloudflareWorkersAIEmbeddings(binding=self.env.AI)`) for a fully
    binding-based Worker setup with no REST calls at all.

    `langchain_cloudflare.vectorstores.CloudflareVectorize`'s binding support
    is async-only -- its sync methods (`similarity_search_with_score`,
    `get_by_ids`, `add_documents`, `delete`) always use REST regardless of
    `binding`. So when `binding` is set, this class's sync methods (`get`,
    `put`, `delete`, `search`, `batch`) bridge to their async counterparts via
    `pyodide.ffi.run_sync()` -- the same mechanism
    `sqlalchemy_cloudflare_d1.SyncWorkerConnection` uses for SQLAlchemy's sync
    engine, and `WorkerCloudflareD1Saver` uses in `langgraph-checkpoint-cloudflare-d1`
    -- which only works from inside a live Worker request. Without a binding,
    the sync methods are REST calls made directly on the calling thread, same
    as before.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        index_name: Optional[str] = None,
        api_token: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        cf_vectorize: Optional[CloudflareVectorize] = None,
        binding: Any = None,
        d1_binding: Any = None,
        dimensions: int = 768,
        base_url: str = "https://api.cloudflare.com/client/v4",
        timeout: float = 60.0,
    ) -> None:
        """Initialize CloudflareVectorizeBaseStore.

        Args:
            account_id: Cloudflare account ID. Required for REST API usage;
                not needed when `binding` is provided.
            index_name: Name of the Vectorize index. Optional when `binding`
                is provided (the index is already configured in wrangler.jsonc).
            api_token: Cloudflare API token. Required for REST API usage; not
                needed when `binding` is provided.
            embedding_function: Embeddings instance to generate embeddings.
            cf_vectorize: An already-configured CloudflareVectorize instance
                to use directly -- e.g. one `with_cloudflare_embeddings()`
                already called `create_index()` on. If not provided, one is
                constructed from account_id/index_name/api_token/binding/d1_binding.
            binding: Optional Vectorize binding (e.g. `self.env.VECTORIZE`)
                for use in Python Workers. When provided, the async methods
                (`aget`/`aput`/`adelete`/`asearch`/`abatch`) use the binding
                instead of REST API calls, and the sync methods bridge to
                them via `pyodide.ffi.run_sync()`.
            d1_binding: Optional D1 database binding (e.g. `self.env.D1`) for
                use in Python Workers.
            dimensions: Number of dimensions for vectors (default: 768)
            base_url: Base URL for Cloudflare API
            timeout: Request timeout in seconds (default: 60.0)

        Raises:
            ValueError: If embedding_function is not provided.
        """
        if embedding_function is None:
            raise ValueError("embedding_function is required")

        self.account_id = account_id
        self.index_name = index_name
        self.api_token = api_token
        self.embedding_function = embedding_function
        self.dimensions = dimensions
        self.base_url = base_url
        self.timeout = timeout
        self.binding = binding
        self.d1_binding = d1_binding

        # Use an already-configured instance if given (e.g. from
        # with_cloudflare_embeddings(), which may have already called
        # create_index() on it) rather than silently building a second,
        # separate one from the raw params below.
        self.vectorstore = cf_vectorize or CloudflareVectorize(
            account_id=account_id,
            index_name=index_name,
            api_token=api_token,
            embedding=embedding_function,
            binding=binding,
            d1_binding=d1_binding,
        )
        self.cf_vectorize = self.vectorstore

    @staticmethod
    def check_index_exists(
        index_list: List[Dict[str, Any]], index_name: str
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if an index with the given name exists in the list of indexes.

        Args:
            index_list: List of dictionaries containing index information
            index_name: Name of the index to search for

        Returns:
            Tuple of (exists: bool, index_info: dict or None)
        """
        for index in index_list:
            if index.get("name") == index_name:
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

        REST API only -- index creation is a management operation, not
        something to do from inside a live Worker request. For Worker usage,
        construct `CloudflareVectorizeBaseStore` directly with `binding`
        (and a binding-aware `embedding_function`) instead.

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
            from langchain_cloudflare import CloudflareVectorize
            from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
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
            model_name=embedding_model,
        )

        # Initialize CloudflareVectorize
        # index_name must be set here, not just passed to create_index()/cls()
        # below -- CloudflareVectorize's own instance methods (aget_by_ids,
        # similarity_search, etc.) fall back to self.index_name when no
        # index_name is given per call, and cls() below reuses this same
        # cf_vectorize instance rather than constructing a fresh one.
        cf_vectorize = CloudflareVectorize(
            embedding=embedder,
            account_id=account_id,
            vectorize_api_token=vectorize_api_token,  # Optional if using global token
            index_name=index_name,
        )

        indexes = cf_vectorize.list_indexes()
        index_exists = False
        for index in indexes:
            if index.get("name") == index_name:
                index_exists = True

        if not index_exists:
            cf_vectorize.create_index(
                index_name=index_name,
                description=description,
                dimensions=768,
                wait=True,
            )

        return cls(
            account_id=account_id,
            index_name=index_name,
            api_token=vectorize_api_token,
            embedding_function=embedder,
            cf_vectorize=cf_vectorize,
            dimensions=dimensions,
            base_url=base_url,
            timeout=timeout,
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
        index: Union[list[str], bool, None] = None,
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
            "vector_id": self._generate_vector_id(namespace, key),
        }

        return text, metadata

    def _extract_item_from_document(
        self, doc: Any
    ) -> tuple[tuple[str, ...], str, dict[str, Any], datetime, datetime]:
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
            created_at = (
                datetime.fromisoformat(created_at_str)
                if created_at_str
                else datetime.now()
            )
        except ValueError:
            created_at = datetime.now()

        try:
            updated_at = (
                datetime.fromisoformat(updated_at_str) if updated_at_str else created_at
            )
        except ValueError:
            updated_at = created_at

        return namespace_tuple, key, value, created_at, updated_at

    # MARK: - Sync Bridge

    def _run_sync(self, make_coro: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Bridge an async method to a synchronous call via pyodide.ffi.run_sync().

        Only used when `self.binding` is set -- REST-mode sync calls never
        reach this. Only importable, and only works, from inside a live
        Cloudflare Python Worker request, not a plain CPython process. Takes
        a zero-arg thunk rather than an already-built coroutine so we never
        construct (and leave dangling/unawaited) a coroutine object when
        run_sync() turns out not to be available.
        """
        try:
            from pyodide.ffi import run_sync  # type: ignore[import-not-found]
        except ImportError as e:
            raise NotImplementedError(
                "Synchronous CloudflareVectorizeBaseStore methods require "
                "Pyodide's run_sync() when using a Vectorize binding, which "
                "is only available inside a Cloudflare Python Worker. Use "
                "the async methods (aget, aput, adelete, asearch, abatch) "
                "instead."
            ) from e
        return run_sync(make_coro())  # type: ignore[no-any-return]

    # LangGraph BaseStore interface methods
    def get(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        """Get a single item by namespace and key.

        Args:
            namespace: Namespace tuple
            key: Item key
            refresh_ttl: Whether to refresh TTL (not used in this implementation)

        Returns:
            Item if found, None otherwise
        """
        if self.binding is not None:
            return self._run_sync(
                lambda: self.aget(namespace, key, refresh_ttl=refresh_ttl)
            )
        try:
            vector_id = self._generate_vector_id(namespace, key)

            # Search by vector_id in metadata
            docs = self.vectorstore.get_by_ids([vector_id])

            if docs:
                doc = docs[0]
                namespace_tuple, key_extracted, value, created_at, updated_at = (
                    self._extract_item_from_document(doc)
                )

                return Item(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=created_at,
                    updated_at=updated_at,
                )

            return None

        except Exception as e:
            raise Exception("Error in retrieval:", str(e))

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | NotProvided | None = NOT_PROVIDED,
    ) -> None:
        """Store or update a single item.

        Args:
            namespace: Namespace tuple
            key: Item key
            value: Item value
            index: Indexing configuration
            ttl: Time to live (not implemented in this version)

        Raises:
            RuntimeError: If storage operation fails
        """
        if self.binding is not None:
            self._run_sync(lambda: self.aput(namespace, key, value, index, ttl=ttl))
            return
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
        if self.binding is not None:
            self._run_sync(lambda: self.adelete(namespace, key))
            return
        try:
            vector_id = self._generate_vector_id(namespace, key)
            self.vectorstore.delete([vector_id])
        except Exception as e:
            raise Exception("Error in deletion:", str(e))

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = DEFAULT_TOP_K,
        offset: int = 0,
        refresh_ttl: bool | None = None,
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
        if self.binding is not None:
            return self._run_sync(
                lambda: self.asearch(
                    namespace_prefix,
                    query=query,
                    filter=filter,
                    limit=limit,
                    offset=offset,
                    refresh_ttl=refresh_ttl,
                )
            )
        try:
            namespace_prefix_str = "/".join(namespace_prefix)

            # Build filter for namespace prefix
            search_filter = filter.copy() if filter else {}

            if query:
                # Semantic search with query
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=limit + offset,  # Get more to handle offset
                    md_filter=search_filter,
                )
            else:
                # No query: fetch candidates with a dummy query and rely on the
                # namespace-prefix filtering below. An empty namespace_prefix_str
                # is a valid "search everything" request (every namespace starts
                # with ""), so this must run regardless of whether a prefix was
                # given -- not just when namespace_prefix_str is truthy.
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query=" ",  # Dummy query to get all results
                    k=limit + offset,
                    md_filter=search_filter,
                )
            docs_and_scores = [(doc, score) for doc, score in docs_with_scores]

            return self._to_search_items(
                docs_and_scores, namespace_prefix_str, limit, offset
            )

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Search failed: {str(e)}")

    def _to_search_items(
        self,
        docs_and_scores: list[tuple[Any, float]],
        namespace_prefix_str: str,
        limit: int,
        offset: int,
    ) -> list[SearchItem]:
        """Filter (doc, score) pairs by namespace prefix and apply pagination.

        Shared by search() and asearch() so the filtering logic only lives
        in one place.
        """
        results = []
        processed_count = 0

        for doc, score in docs_and_scores:
            namespace_tuple, key, value, created_at, updated_at = (
                self._extract_item_from_document(doc)
            )

            # Check if namespace matches prefix
            namespace_str = "/".join(namespace_tuple)
            if namespace_str.startswith(namespace_prefix_str):
                if processed_count >= offset:
                    results.append(
                        SearchItem(
                            namespace=namespace_tuple,
                            key=key,
                            value=value,
                            created_at=created_at,
                            updated_at=updated_at,
                            score=score,
                        )
                    )

                    if len(results) >= limit:
                        break

                processed_count += 1
        return results

    def batch(
        self, operations: Iterable[GetOp | SearchOp | PutOp | ListNamespacesOp]
    ) -> list[Item | None | list[Item] | list[SearchItem] | list[tuple[str, ...]]]:
        """Execute multiple operations in a batch.

        Args:
            operations: Iterable of operation objects

        Returns:
            List of results corresponding to each operation
        """
        if self.binding is not None:
            return self._run_sync(lambda: self.abatch(operations))

        results: list[
            Item | None | list[Item] | list[SearchItem] | list[tuple[str, ...]]
        ] = []

        for op in operations:
            try:
                # Handle GetOp
                if isinstance(op, GetOp):
                    result = self.get(op.namespace, op.key)
                    results.append(result)
                # Handle PutOp
                elif isinstance(op, PutOp):
                    # Fix: Handle None value properly
                    if op.value is not None:
                        self.put(
                            op.namespace,
                            op.key,
                            op.value,
                            index=op.index,
                            ttl=getattr(op, "ttl", NOT_PROVIDED),
                        )
                    else:
                        # If value is None, delete the item
                        self.delete(op.namespace, op.key)
                    results.append(None)
                # Handle SearchOp
                elif isinstance(op, SearchOp):
                    search_result = self.search(
                        op.namespace_prefix,
                        query=getattr(op, "query", None),
                        filter=getattr(op, "filter", None),
                        limit=getattr(op, "limit", DEFAULT_TOP_K),
                        offset=getattr(op, "offset", 0),
                    )
                    results.append(search_result)
                # Handle ListNamespacesOp
                elif isinstance(op, ListNamespacesOp):
                    # This is a placeholder implementation
                    # You would need to implement namespace listing based on your storage
                    empty_namespaces: list[tuple[str, ...]] = []
                    results.append(empty_namespaces)
                else:
                    results.append(None)
            except Exception:
                results.append(None)

        return results

    # MARK: - Async Methods

    async def aget(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        """Get a single item by namespace and key, asynchronously.

        Uses the Vectorize binding when `self.binding` is set, otherwise the
        REST API -- either way, this is a real async call, not a wrapper
        around the sync method.

        Args:
            namespace: Namespace tuple
            key: Item key
            refresh_ttl: Whether to refresh TTL (not used in this implementation)

        Returns:
            Item if found, None otherwise
        """
        try:
            vector_id = self._generate_vector_id(namespace, key)

            docs = await self.vectorstore.aget_by_ids([vector_id])

            if docs:
                doc = docs[0]
                namespace_tuple, key_extracted, value, created_at, updated_at = (
                    self._extract_item_from_document(doc)
                )

                return Item(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=created_at,
                    updated_at=updated_at,
                )

            return None

        except Exception as e:
            raise Exception("Error in retrieval:", str(e))

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | NotProvided | None = NOT_PROVIDED,
    ) -> None:
        """Store or update a single item, asynchronously.

        Args:
            namespace: Namespace tuple
            key: Item key
            value: Item value
            index: Indexing configuration
            ttl: Time to live (not implemented in this version)

        Raises:
            RuntimeError: If storage operation fails
        """
        try:
            existing_item = await self.aget(namespace, key)

            text, metadata = self._create_document(namespace, key, value, index)

            if existing_item:
                metadata["created_at"] = existing_item.created_at.isoformat()
                await self.adelete(namespace, key)

            vector_id = self._generate_vector_id(namespace, key)

            from langchain_core.documents import Document

            doc = Document(page_content=text, metadata=metadata)

            await self.vectorstore.aadd_documents([doc], ids=[vector_id])

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to store item: {str(e)}")

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete a single item, asynchronously.

        Args:
            namespace: Namespace tuple
            key: Item key
        """
        try:
            vector_id = self._generate_vector_id(namespace, key)
            await self.vectorstore.adelete([vector_id])
        except Exception as e:
            raise Exception("Error in deletion:", str(e))

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = DEFAULT_TOP_K,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Search for items within a namespace prefix, asynchronously.

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
            search_filter = filter.copy() if filter else {}

            docs_with_scores = await self.vectorstore.asimilarity_search_with_score(
                query=query if query else " ",
                k=limit + offset,
                md_filter=search_filter,
                # asimilarity_search_with_score defaults to return_metadata="none"
                # (unlike the sync method, which defaults to "all"), but
                # _extract_item_from_document reads namespace/key/data out of
                # metadata -- so this must be explicit or every result comes
                # back with an empty namespace and key.
                return_metadata="all",
            )
            docs_and_scores = [(doc, score) for doc, score in docs_with_scores]

            return self._to_search_items(
                docs_and_scores, namespace_prefix_str, limit, offset
            )

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Search failed: {str(e)}")

    async def abatch(
        self, operations: Iterable[GetOp | SearchOp | PutOp | ListNamespacesOp]
    ) -> list[Item | None | list[Item] | list[SearchItem] | list[tuple[str, ...]]]:
        """Execute multiple operations in a batch, asynchronously.

        Args:
            operations: Iterable of operation objects

        Returns:
            List of results corresponding to each operation
        """
        results: list[
            Item | None | list[Item] | list[SearchItem] | list[tuple[str, ...]]
        ] = []

        for op in operations:
            try:
                if isinstance(op, GetOp):
                    result = await self.aget(op.namespace, op.key)
                    results.append(result)
                elif isinstance(op, PutOp):
                    if op.value is not None:
                        await self.aput(
                            op.namespace,
                            op.key,
                            op.value,
                            index=op.index,
                            ttl=getattr(op, "ttl", NOT_PROVIDED),
                        )
                    else:
                        await self.adelete(op.namespace, op.key)
                    results.append(None)
                elif isinstance(op, SearchOp):
                    search_result = await self.asearch(
                        op.namespace_prefix,
                        query=getattr(op, "query", None),
                        filter=getattr(op, "filter", None),
                        limit=getattr(op, "limit", DEFAULT_TOP_K),
                        offset=getattr(op, "offset", 0),
                    )
                    results.append(search_result)
                elif isinstance(op, ListNamespacesOp):
                    empty_namespaces: list[tuple[str, ...]] = []
                    results.append(empty_namespaces)
                else:
                    results.append(None)
            except Exception:
                results.append(None)

        return results
