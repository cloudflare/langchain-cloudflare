"""Cloudflare AI Search retriever.

This module provides a LangChain ``BaseRetriever`` backed by Cloudflare AI Search
(the managed retrieval / RAG service, fka AutoRAG). It queries an AI Search
instance's ``/search`` endpoint and returns the matching content chunks as
LangChain ``Document`` objects.

AI Search is a *managed* service: ingestion and indexing happen out-of-band (from
an R2 bucket, a website, or uploaded files), so this integration is a retriever
(like ``AmazonKnowledgeBasesRetriever`` / ``VertexAISearchRetriever``), not a
``VectorStore``.

Example (REST API):
    .. code-block:: python

        from langchain_cloudflare import CloudflareAISearchRetriever

        retriever = CloudflareAISearchRetriever(
            account_id="my_account_id",
            api_token="my_secret_api_token",
            instance_name="my-instance",
            k=5,
        )

        docs = retriever.invoke("How do I configure Workers AI?")

Example (Worker binding):
    .. code-block:: python

        from langchain_cloudflare import CloudflareAISearchRetriever

        # `env.MY_SEARCH` is a dedicated `ai_search` binding (NOT env.AI).
        # For a namespace binding, pass `env.<NAMESPACE>.get("my-instance")`.
        retriever = CloudflareAISearchRetriever(binding=env.MY_SEARCH)

        docs = await retriever.ainvoke("How do I configure Workers AI?")

For multi-tenant or per-agent setups, give each tenant (or agent) its own instance
(which holds just their files) and point a retriever at it.
"""

# MARK: - Imports
import os
from typing import Any, Dict, List, Literal, Optional

import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from ._errors import TokenErrors

# MARK: - Constants
DEFAULT_K = 10
MAX_NUM_RESULTS = 50

# Enum value sets for the typed ai_search_options fields.
RetrievalType = Literal["vector", "keyword", "hybrid"]
FusionMethod = Literal["max", "rrf"]
KeywordMatchMode = Literal["and", "or"]
CacheThreshold = Literal[
    "super_strict_match", "close_enough", "flexible_friend", "anything_goes"
]


# MARK: - Helpers
def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``data`` with all ``None``-valued keys removed."""
    return {key: value for key, value in data.items() if value is not None}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on conflicts)."""
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


# MARK: - CloudflareAISearchRetriever
class CloudflareAISearchRetriever(BaseRetriever):
    """Cloudflare AI Search retriever.

    Queries a Cloudflare AI Search instance and returns matching chunks as
    ``Document`` objects. Provide credentials + an ``instance_name`` for the REST
    API, or pass a Worker ``binding`` when running inside a Python Worker.

    Key init args:
        account_id: str
            Cloudflare account ID. Defaults to the ``CF_ACCOUNT_ID`` env var.

        api_token: str
            AI Search API token with the ``AI Search:Run`` permission (this
            retriever only issues read-only ``/search`` requests). Defaults to the
            ``CF_AI_SEARCH_API_TOKEN`` env var, falling back to ``CF_API_TOKEN``.
            This is the query token; it is distinct from the service API token AI
            Search uses internally to read your R2 bucket during indexing.
            ``AI Search:Edit`` is only needed to provision an instance (create,
            upload items, sync), which is done outside this retriever.

        instance_name: str
            AI Search instance name. Defaults to the ``CF_AI_SEARCH_INSTANCE_NAME``
            env var. Required for the REST API.

        namespace: str
            AI Search namespace. Defaults to ``"default"`` (overridable via the
            ``CF_AI_SEARCH_NAMESPACE`` env var); set it to target an instance in a
            non-default namespace, and the namespace-scoped endpoint is used.

        k: int
            Maximum number of results (mapped to ``retrieval.max_num_results``,
            clamped to 50). Also accepted as an ``.invoke(query, k=...)`` kwarg.

        retrieval_type: Optional[str]
            One of ``"vector"``, ``"keyword"``, or ``"hybrid"``.

        filters: Optional[dict]
            AI Search metadata filter applied to results.

        binding: Any
            AI Search binding (a dedicated ``ai_search`` binding, NOT ``env.AI``)
            for use in Python Workers. The binding path is async-only.
    """

    # MARK: - Connection / identity
    api_base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    account_id: str = Field(default_factory=from_env("CF_ACCOUNT_ID", default=""))
    api_token: SecretStr = Field(
        default_factory=secret_from_env("CF_AI_SEARCH_API_TOKEN", default="")
    )
    instance_name: str = Field(
        default_factory=from_env("CF_AI_SEARCH_INSTANCE_NAME", default="")
    )
    namespace: str = Field(
        default_factory=from_env("CF_AI_SEARCH_NAMESPACE", default="default")
    )
    headers: Dict[str, str] = {"Authorization": "Bearer "}
    binding: Any = Field(default=None, exclude=True)
    """AI Search binding (a dedicated ``ai_search`` binding, NOT ``env.AI``).

    The binding path is async-only. For a namespace binding, pass
    ``env.<NAMESPACE>.get("<instance>")``."""

    # MARK: - Retrieval options
    k: int = DEFAULT_K
    retrieval_type: Optional[RetrievalType] = None
    match_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    boost_by: Optional[List[Dict[str, str]]] = None
    fusion_method: Optional[FusionMethod] = None
    keyword_match_mode: Optional[KeywordMatchMode] = None
    context_expansion: Optional[int] = None
    return_on_failure: Optional[bool] = None
    rewrite_query: Optional[bool] = None
    rewrite_model: Optional[str] = None
    rewrite_prompt: Optional[str] = None
    reranking: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_match_threshold: Optional[float] = None
    cache: Optional[bool] = None
    cache_threshold: Optional[CacheThreshold] = None
    ai_search_options: Optional[Dict[str, Any]] = None
    """Raw ``ai_search_options`` for passing any AI Search option that has no
    dedicated field above; deep-merged over (and takes precedence over) the typed
    fields."""

    _search_url: str = PrivateAttr()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # MARK: - Init
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Cloudflare AI Search retriever."""
        super().__init__(**kwargs)

        # Binding path: skip REST setup (binding calls are async-only).
        if self.binding is not None:
            self._search_url = ""
            return

        # Validate credentials.
        if not self.account_id:
            raise ValueError(TokenErrors.NO_ACCOUNT_ID_SET)

        token = self.api_token.get_secret_value() if self.api_token else ""
        if not token:
            # Fall back to the shared global token.
            token = os.environ.get("CF_API_TOKEN", "")
        if not token:
            raise ValueError(TokenErrors.INSUFFICIENT_AI_SEARCH_TOKENS)

        if not self.instance_name:
            raise ValueError(TokenErrors.NO_AI_SEARCH_INSTANCE)

        self.headers = {"Authorization": f"Bearer {token}"}

        # The default namespace uses the canonical bare-instance endpoint; a
        # non-default namespace uses the namespace-scoped endpoint.
        if self.namespace and self.namespace != "default":
            self._search_url = (
                f"{self.api_base_url}/{self.account_id}/ai-search/namespaces/"
                f"{self.namespace}/instances/{self.instance_name}/search"
            )
        else:
            self._search_url = (
                f"{self.api_base_url}/{self.account_id}/ai-search/instances/"
                f"{self.instance_name}/search"
            )

    # MARK: - Payload Builders
    def _effective_k(self, override: Optional[int]) -> int:
        """Resolve and clamp the effective result count."""
        k = self.k if override is None else override
        return max(1, min(k, MAX_NUM_RESULTS))

    def _build_ai_search_options(self, k: int) -> Dict[str, Any]:
        """Build the ``ai_search_options`` object, omitting unset fields."""
        retrieval = _drop_none(
            {
                "max_num_results": k,
                "match_threshold": self.match_threshold,
                "filters": self.filters,
                "retrieval_type": self.retrieval_type,
                "fusion_method": self.fusion_method,
                "keyword_match_mode": self.keyword_match_mode,
                "boost_by": self.boost_by,
                "context_expansion": self.context_expansion,
                "return_on_failure": self.return_on_failure,
            }
        )
        query_rewrite = _drop_none(
            {
                "enabled": self.rewrite_query,
                "model": self.rewrite_model,
                "rewrite_prompt": self.rewrite_prompt,
            }
        )
        reranking = _drop_none(
            {
                "enabled": self.reranking,
                "match_threshold": self.rerank_match_threshold,
                "model": self.rerank_model,
            }
        )
        cache = _drop_none(
            {
                "enabled": self.cache,
                "cache_threshold": self.cache_threshold,
            }
        )
        sections = {
            "retrieval": retrieval,
            "query_rewrite": query_rewrite,
            "reranking": reranking,
            "cache": cache,
        }
        options = {name: body for name, body in sections.items() if body}
        return _deep_merge(options, self.ai_search_options or {})

    def _build_body(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Build the search request body (uses ``query``, never ``messages``)."""
        body: Dict[str, Any] = {"query": query}
        options = self._build_ai_search_options(self._effective_k(k))
        if options:
            body["ai_search_options"] = options
        return body

    # MARK: - Response Mapping
    def _chunk_to_document(self, chunk: Dict[str, Any]) -> Document:
        """Map a single AI Search chunk to a ``Document`` with citation metadata."""
        item = chunk.get("item") or {}
        # Spread user metadata first so canonical keys win on collision.
        metadata: Dict[str, Any] = dict(item.get("metadata") or {})
        metadata.update(
            {
                "id": chunk.get("id"),
                "score": chunk.get("score"),
                "type": chunk.get("type"),
                "filename": item.get("key"),
                "timestamp": item.get("timestamp"),
                "scoring_details": chunk.get("scoring_details"),
                "instance_id": self.instance_name or None,
            }
        )
        return Document(page_content=chunk.get("text") or "", metadata=metadata)

    @staticmethod
    def _extract_chunks(data: Any) -> List[Dict[str, Any]]:
        """Extract the ``chunks`` list from a REST or binding response."""
        if not isinstance(data, dict):
            return []
        result = data.get("result")
        if isinstance(result, dict) and "chunks" in result:
            return result.get("chunks") or []
        if "chunks" in data:
            return data.get("chunks") or []
        return []

    # MARK: - Retrieval
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve documents for ``query`` via the AI Search REST API."""
        if self.binding is not None:
            raise NotImplementedError(
                "The AI Search Worker binding is async-only. "
                "Use `ainvoke` / `aget_relevant_documents` instead."
            )
        body = self._build_body(query, kwargs.get("k"))
        response = requests.post(
            url=self._search_url,
            headers=self.headers,
            json=body,
        )
        response.raise_for_status()
        data = response.json()
        return [self._chunk_to_document(c) for c in self._extract_chunks(data)]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously retrieve documents for ``query``."""
        body = self._build_body(query, kwargs.get("k"))

        # Use the binding if available (for Python Workers).
        if self.binding is not None:
            return await self._asearch_with_binding(body)

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self._search_url,
                headers=self.headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        return [self._chunk_to_document(c) for c in self._extract_chunks(data)]

    async def _asearch_with_binding(self, body: Dict[str, Any]) -> List[Document]:
        """Retrieve documents using the AI Search Worker binding."""
        from .bindings import (
            convert_aisearch_response,
            convert_payload_for_binding,
        )

        js_body = convert_payload_for_binding(body)
        response = await self.binding.search(js_body)
        data = convert_aisearch_response(response)
        return [self._chunk_to_document(c) for c in self._extract_chunks(data)]
