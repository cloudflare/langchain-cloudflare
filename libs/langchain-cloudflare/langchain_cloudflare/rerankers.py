"""Cloudflare Workers AI Reranker.

This module provides a reranker class for Cloudflare Workers AI that can be used
to rerank documents based on their relevance to a query.

Since LangChain does not have a base reranker class, this is a standalone
implementation following the same patterns as the embeddings module.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
from langchain_core.documents import Document
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

DEFAULT_RERANKER_MODEL = "@cf/baai/bge-reranker-base"


@dataclass
class RerankResult:
    """Result from reranking a document.

    Attributes:
        index: The original index of the document in the input list.
        score: The relevance score (higher is more relevant).
        text: The text content (optional, included if return_documents=True).
        document: The original Document object (optional).
    """

    index: int
    score: float
    text: Optional[str] = None
    document: Optional[Document] = None


class CloudflareWorkersAIReranker(BaseModel):
    """Cloudflare Workers AI reranker model.

    To use, you need to provide an API token and account ID to access
    Cloudflare Workers AI, or pass a Workers AI binding when running
    in a Python Worker.

    Example (REST API):
        .. code-block:: python

            from langchain_cloudflare import CloudflareWorkersAIReranker

            reranker = CloudflareWorkersAIReranker(
                account_id="my_account_id",
                api_token="my_secret_api_token",
            )

            results = reranker.rerank(
                query="What is the capital of France?",
                documents=["Paris is the capital of France.", "Berlin is in Germany."],
                top_k=2,
            )

    Example (Worker binding):
        .. code-block:: python

            from langchain_cloudflare import CloudflareWorkersAIReranker

            reranker = CloudflareWorkersAIReranker(
                binding=self.env.AI,
            )

            results = await reranker.arerank(
                query="What is the capital of France?",
                documents=["Paris is the capital of France.", "Berlin is in Germany."],
            )

    Key init args:
        account_id: str
            Cloudflare account ID. If not specified, will be read from
            the CF_ACCOUNT_ID environment variable.

        api_token: str
            Cloudflare Workers AI API token. If not specified, will be read from
            the CF_AI_API_TOKEN environment variable.

        model_name: str
            Reranker model name on Workers AI (default: "@cf/baai/bge-reranker-base")

        ai_gateway: str
            Optional AI Gateway name for routing requests through Cloudflare AI Gateway.

        binding: Any
            Workers AI binding (env.AI) for use in Python Workers.
    """

    api_base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    account_id: str = Field(default_factory=from_env("CF_ACCOUNT_ID", default=""))
    api_token: SecretStr = Field(
        default_factory=secret_from_env("CF_AI_API_TOKEN", default="")
    )
    model_name: str = DEFAULT_RERANKER_MODEL
    headers: Dict[str, str] = {"Authorization": "Bearer "}
    ai_gateway: Optional[str] = Field(
        default_factory=from_env("AI_GATEWAY", default=None)
    )
    binding: Any = Field(default=None, exclude=True)
    """Workers AI binding (env.AI) for use in Python Workers."""

    _inference_url: str = PrivateAttr()

    def __init__(self, **kwargs: Any):
        """Initialize the Cloudflare Workers AI reranker."""
        super().__init__(**kwargs)

        # If binding is provided, skip REST API setup
        if self.binding is not None:
            self._inference_url = ""
            return

        # Validate credentials
        if not self.account_id:
            raise ValueError(
                "A Cloudflare account ID must be provided either through "
                "the account_id parameter or CF_ACCOUNT_ID environment variable. "
                "Or pass the 'binding' parameter (env.AI) in a Python Worker."
            )

        if not self.api_token or self.api_token.get_secret_value() == "":
            raise ValueError(
                "A Cloudflare API token must be provided either through "
                "the api_token parameter or CF_AI_API_TOKEN environment variable. "
                "Or pass the 'binding' parameter (env.AI) in a Python Worker."
            )

        self.headers = {"Authorization": f"Bearer {self.api_token.get_secret_value()}"}

        if self.ai_gateway:
            self._inference_url = (
                f"https://gateway.ai.cloudflare.com/v1/"
                f"{self.account_id}/{self.ai_gateway}/workers-ai/run/{self.model_name}"
            )
        else:
            self._inference_url = (
                f"{self.api_base_url}/{self.account_id}/ai/run/{self.model_name}"
            )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def _prepare_documents(
        self, documents: Sequence[Union[str, Document]]
    ) -> tuple[List[Dict[str, str]], List[Optional[Document]]]:
        """Prepare documents for the reranker API.

        Args:
            documents: List of strings or Document objects.

        Returns:
            Tuple of (contexts list for API, original documents list).
        """
        contexts = []
        original_docs: List[Optional[Document]] = []

        for doc in documents:
            if isinstance(doc, Document):
                contexts.append({"text": doc.page_content})
                original_docs.append(doc)
            else:
                contexts.append({"text": doc})
                original_docs.append(None)

        return contexts, original_docs

    def _process_response(
        self,
        response_data: List[Dict[str, Any]],
        documents: Sequence[Union[str, Document]],
        original_docs: List[Optional[Document]],
        return_documents: bool,
    ) -> List[RerankResult]:
        """Process the reranker API response.

        Args:
            response_data: The response from the API (list of {id, score}).
            documents: Original input documents.
            original_docs: List of original Document objects (or None for strings).
            return_documents: Whether to include document text in results.

        Returns:
            List of RerankResult objects sorted by score (descending).
        """
        results = []
        for item in response_data:
            idx = item["id"]
            score = item["score"]

            text = None
            doc = None

            if return_documents and 0 <= idx < len(documents):
                original = documents[idx]
                if isinstance(original, Document):
                    text = original.page_content
                    doc = original
                else:
                    text = original

            results.append(
                RerankResult(
                    index=idx,
                    score=score,
                    text=text,
                    document=doc if return_documents else None,
                )
            )

        return results

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, Document]],
        *,
        top_k: Optional[int] = None,
        return_documents: bool = True,
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The query to rank documents against.
            documents: List of documents to rerank. Can be strings or Document objects.
            top_k: Maximum number of results to return. If None, returns all documents.
            return_documents: Whether to include document text in results.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        if not documents:
            return []

        contexts, original_docs = self._prepare_documents(documents)

        payload: Dict[str, Any] = {
            "query": query,
            "contexts": contexts,
        }
        if top_k is not None:
            payload["top_k"] = top_k

        response = requests.post(
            url=self._inference_url,
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()

        response_json = response.json()
        # Handle Cloudflare REST API response format:
        # {"result": {"response": [...], "usage": {...}}, "success": true, ...}
        if "result" in response_json:
            result = response_json["result"]
            if isinstance(result, dict) and "response" in result:
                response_data = result["response"]
            else:
                response_data = result
        elif "response" in response_json:
            response_data = response_json["response"]
        else:
            response_data = response_json

        return self._process_response(
            response_data, documents, original_docs, return_documents
        )

    async def arerank(
        self,
        query: str,
        documents: Sequence[Union[str, Document]],
        *,
        top_k: Optional[int] = None,
        return_documents: bool = True,
    ) -> List[RerankResult]:
        """Asynchronously rerank documents based on relevance to the query.

        Args:
            query: The query to rank documents against.
            documents: List of documents to rerank. Can be strings or Document objects.
            top_k: Maximum number of results to return. If None, returns all documents.
            return_documents: Whether to include document text in results.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        if not documents:
            return []

        contexts, original_docs = self._prepare_documents(documents)

        payload: Dict[str, Any] = {
            "query": query,
            "contexts": contexts,
        }
        if top_k is not None:
            payload["top_k"] = top_k

        # Use binding if available (for Python Workers)
        if self.binding is not None:
            return await self._arerank_with_binding(
                payload, documents, original_docs, return_documents
            )

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self._inference_url,
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()

        response_json = response.json()
        # Handle Cloudflare REST API response format:
        # {"result": {"response": [...], "usage": {...}}, "success": true, ...}
        if "result" in response_json:
            result = response_json["result"]
            if isinstance(result, dict) and "response" in result:
                response_data = result["response"]
            else:
                response_data = result
        elif "response" in response_json:
            response_data = response_json["response"]
        else:
            response_data = response_json

        return self._process_response(
            response_data, documents, original_docs, return_documents
        )

    async def _arerank_with_binding(
        self,
        payload: Dict[str, Any],
        documents: Sequence[Union[str, Document]],
        original_docs: List[Optional[Document]],
        return_documents: bool,
    ) -> List[RerankResult]:
        """Rerank documents using the Workers AI binding.

        Args:
            payload: The request payload (query, contexts, top_k).
            documents: Original input documents.
            original_docs: List of original Document objects.
            return_documents: Whether to include document text in results.

        Returns:
            List of RerankResult objects sorted by relevance score.
        """
        from .bindings import (
            convert_payload_for_binding,
            convert_reranker_response,
            create_gateway_options,
        )

        # Convert payload to JS-compatible format for Pyodide
        js_payload = convert_payload_for_binding(payload)

        # Create AI Gateway options if configured
        gateway_options = create_gateway_options(self.ai_gateway)

        # Call the binding with optional gateway
        if gateway_options is not None:
            response = await self.binding.run(
                self.model_name, js_payload, gateway_options
            )
        else:
            response = await self.binding.run(self.model_name, js_payload)

        # Convert response to Python format
        response_data = convert_reranker_response(response)

        return self._process_response(
            response_data, documents, original_docs, return_documents
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Compress documents by reranking and returning top results.

        This method provides compatibility with LangChain's document compressor
        interface, making it easy to use in retrieval pipelines.

        Args:
            documents: List of Document objects to rerank.
            query: The query to rank documents against.
            top_k: Maximum number of documents to return.

        Returns:
            List of Document objects sorted by relevance.
        """
        results = self.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=True,
        )

        return [r.document for r in results if r.document is not None]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Asynchronously compress documents by reranking and returning top results.

        This method provides compatibility with LangChain's document compressor
        interface, making it easy to use in retrieval pipelines.

        Args:
            documents: List of Document objects to rerank.
            query: The query to rank documents against.
            top_k: Maximum number of documents to return.

        Returns:
            List of Document objects sorted by relevance.
        """
        results = await self.arerank(
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=True,
        )

        return [r.document for r in results if r.document is not None]
