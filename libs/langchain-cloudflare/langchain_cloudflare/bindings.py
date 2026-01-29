"""Cloudflare Workers binding utilities for langchain-cloudflare.

This module provides utilities to use Cloudflare Workers bindings (env.AI,
env.VECTORIZE) with langchain-cloudflare in Python Workers (Pyodide environment).

Example usage in a Python Worker:

    from workers import WorkerEntrypoint, Response
    from langchain_cloudflare import ChatCloudflareWorkersAI, CloudflareVectorize

    class Default(WorkerEntrypoint):
        async def fetch(self, request, env):
            # Using Workers AI binding
            llm = ChatCloudflareWorkersAI(
                model_name="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                binding=self.env.AI,
            )

            response = await llm.ainvoke("Hello!")

            # Using Vectorize binding
            vectorstore = CloudflareVectorize(
                embedding=embeddings,
                binding=self.env.VECTORIZE,  # Pass the Vectorize binding
            )

            results = await vectorstore.asimilarity_search("query")
            return Response.json({"response": response.content})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# MARK: - AI Gateway Options


def create_gateway_options(gateway_id: Optional[str]) -> Any:
    """Create AI Gateway options for Workers AI binding.

    When using Workers AI bindings with AI Gateway, you pass a third parameter:
    await env.AI.run(model, payload, { gateway: { id: "my-gateway" } })

    Args:
        gateway_id: The AI Gateway ID (name) to route requests through

    Returns:
        JS-compatible options object in Pyodide, or Python dict otherwise.
        Returns None if gateway_id is not provided.
    """
    if not gateway_id:
        return None

    options = {"gateway": {"id": gateway_id}}

    try:
        import json

        from js import JSON  # type: ignore[import-not-found]

        json_str = json.dumps(options)
        return JSON.parse(json_str)
    except ImportError:
        return options


# MARK: - Payload Conversion


def convert_payload_for_binding(payload: Dict[str, Any]) -> Any:
    """Convert a Python payload dict for Workers AI binding compatibility.

    In Pyodide, Python dicts with nested lists (like messages arrays) cause
    proxy iterator issues when passed to JS. This converts them properly.

    Args:
        payload: The Python dict payload to convert

    Returns:
        JS-compatible object in Pyodide, or original dict otherwise
    """
    try:
        import json

        from js import JSON  # type: ignore[import-not-found]

        # Use JSON round-trip to ensure deep conversion of all nested structures
        # This handles complex nested dicts/lists like tools arrays properly
        json_str = json.dumps(payload)
        return JSON.parse(json_str)
    except ImportError:
        return payload


# MARK: - Response Conversion


def convert_binding_response_to_rest_format(
    response: Any,
    model: str,
) -> Dict[str, Any]:
    """Convert a Workers AI binding response to REST API format.

    The binding returns responses differently than the REST API.
    This normalizes to the format ChatCloudflareWorkersAI expects.

    Args:
        response: The raw response from env.AI.run()
        model: The model name (unused, kept for future format detection)

    Returns:
        Dict in REST API response format with "result" wrapper
    """
    # Convert JS proxy to Python dict
    if hasattr(response, "to_py"):
        response = response.to_py()

    if isinstance(response, dict):
        if "result" in response:
            return response
        return {"result": response}

    return {"result": {"response": str(response)}}


# MARK: - Vectorize Binding Utilities


def convert_vectors_for_binding(vectors: Any) -> Any:
    """Convert vectors/IDs for Vectorize binding compatibility.

    In Pyodide, Python dicts/lists cause proxy iterator issues when passed
    to JS. This converts them properly for the Vectorize binding.

    Args:
        vectors: List of vector dicts, IDs, or embedding values

    Returns:
        JS-compatible array in Pyodide, or original list otherwise
    """
    try:
        import json

        from js import JSON  # type: ignore[import-not-found]

        # Use JSON round-trip to ensure deep conversion
        json_str = json.dumps(vectors)
        return JSON.parse(json_str)
    except ImportError:
        return vectors


def convert_query_options_for_binding(options: Dict[str, Any]) -> Any:
    """Convert query options dict for Vectorize binding compatibility.

    Args:
        options: Query options dict (topK, returnMetadata, returnValues, filter, etc.)

    Returns:
        JS-compatible object in Pyodide, or original dict otherwise
    """
    try:
        import json

        from js import JSON  # type: ignore[import-not-found]

        json_str = json.dumps(options)
        return JSON.parse(json_str)
    except ImportError:
        return options


def convert_vectorize_query_response(response: Any) -> Dict[str, Any]:
    """Convert a Vectorize binding query response to Python format.

    The binding returns JS objects that need to be converted to Python dicts.

    Args:
        response: The raw response from env.VECTORIZE.query()

    Returns:
        Dict with matches list containing id, score, and optional metadata/values
    """
    # Convert JS proxy to Python
    if hasattr(response, "to_py"):
        response = response.to_py()

    # Response should have a "matches" array
    if isinstance(response, dict):
        return response

    # Handle list directly (some bindings return matches array directly)
    if isinstance(response, list):
        return {"matches": response}

    return {"matches": []}


def convert_vectorize_mutation_response(response: Any) -> Dict[str, Any]:
    """Convert a Vectorize binding mutation response to Python format.

    Used for insert/upsert/delete operations.

    Args:
        response: The raw response from env.VECTORIZE.insert/upsert/deleteByIds()

    Returns:
        Dict with mutationId and count information
    """
    # Convert JS proxy to Python
    if hasattr(response, "to_py"):
        response = response.to_py()

    if isinstance(response, dict):
        return response

    return {"mutationId": None, "count": 0}


def convert_vectorize_get_response(response: Any) -> List[Dict[str, Any]]:
    """Convert a Vectorize binding getByIds response to Python format.

    Args:
        response: The raw response from env.VECTORIZE.getByIds()

    Returns:
        List of vector dicts with id, values, and metadata
    """
    # Convert JS proxy to Python
    if hasattr(response, "to_py"):
        response = response.to_py()

    if isinstance(response, list):
        return response

    # Some bindings wrap in a result
    if isinstance(response, dict) and "vectors" in response:
        return response["vectors"]

    return []


def convert_vectorize_describe_response(response: Any) -> Dict[str, Any]:
    """Convert a Vectorize binding describe response to Python format.

    Args:
        response: The raw response from env.VECTORIZE.describe()

    Returns:
        Dict with index configuration details
    """
    # Convert JS proxy to Python
    if hasattr(response, "to_py"):
        response = response.to_py()

    if isinstance(response, dict):
        return response

    return {}


# MARK: - Reranker Binding Utilities


def convert_reranker_response(response: Any) -> List[Dict[str, Any]]:
    """Convert a Reranker binding response to Python format.

    The binding returns a list of {id, score} objects that need to be
    converted from JS proxies to Python dicts.

    Args:
        response: The raw response from env.AI.run() for reranker model

    Returns:
        List of dicts with 'id' (int) and 'score' (float) keys
    """
    # Convert JS proxy to Python
    if hasattr(response, "to_py"):
        response = response.to_py()

    # Response should be a list of {id, score} objects
    if isinstance(response, list):
        return response

    # Handle wrapped response format
    if isinstance(response, dict):
        if "result" in response:
            result = response["result"]
            if hasattr(result, "to_py"):
                result = result.to_py()
            if isinstance(result, list):
                return result
        # Native AI binding returns {"response": [...], "usage": {...}}
        if "response" in response:
            resp = response["response"]
            if hasattr(resp, "to_py"):
                resp = resp.to_py()
            if isinstance(resp, list):
                return resp
        # Some responses might have a different structure
        if "data" in response:
            data = response["data"]
            if hasattr(data, "to_py"):
                data = data.to_py()
            if isinstance(data, list):
                return data

    return []


__all__ = [
    # Workers AI binding utilities
    "create_gateway_options",
    "convert_payload_for_binding",
    "convert_binding_response_to_rest_format",
    # Vectorize binding utilities
    "convert_vectors_for_binding",
    "convert_query_options_for_binding",
    "convert_vectorize_query_response",
    "convert_vectorize_mutation_response",
    "convert_vectorize_get_response",
    "convert_vectorize_describe_response",
    # Reranker binding utilities
    "convert_reranker_response",
]
