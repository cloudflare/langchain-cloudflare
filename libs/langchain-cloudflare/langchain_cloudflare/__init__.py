from importlib import metadata

from langchain_cloudflare.bindings import (
    # Workers AI binding utilities
    convert_binding_response_to_rest_format,
    convert_payload_for_binding,
    # Vectorize binding utilities
    convert_query_options_for_binding,
    # Reranker binding utilities
    convert_reranker_response,
    convert_vectorize_describe_response,
    convert_vectorize_get_response,
    convert_vectorize_mutation_response,
    convert_vectorize_query_response,
    convert_vectors_for_binding,
)
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langchain_cloudflare.rerankers import CloudflareWorkersAIReranker, RerankResult
from langchain_cloudflare.vectorstores import CloudflareVectorize

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCloudflareWorkersAI",
    "CloudflareVectorize",
    "CloudflareWorkersAIEmbeddings",
    "CloudflareWorkersAIReranker",
    "RerankResult",
    # Workers AI binding utilities
    "convert_binding_response_to_rest_format",
    "convert_payload_for_binding",
    # Vectorize binding utilities
    "convert_query_options_for_binding",
    "convert_vectorize_describe_response",
    "convert_vectorize_get_response",
    "convert_vectorize_mutation_response",
    "convert_vectorize_query_response",
    "convert_vectors_for_binding",
    # Reranker binding utilities
    "convert_reranker_response",
    "__version__",
]
