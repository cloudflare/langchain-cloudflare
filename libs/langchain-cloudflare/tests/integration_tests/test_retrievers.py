# ruff: noqa: T201
"""Integration tests for CloudflareAISearchRetriever.

Prerequisite: a real, pre-populated AI Search instance. The standard
``RetrieversIntegrationTests`` suite asserts exact result counts (3 and 1), so the
instance must return at least 3 chunks for ``retriever_query_example`` and the
retriever must not be configured with a pruning ``match_threshold`` or with
reranking/query rewriting enabled.

Required environment variables:
    - CF_ACCOUNT_ID
    - CF_AI_SEARCH_API_TOKEN (or CF_API_TOKEN)
    - CF_AI_SEARCH_INSTANCE_NAME
    - CF_AI_SEARCH_QUERY (optional; a broad query with many matches)
"""

import os
from typing import Type

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_cloudflare.retrievers import CloudflareAISearchRetriever

_HAS_CREDS = bool(
    os.environ.get("CF_ACCOUNT_ID")
    and os.environ.get("CF_AI_SEARCH_INSTANCE_NAME")
    and (os.environ.get("CF_AI_SEARCH_API_TOKEN") or os.environ.get("CF_API_TOKEN"))
)

pytestmark = pytest.mark.skipif(
    not _HAS_CREDS,
    reason="AI Search credentials / instance not configured",
)


class TestCloudflareAISearchRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[BaseRetriever]:
        return CloudflareAISearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        # account_id / api_token are read from the environment by the field
        # factories. Do not set k here (the harness injects it), and do not set
        # match_threshold / reranking / query rewriting (they break the exact
        # result-count assertions).
        return {"instance_name": os.environ["CF_AI_SEARCH_INSTANCE_NAME"]}

    @property
    def retriever_query_example(self) -> str:
        return os.environ.get("CF_AI_SEARCH_QUERY", "cloudflare")
