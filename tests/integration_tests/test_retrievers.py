from typing import Type

from langchain_cloudflare.retrievers import CloudflareWorkersAIRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestCloudflareWorkersAIRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[CloudflareWorkersAIRetriever]:
        """Get an empty vectorstore for unit tests."""
        return CloudflareWorkersAIRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "example query"
