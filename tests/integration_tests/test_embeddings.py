"""Test CloudflareWorkersAI embeddings."""

from typing import Type

from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[CloudflareWorkersAIEmbeddings]:
        return CloudflareWorkersAIEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
