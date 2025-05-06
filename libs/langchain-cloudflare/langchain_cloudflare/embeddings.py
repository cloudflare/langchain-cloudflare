from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, PrivateAttr

DEFAULT_MODEL_NAME = "@cf/baai/bge-base-en-v1.5"


class CloudflareWorkersAIEmbeddings(BaseModel, Embeddings):
    """Cloudflare Workers AI embedding model.

    To use, you need to provide an API token and
    account ID to access Cloudflare Workers AI.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import CloudflareWorkersAIEmbeddings

            account_id = "my_account_id"
            api_token = "my_secret_api_token"
            model_name = "@cf/baai/bge-small-en-v1.5"

            cf = CloudflareWorkersAIEmbeddings(
                account_id=account_id,
                api_token=api_token,
                model_name=model_name
            )
    """

    """CloudflareWorkersAIEmbeddings embedding model integration.

    Key init args — completion params:
        account_id: str
            Cloudflare account ID
            
        api_token: str
            Cloudflare Workers AI API token
            
        model_name: str
            Embeddings model name on Workers AI (default: "@cf/baai/bge-base-en-v1.5")

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings

            account_id = "my_account_id"
            api_token = "my_secret_api_token"
            model_name = "@cf/baai/bge-small-en-v1.5"

            cf = CloudflareWorkersAIEmbeddings(
                account_id=account_id,
                api_token=api_token,
                model_name=model_name
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            cf.embed_query(input_text)

        .. code-block:: python

            [0.007663726806640625, 0.029022216796875, 0.006626129150390625,...]

    Embed multiple text:
        .. code-block:: python

            input_texts = ["Document 1...", "Document 2..."]
            cf.embed_documents(input_texts)

        .. code-block:: python

            [[-0.0015087127685546875, 0.03216552734375, -0.0025310516357421875,...]]

    Async:
        .. code-block:: python

            await cf.aembed_query(input_text)

            # multiple:
            # await cf.aembed_documents(input_texts)

        .. code-block:: python

            [0.007663726806640625, 0.029022216796875, 0.006626129150390625,...]
            [[-0.0015087127685546875, 0.03216552734375, -0.0025310516357421875,...]]

    """

    api_base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    account_id: str
    api_token: str
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 50
    strip_new_lines: bool = True
    headers: Dict[str, str] = {"Authorization": "Bearer "}
    ai_gateway: Optional[str] = None

    _inference_url: str = PrivateAttr()

    def __init__(self, **kwargs: Any):
        """Initialize the Cloudflare Workers AI client."""
        super().__init__(**kwargs)

        self.headers = {"Authorization": f"Bearer {self.api_token}"}

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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using Cloudflare Workers AI.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if self.strip_new_lines:
            texts = [text.replace("\n", " ") for text in texts]

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        embeddings = []

        for batch in batches:
            response = requests.post(
                url=self._inference_url,
                headers=self.headers,
                json={"text": batch},
            )
            response.raise_for_status()
            embeddings.extend(response.json()["result"]["data"])

        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously compute doc embeddings using Cloudflare Workers AI.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import httpx

        if self.strip_new_lines:
            texts = [text.replace("\n", " ") for text in texts]

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        embeddings = []

        async with httpx.AsyncClient() as client:
            for batch in batches:
                response = await client.post(
                    url=self._inference_url,
                    headers=self.headers,
                    json={"text": batch},
                )
                response.raise_for_status()
                embeddings.extend(response.json()["result"]["data"])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using Cloudflare Workers AI.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ") if self.strip_new_lines else text
        response = requests.post(
            url=self._inference_url,
            headers=self.headers,
            json={"text": [text]},
        )
        response.raise_for_status()
        return response.json()["result"]["data"][0]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously compute query embeddings using Cloudflare Workers AI.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        import httpx

        text = text.replace("\n", " ") if self.strip_new_lines else text

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self._inference_url,
                headers=self.headers,
                json={"text": [text]},
            )
            response.raise_for_status()

        return response.json()["result"]["data"][0]
