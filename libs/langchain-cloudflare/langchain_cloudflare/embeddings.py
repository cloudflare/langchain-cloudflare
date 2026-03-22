# MARK: - Imports
from typing import Any, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, secret_from_env
from more_itertools import chunked
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr
from typing_extensions import TypedDict

# MARK: - Constants
DEFAULT_MODEL_NAME = "@cf/baai/bge-base-en-v1.5"


# MARK: - Type Definitions
class Headers(TypedDict):
    Authorization: str
    "Authorization header for Cloudflare API"
    "Format: Bearer <api_token>"


# MARK: - CloudflareWorkersAIEmbeddings
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
            Cloudflare account ID. If not specified, will be read from
            the CF_ACCOUNT_ID environment variable.

        api_token: str
            Cloudflare Workers AI API token. If not specified, will be read from
            the CF_AI_API_TOKEN environment variable.

        model_name: str
            Embeddings model name on Workers AI (default: "@cf/baai/bge-base-en-v1.5")

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings

            # From environment variables
            cf = CloudflareWorkersAIEmbeddings()

            # Or with explicit credentials
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
    account_id: str = Field(default_factory=from_env("CF_ACCOUNT_ID", default=""))
    api_token: SecretStr = Field(
        default_factory=secret_from_env("CF_AI_API_TOKEN", default="")
    )
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 50
    strip_new_lines: bool = True
    headers: Headers = {"Authorization": "Bearer "}
    ai_gateway: Optional[str] = Field(
        default_factory=from_env("AI_GATEWAY", default=None)
    )
    binding: Any = Field(default=None, exclude=True)
    """Workers AI binding (env.AI) for use in Python Workers."""

    _inference_url: str = PrivateAttr()

    def __init__(self, **kwargs: Any):
        """Initialize the Cloudflare Workers AI client."""
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

        # Check if either api_token or CF_AI_API_TOKEN is provided
        if not any([self.api_token, self.api_token.get_secret_value()]):
            raise ValueError(
                "A Cloudflare API token must be provided either through "
                "the api_token parameter or CF_AI_API_TOKEN environment variable. "
                "Or pass the 'binding' parameter (env.AI) in a Python Worker."
            )

        # Set up headers
        self.headers = {"Authorization": f"Bearer {self.api_token.get_secret_value()}"}

        # Set up inference URL
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

    # MARK: - Embed Documents
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using Cloudflare Workers AI.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if self.strip_new_lines:
            texts = [text.replace("\n", " ") for text in texts]

        embeddings = []

        for batch in chunked(texts, self.batch_size):
            response = requests.post(
                url=self._inference_url,
                headers=self.headers,  # type: ignore
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

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        if self.strip_new_lines:
            texts = [text.replace("\n", " ") for text in texts]

        # Use binding if available (for Python Workers)
        if self.binding is not None:
            return await self._aembed_with_binding(texts)

        import httpx

        embeddings = []

        async with httpx.AsyncClient() as client:
            for batch in chunked(texts, self.batch_size):
                response = await client.post(
                    url=self._inference_url,
                    headers=self.headers,  # type: ignore
                    json={"text": batch},
                )
                response.raise_for_status()
                embeddings.extend(response.json()["result"]["data"])

        return embeddings

    # MARK: - Binding Helper
    async def _aembed_with_binding(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings using the Workers AI binding.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        from .bindings import convert_payload_for_binding, create_gateway_options

        embeddings = []

        # Create AI Gateway options if configured
        gateway_options = create_gateway_options(self.ai_gateway)

        for batch in chunked(texts, self.batch_size):
            js_payload = convert_payload_for_binding({"text": batch})

            # Call the binding with optional gateway
            if gateway_options is not None:
                response = await self.binding.run(
                    self.model_name, js_payload, gateway_options
                )
            else:
                response = await self.binding.run(self.model_name, js_payload)

            # Convert JS proxy to Python
            if hasattr(response, "to_py"):
                response = response.to_py()

            # Extract embeddings from response
            if isinstance(response, dict) and "data" in response:
                embeddings.extend(response["data"])
            elif isinstance(response, list):
                embeddings.extend(response)

        return embeddings

    # MARK: - Embed Query
    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using Cloudflare Workers AI.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        Raises:
            requests.HTTPError: If the request fails.
        """
        text = text.replace("\n", " ") if self.strip_new_lines else text
        response = requests.post(
            url=self._inference_url,
            headers=self.headers,  # type: ignore
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

        Raises:
            httpx.HTTPError: If the request fails.
        """
        text = text.replace("\n", " ") if self.strip_new_lines else text

        # Use binding if available (for Python Workers)
        if self.binding is not None:
            embeddings = await self._aembed_with_binding([text])
            return embeddings[0]

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self._inference_url,
                headers=self.headers,  # type: ignore
                json={"text": [text]},
            )
            response.raise_for_status()

        return response.json()["result"]["data"][0]
