# MARK: - Imports
from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from ._errors import TokenErrors
from ._options import apply_reject_if_busy

# MARK: - Constants
DEFAULT_MODEL_NAME = "@cf/baai/bge-base-en-v1.5"


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
    headers: Dict[str, str] = {"Authorization": "Bearer "}
    ai_gateway: Optional[str] = Field(
        default_factory=from_env("AI_GATEWAY", default=None)
    )
    binding: Any = Field(default=None, exclude=True)
    """Workers AI binding (env.AI) for use in Python Workers."""
    reject_if_busy: Optional[bool] = None
    """Fail fast instead of queueing when Workers AI is at capacity.

    When True, a request that would otherwise wait in the capacity queue is
    rejected immediately with HTTP 429 and Cloudflare error code 3040
    ("Capacity temporarily exceeded, please try again"). Works on both the
    REST path (sent as ``options.rejectIfBusy`` in the request body) and the
    Workers AI binding (sent in the options argument to ``env.AI.run()``,
    which is the only place the binding reads it from).
    """

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
            raise ValueError(TokenErrors.NO_ACCOUNT_ID_SET)

        if not self.api_token or not self.api_token.get_secret_value():
            raise ValueError(TokenErrors.INSUFFICIENT_AI_TOKENS)

        self.headers = {"Authorization": f"Bearer {self.api_token.get_secret_value()}"}

        # Unified endpoint (see
        # https://blog.cloudflare.com/workers-ai-gateway-unification/):
        # AI Gateway routing no longer uses a separate
        # gateway.ai.cloudflare.com host -- requests always go to the
        # standard Workers AI endpoint, gated through cf-aig-gateway-id below
        # instead.
        self._inference_url = (
            f"{self.api_base_url}/{self.account_id}/ai/run/{self.model_name}"
        )
        if self.ai_gateway:
            self.headers["cf-aig-gateway-id"] = self.ai_gateway

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # MARK: - Request Payload
    def _embed_payload(self, texts: List[str]) -> Dict[str, Any]:
        """Build the request body for an embedding batch."""
        return apply_reject_if_busy({"text": texts}, self.reject_if_busy)

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

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        embeddings = []

        for batch in batches:
            response = requests.post(
                url=self._inference_url,
                headers=self.headers,
                json=self._embed_payload(batch),
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
        if self.strip_new_lines:
            texts = [text.replace("\n", " ") for text in texts]

        # Use binding if available (for Python Workers)
        if self.binding is not None:
            return await self._aembed_with_binding(texts)

        import httpx

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
                    json=self._embed_payload(batch),
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
        from .bindings import convert_payload_for_binding, create_binding_run_options

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        embeddings = []

        # Gateway and rejectIfBusy both belong in the run options argument --
        # the binding ignores rejectIfBusy inside the model input object.
        run_options = create_binding_run_options(
            gateway_id=self.ai_gateway,
            reject_if_busy=self.reject_if_busy,
        )

        for batch in batches:
            js_payload = convert_payload_for_binding({"text": batch})

            # Call the binding with optional options
            if run_options is not None:
                response = await self.binding.run(
                    self.model_name, js_payload, run_options
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
        """
        text = text.replace("\n", " ") if self.strip_new_lines else text
        response = requests.post(
            url=self._inference_url,
            headers=self.headers,
            json=self._embed_payload([text]),
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
        text = text.replace("\n", " ") if self.strip_new_lines else text

        # Use binding if available (for Python Workers)
        if self.binding is not None:
            embeddings = await self._aembed_with_binding([text])
            return embeddings[0]

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self._inference_url,
                headers=self.headers,
                json=self._embed_payload([text]),
            )
            response.raise_for_status()

        return response.json()["result"]["data"][0]
