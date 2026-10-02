"""Cloudflare Workers AI decision models (Clef and Clef Flash)."""

# MARK: - IMPORTS
from typing import Any, Dict, List, Literal, Optional, Union

import requests
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from ._errors import TokenErrors
from ._options import apply_reject_if_busy
from .bindings import (
    convert_binding_response_to_rest_format,
    convert_payload_for_binding,
    create_binding_run_options,
)

# MARK: - MODELS
DECISION_MODELS = ("@cf/cloudflare/clef", "@cf/cloudflare/clef-flash")


# MARK: - DECISION MODEL
class CloudflareWorkersAIDecisionModel(BaseModel):
    """Evaluate state against Clef's typed questions and return probabilities.

    Questions use Cloudflare's native ``noul`` (yes/no probability), ``choice``
    (categorical decision), or ``score`` (ordered rubric) schemas. The returned
    dictionary preserves ``model``, ``answers``, and ``usage`` from the API.
    These models use a decision API rather than chat messages or tool calls.

    REST usage::

        model = CloudflareWorkersAIDecisionModel()
        result = model.evaluate(
            state="Checkout has been failing for every customer.",
            questions={
                "urgent": {"type": "noul", "instructions": "Is this urgent?"}
            },
        )

    Python Worker usage::

        model = CloudflareWorkersAIDecisionModel(binding=self.env.AI)
        result = await model.aevaluate(state=state, questions=questions)

    Credentials default to ``CF_ACCOUNT_ID`` and ``CF_AI_API_TOKEN``. A Worker
    binding authenticates through the runtime and needs neither credential.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: Literal["@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"] = (
        "@cf/cloudflare/clef"
    )
    api_base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    account_id: str = Field(default_factory=from_env("CF_ACCOUNT_ID", default=""))
    api_token: SecretStr = Field(
        default_factory=secret_from_env("CF_AI_API_TOKEN", default="")
    )
    binding: Any = Field(default=None, exclude=True)
    ai_gateway: Optional[str] = Field(
        default_factory=from_env("AI_GATEWAY", default=None)
    )
    reject_if_busy: Optional[bool] = None
    timeout: float = 60.0
    headers: Dict[str, str] = Field(default_factory=dict)

    _inference_url: str = PrivateAttr(default="")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.binding is not None:
            return
        if not self.account_id:
            raise ValueError(TokenErrors.NO_ACCOUNT_ID_SET)
        if not self.api_token.get_secret_value():
            raise ValueError(TokenErrors.INSUFFICIENT_AI_TOKENS)
        self.headers = {
            **self.headers,
            "Authorization": f"Bearer {self.api_token.get_secret_value()}",
        }
        if self.ai_gateway:
            self.headers["cf-aig-gateway-id"] = self.ai_gateway
        self._inference_url = (
            f"{self.api_base_url}/{self.account_id}/ai/run/{self.model_name}"
        )

    # MARK: - PAYLOAD
    def _prepare_payload(
        self,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]],
    ) -> Dict[str, Any]:
        """Derive the required body selector from the endpoint model ID."""
        payload: Dict[str, Any] = {
            "model": self.model_name.rsplit("/", 1)[-1],
            "state": state,
            "questions": questions,
        }
        if images is not None:
            payload["images"] = images
        return payload

    # MARK: - EVALUATION
    def evaluate(
        self,
        *,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate text or JSON state through REST.

        ``questions`` maps 1–64 IDs to native question objects. Optional images
        are embedded PNG/JPEG/WebP data URLs or ``{content_type, base64}``
        objects; remote URLs are unsupported. Use :meth:`aevaluate` in Workers.
        """
        if self.binding is not None:
            raise ValueError("Use aevaluate() with a Workers AI binding")
        payload = self._prepare_payload(state, questions, images)
        response = requests.post(
            url=self._inference_url,
            headers=self.headers,
            json=apply_reject_if_busy(payload, self.reject_if_busy),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["result"]  # type: ignore[no-any-return]

    async def aevaluate(
        self,
        *,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the same native input through async REST or ``env.AI.run``.

        Returns the same dictionary as :meth:`evaluate`, including each
        question's probabilities, confidence where applicable, and token usage.
        """
        payload = self._prepare_payload(state, questions, images)
        if self.binding is not None:
            js_payload = convert_payload_for_binding(payload)
            options = create_binding_run_options(
                gateway_id=self.ai_gateway,
                reject_if_busy=self.reject_if_busy,
            )
            if options is None:
                result = await self.binding.run(self.model_name, js_payload)
            else:
                result = await self.binding.run(self.model_name, js_payload, options)
            return convert_binding_response_to_rest_format(result, self.model_name)[
                "result"
            ]  # type: ignore[no-any-return]

        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url=self._inference_url,
                headers=self.headers,
                json=apply_reject_if_busy(payload, self.reject_if_busy),
            )
            response.raise_for_status()
            return response.json()["result"]  # type: ignore[no-any-return]
