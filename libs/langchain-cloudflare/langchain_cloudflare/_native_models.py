"""Shared transport for native Workers AI decision and extraction models."""

# MARK: - IMPORTS
from typing import Any, Dict, Optional

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


# MARK: - NATIVE MODEL TRANSPORT
class WorkersAINativeModel(BaseModel):
    """Reuse authentication and native result handling across typed model APIs."""

    model_name: str
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

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

    # MARK: - REST TRANSPORT
    def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            url=self._inference_url,
            headers=self.headers,
            json=apply_reject_if_busy(payload, self.reject_if_busy),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["result"]  # type: ignore[no-any-return]

    # MARK: - ASYNC TRANSPORT
    async def _arun(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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
