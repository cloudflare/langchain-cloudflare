"""Decision adapter transport contracts; live coverage is in integration suites."""

# MARK: - IMPORTS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from langchain_cloudflare import CloudflareWorkersAIDecisionModel
from langchain_cloudflare.decision_models import DECISION_MODELS

# MARK: - FIXTURES
INPUT = {
    "state": {"status": "outage"},
    "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}},
    "images": [{"content_type": "image/png", "base64": "encoded-image"}],
}
RESULT = {
    "model": "clef",
    "answers": {"urgent": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 10, "output_tokens": 0},
}


# MARK: - REST CONTRACT
@pytest.mark.parametrize("model", DECISION_MODELS)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_rest_transport(model, asynchronous):
    """Both transports send native inputs and unwrap only the REST envelope."""
    decision_model = CloudflareWorkersAIDecisionModel(
        model_name=model,
        account_id="account",
        api_token="token",
        ai_gateway="gateway",
        reject_if_busy=True,
        timeout=45,
    )
    response = MagicMock()
    response.json.return_value = {"result": RESULT, "success": True}
    if asynchronous:
        with patch("httpx.AsyncClient") as client_type:
            client = client_type.return_value.__aenter__.return_value
            client.post = AsyncMock(return_value=response)
            result = await decision_model.aevaluate(**INPUT)
            kwargs = client.post.call_args.kwargs
            assert client_type.call_args.kwargs["timeout"] == 45
    else:
        with patch(
            "langchain_cloudflare.decision_models.requests.post", return_value=response
        ) as post:
            result = decision_model.evaluate(**INPUT)
            kwargs = post.call_args.kwargs
            assert kwargs["timeout"] == 45
    assert result == RESULT
    response.raise_for_status.assert_called_once()
    assert kwargs["url"] == (
        f"https://api.cloudflare.com/client/v4/accounts/account/ai/run/{model}"
    )
    assert kwargs["headers"] == {
        "Authorization": "Bearer token",
        "cf-aig-gateway-id": "gateway",
    }
    assert kwargs["json"] == {
        **INPUT,
        "model": model.rsplit("/", 1)[-1],
        "options": {"rejectIfBusy": True},
    }


# MARK: - BINDING CONTRACT
@pytest.mark.parametrize("model", DECISION_MODELS)
@pytest.mark.parametrize("with_options", [False, True])
async def test_binding_transport(model, with_options):
    """Record the actual binding call, including native payload and run options."""
    binding = MagicMock()
    binding.run = AsyncMock(return_value=RESULT)
    decision_model = CloudflareWorkersAIDecisionModel(
        model_name=model,
        binding=binding,
        account_id="",
        api_token="",
        ai_gateway="gateway" if with_options else None,
        reject_if_busy=with_options,
    )
    assert await decision_model.aevaluate(**INPUT) == RESULT
    payload = {**INPUT, "model": model.rsplit("/", 1)[-1]}
    if with_options:
        binding.run.assert_awaited_once_with(
            model, payload, {"gateway": {"id": "gateway"}, "rejectIfBusy": True}
        )
    else:
        binding.run.assert_awaited_once_with(model, payload)


async def test_binding_js_response_preserves_answers_and_usage():
    response = MagicMock()
    response.to_py.return_value = RESULT
    binding = MagicMock()
    binding.run = AsyncMock(return_value=response)
    decision_model = CloudflareWorkersAIDecisionModel(binding=binding, ai_gateway=None)
    assert (
        await decision_model.aevaluate(
            state=INPUT["state"], questions=INPUT["questions"]
        )
        == RESULT
    )
    payload = binding.run.call_args.args[1]
    assert "images" not in payload
    assert "options" not in payload


# MARK: - ERRORS
@pytest.mark.parametrize(
    "account_id,api_token,message",
    [("", "token", "account ID"), ("account", "", "API token")],
)
def test_missing_credentials(account_id, api_token, message):
    with pytest.raises(ValueError, match=message):
        CloudflareWorkersAIDecisionModel(account_id=account_id, api_token=api_token)


def test_chat_model_rejected():
    with pytest.raises(ValidationError):
        CloudflareWorkersAIDecisionModel(model_name="@cf/qwen/qwen3-30b-a3b-fp8")


def test_sync_binding_requires_async():
    model = CloudflareWorkersAIDecisionModel(binding=MagicMock())
    with pytest.raises(ValueError, match="aevaluate"):
        model.evaluate(**INPUT)


def test_rest_http_error_propagates():
    model = CloudflareWorkersAIDecisionModel(account_id="account", api_token="token")
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("Invalid questions")
    with (
        patch(
            "langchain_cloudflare.decision_models.requests.post", return_value=response
        ),
        pytest.raises(requests.HTTPError, match="Invalid questions"),
    ):
        model.evaluate(**INPUT)
    response.json.assert_not_called()
