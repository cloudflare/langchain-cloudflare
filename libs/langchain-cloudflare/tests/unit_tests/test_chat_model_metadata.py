"""Resolved model identity through production chat operations."""

# MARK: - IMPORTS
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import HumanMessage

from langchain_cloudflare import ChatCloudflareWorkersAI

# MARK: - FIXTURES
ROUTE = "dynamic/test-route"
MODELS = ["@cf/qwen/qwen3-30b-a3b-fp8", "@cf/zai-org/glm-5.2"]


def completion(model):
    return {
        "model": model,
        "choices": [{"message": {"content": "Hello"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


def chat_model(**kwargs):
    return ChatCloudflareWorkersAI(
        model=ROUTE,
        account_id="account",
        api_token="token",
        ai_gateway="gateway",
        **kwargs,
    )


# MARK: - INVOCATION
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_rest_invoke_identity(wrapped, asynchronous):
    body = completion(MODELS[0])
    if wrapped:
        body = {"result": body}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    with httpx.Client(
        transport=transport, base_url="https://example.test/client/v4/"
    ) as client:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/client/v4/"
        ) as async_client:
            llm = chat_model(
                endpoint_format="openai_compatible",
                client=client,
                async_client=async_client,
            )
            result = await llm.ainvoke("Hello") if asynchronous else llm.invoke("Hello")
    assert result.response_metadata["model_name"] == MODELS[0]
    assert result.response_metadata["requested_model"] == ROUTE


@pytest.mark.parametrize("reported_model", [None, ROUTE, *MODELS])
async def test_binding_identity(reported_model):
    binding = MagicMock()
    binding.run = AsyncMock(return_value=completion(reported_model))
    llm = chat_model(binding=binding)
    result = await llm.ainvoke("Hello")
    assert result.response_metadata["model_name"] == (
        reported_model if reported_model in MODELS else None
    )
    assert result.response_metadata["requested_model"] == ROUTE
    assert binding.run.await_count == 1


def test_generate_preserves_each_fallback_model():
    models = iter(MODELS)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=completion(next(models)))
    )
    with httpx.Client(
        transport=transport, base_url="https://example.test/client/v4/"
    ) as client:
        llm = chat_model(endpoint_format="openai_compatible", client=client)
        result = llm.generate([[HumanMessage(content="Hello")]] * 2)
    assert [
        g[0].message.response_metadata["model_name"] for g in result.generations
    ] == MODELS
    assert result.llm_output["model_name"] is None
    assert result.llm_output["requested_model"] == ROUTE
    assert result.llm_output["token_usage"]["total_tokens"] == 6


# MARK: - STREAMING
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reported_model", [None, MODELS[0]])
@pytest.mark.parametrize("model_only_final", [False, True])
async def test_stream_identity_is_not_concatenated(
    asynchronous, reported_model, model_only_final
):
    chunks = [
        {
            "choices": [{"delta": {"content": part}}],
            "model": None if model_only_final else reported_model,
        }
        for part in ("Hello", " world")
    ]
    if model_only_final:
        chunks.append({"choices": [], "model": reported_model})
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse + "data: [DONE]\n\n")
    )
    with httpx.Client(
        transport=transport, base_url="https://example.test/client/v4/"
    ) as client:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/client/v4/"
        ) as async_client:
            llm = chat_model(
                endpoint_format="openai_compatible",
                client=client,
                async_client=async_client,
            )
            messages = (
                [m async for m in llm.astream("Hello")]
                if asynchronous
                else list(llm.stream("Hello"))
            )
    combined = messages[0]
    for message in messages[1:]:
        combined += message
    assert combined.content == "Hello world"
    assert combined.response_metadata["model_name"] == reported_model
    assert combined.response_metadata["requested_model"] == ROUTE


def test_direct_model_identity_without_reported_model():
    llm = ChatCloudflareWorkersAI(model=MODELS[0], binding=object(), ai_gateway=None)
    result = llm._create_chat_result(completion(None))
    assert result.generations[0].message.response_metadata["model_name"] == MODELS[0]
    assert "requested_model" not in result.generations[0].message.response_metadata


def test_auto_router_batch_preserves_each_selected_model():
    models = ["@cf/qwen/qwen3.8-27b", "@cf/deepseek-ai/deepseek-v4-flash-0731"]
    selected = iter(models)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=completion(next(selected)))
    )
    with httpx.Client(
        transport=transport, base_url="https://example.test/client/v4/"
    ) as client:
        llm = ChatCloudflareWorkersAI(
            model="cloudflare/auto",
            account_id="account",
            api_token="token",
            ai_gateway="gateway",
            endpoint_format="openai_compatible",
            aig_allowed_models=models,
            client=client,
        )
        result = llm.generate([[HumanMessage(content="Hello")]] * 2)
    assert [
        generation[0].message.response_metadata["model_name"]
        for generation in result.generations
    ] == models
    assert result.llm_output["model_name"] is None
    assert result.llm_output["requested_model"] == "cloudflare/auto"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_auto_router_stream_preserves_routing_headers(asynchronous):
    selected_model = "@cf/qwen/qwen3.8-27b"
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}], "model": "cloudflare/auto"},
        {"choices": [{"delta": {"content": " world"}}], "model": selected_model},
    ]
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=sse + "data: [DONE]\n\n",
            headers={
                "cf-aig-routed-model": selected_model,
                "cf-aig-routing-reason": "cost_optimal_within_pool",
            },
        )
    )
    with httpx.Client(
        transport=transport, base_url="https://example.test/client/v4/"
    ) as client:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/client/v4/"
        ) as async_client:
            llm = ChatCloudflareWorkersAI(
                model="cloudflare/auto",
                account_id="account",
                api_token="token",
                ai_gateway="gateway",
                endpoint_format="openai_compatible",
                aig_allowed_models=[selected_model],
                client=client,
                async_client=async_client,
            )
            messages = (
                [message async for message in llm.astream("Hello")]
                if asynchronous
                else list(llm.stream("Hello"))
            )
    combined = messages[0]
    for message in messages[1:]:
        combined += message
    assert combined.content == "Hello world"
    assert combined.response_metadata == {
        "model_name": selected_model,
        "requested_model": "cloudflare/auto",
        "ai_gateway": {"routing_reason": "cost_optimal_within_pool"},
    }
