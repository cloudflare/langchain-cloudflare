"""Agent contracts exercised through production native adapters."""

# MARK: - IMPORTS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_cloudflare import (
    CloudflareWorkersAIDecisionModel,
    CloudflareWorkersAIExtractionModel,
)


# MARK: - INPUT FIXTURES
class Review(BaseModel):
    company: str = Field(alias="name", min_length=3, description="Full company name")
    approved: bool = Field(description="Is the agreement approved?")


EXTRACTION = {
    "model": "clairic",
    "truncated": False,
    "answers": {"f0": {"type": "text", "value": "Acme Corp", "span": [0, 9]}},
    "usage": {"input_tokens": 10, "output_tokens": 0},
}
DECISION = {
    "model": "clef",
    "answers": {"f1": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 20, "output_tokens": 0},
}
MESSAGES = {
    "messages": [{"role": "user", "content": "Acme Corp has an approved agreement."}]
}


def native_model(primary="extraction", binding=None):
    params = {
        "account_id": "account",
        "api_token": "secret-token",
        "ai_gateway": None,
        "binding": binding,
    }
    if primary == "extraction":
        return CloudflareWorkersAIExtractionModel(
            decision_model=CloudflareWorkersAIDecisionModel(**params), **params
        )
    return CloudflareWorkersAIDecisionModel(
        extraction_model=CloudflareWorkersAIExtractionModel(**params), **params
    )


def post_response(url, **kwargs):
    response = MagicMock()
    response.json.return_value = {
        "result": EXTRACTION if url.endswith("clairic") else DECISION
    }
    return response


def output_format(strategy, schema=Review):
    if strategy == "provider":
        return ProviderStrategy(schema)
    if strategy == "tool":
        return ToolStrategy(schema, handle_errors=False)
    return schema


def assert_agent_result(result):
    parsed = result["structured_response"]
    assert isinstance(parsed, Review)
    assert parsed.company == "Acme Corp"
    assert parsed.approved is True
    message = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
    assert message.response_metadata["native"] == {
        "extraction": EXTRACTION,
        "decision": DECISION,
    }
    assert message.usage_metadata == {
        "input_tokens": 30,
        "output_tokens": 0,
        "total_tokens": 30,
    }


# MARK: - STRATEGIES AND EXECUTION
@pytest.mark.parametrize("primary", ["extraction", "decision"])
@pytest.mark.parametrize("strategy", ["auto", "provider", "tool"])
def test_direct_create_agent(primary, strategy):
    agent = create_agent(
        model=native_model(primary), tools=[], response_format=output_format(strategy)
    )
    with patch(
        "langchain_cloudflare._native_models.requests.post", side_effect=post_response
    ) as post:
        result = agent.invoke(MESSAGES)
    assert_agent_result(result)
    assert post.call_count == 2


@pytest.mark.parametrize("primary", ["extraction", "decision"])
@pytest.mark.parametrize("strategy", ["auto", "provider", "tool"])
async def test_async_agent_uses_existing_binding(primary, strategy):
    binding = MagicMock()
    binding.run = AsyncMock(
        side_effect=lambda model, *args: (
            EXTRACTION if model.endswith("clairic") else DECISION
        )
    )
    agent = create_agent(
        model=native_model(primary, binding),
        tools=[],
        response_format=output_format(strategy),
    )
    assert_agent_result(await agent.ainvoke(MESSAGES))
    assert binding.run.await_count == 2


def test_agent_batch():
    agent = create_agent(model=native_model(), tools=[], response_format=Review)
    with patch(
        "langchain_cloudflare._native_models.requests.post", side_effect=post_response
    ) as post:
        results = agent.batch([MESSAGES, MESSAGES], config={"max_concurrency": 1})
    assert len(results) == 2
    for result in results:
        assert_agent_result(result)
    assert post.call_count == 4


async def test_agent_abatch():
    binding = MagicMock()
    binding.run = AsyncMock(
        side_effect=lambda model, *args: (
            EXTRACTION if model.endswith("clairic") else DECISION
        )
    )
    agent = create_agent(
        model=native_model(binding=binding), tools=[], response_format=Review
    )
    results = await agent.abatch([MESSAGES, MESSAGES], config={"max_concurrency": 1})
    assert len(results) == 2
    for result in results:
        assert_agent_result(result)
    assert binding.run.await_count == 4


def test_system_instructions_are_not_source_facts():
    agent = create_agent(
        model=native_model(),
        tools=[],
        response_format=Review,
        system_prompt="Only extract from user messages.",
    )
    with patch(
        "langchain_cloudflare._native_models.requests.post", side_effect=post_response
    ) as post:
        assert_agent_result(agent.invoke(MESSAGES))
    body = post.call_args_list[0].kwargs["json"]
    assert body["state"] == MESSAGES["messages"][0]["content"]
    assert "Only extract" in body["fields"]["f0"]["instructions"]


def test_image_message_uses_native_image_input():
    class Ticker(BaseModel):
        ticker: str

    model = CloudflareWorkersAIExtractionModel(
        account_id="account", api_token="token", ai_gateway=None
    )
    agent = create_agent(model=model, tools=[], response_format=Ticker)
    image = "data:image/png;base64,encoded-image"
    message = HumanMessage(content=[{"type": "image_url", "image_url": {"url": image}}])
    native = {
        "answers": {
            "f0": {"type": "text", "value": "ACME", "source": "image", "span": None}
        }
    }
    response = MagicMock()
    response.json.return_value = {"result": native}
    with patch(
        "langchain_cloudflare._native_models.requests.post", return_value=response
    ) as post:
        result = agent.invoke({"messages": [message]})
    assert result["structured_response"].ticker == "ACME"
    assert post.call_args.kwargs["json"]["state"] == ""
    assert post.call_args.kwargs["json"]["images"] == [image]


# MARK: - REJECTION AND VALIDATION
def test_executable_tools_rejected_before_network():
    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        raise AssertionError("An application tool must not be executed")

    for strategy in (Review, ToolStrategy(Review, handle_errors=False)):
        agent = create_agent(
            model=native_model(), tools=[echo], response_format=strategy
        )
        with patch("langchain_cloudflare._native_models.requests.post") as post:
            with pytest.raises(ValueError, match="tools|schema"):
                agent.invoke(MESSAGES)
        post.assert_not_called()


def test_unsupported_schema_rejected_before_network():
    class Group(BaseModel):
        reviews: list[Review]

    agent = create_agent(model=native_model(), tools=[], response_format=Group)
    with patch("langchain_cloudflare._native_models.requests.post") as post:
        with pytest.raises(ValueError, match="Unsupported structured-output field"):
            agent.invoke(MESSAGES)
    post.assert_not_called()


@pytest.mark.parametrize("strategy", ["auto", "provider", "tool"])
def test_original_pydantic_constraints_are_validated(strategy):
    class TooLong(Review):
        company: str = Field(alias="name", min_length=100)

    agent = create_agent(
        model=native_model(), tools=[], response_format=output_format(strategy, TooLong)
    )
    with patch(
        "langchain_cloudflare._native_models.requests.post", side_effect=post_response
    ) as post:
        with pytest.raises(Exception, match="validation|100"):
            agent.invoke(MESSAGES)
    assert post.call_count == 2


def test_agent_without_schema_rejected_before_network():
    agent = create_agent(model=native_model(), tools=[])
    with patch("langchain_cloudflare._native_models.requests.post") as post:
        with pytest.raises(ValueError, match="response_format"):
            agent.invoke(MESSAGES)
    post.assert_not_called()


def test_tool_strategy_validation_feedback_uses_original_source():
    bad = {**EXTRACTION, "answers": {"f0": {"type": "text", "value": "x"}}}
    bad_response = MagicMock()
    bad_response.json.return_value = {"result": bad}
    extraction_calls = 0

    def respond(url, **kwargs):
        nonlocal extraction_calls
        if url.endswith("clairic"):
            extraction_calls += 1
            if extraction_calls == 1:
                return bad_response
        return post_response(url, **kwargs)

    agent = create_agent(
        model=native_model(), tools=[], response_format=ToolStrategy(Review)
    )
    with patch(
        "langchain_cloudflare._native_models.requests.post", side_effect=respond
    ) as post:
        assert_agent_result(agent.invoke(MESSAGES, config={"recursion_limit": 4}))
    assert post.call_count == 4
    second_body = post.call_args_list[2].kwargs["json"]
    assert second_body["state"] == MESSAGES["messages"][0]["content"]
    assert "validation feedback" in second_body["fields"]["f0"]["instructions"]


def test_credentials_excluded_from_repr_serialization_and_metadata():
    model = native_model()
    assert "secret-token" not in repr(model)
    assert "headers" not in model.model_dump()
    assert "secret-token" not in str(model._identifying_params)
    assert "decision_model" not in model.model_dump()


def test_http_failure_still_propagates():
    agent = create_agent(model=native_model(), tools=[], response_format=Review)
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        side_effect=requests.HTTPError("upstream failed"),
    ):
        with pytest.raises(requests.HTTPError, match="upstream failed"):
            agent.invoke(MESSAGES)


def test_legacy_provider_binding_hint_is_not_sent_to_native_api():
    bound = native_model().bind_tools(
        [], strict=True, **ProviderStrategy(Review).to_model_kwargs()
    )
    with patch(
        "langchain_cloudflare._native_models.requests.post", side_effect=post_response
    ) as post:
        message = bound.invoke(MESSAGES["messages"])
    assert message.response_metadata["native"]["extraction"] == EXTRACTION
    assert all("strict" not in call.kwargs["json"] for call in post.call_args_list)


def test_remote_image_and_instruction_only_inputs_rejected():
    model = native_model().bind(
        response_format=ProviderStrategy(Review).to_model_kwargs()["response_format"]
    )
    for messages in (
        [SystemMessage(content="Extract a review")],
        [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    }
                ]
            )
        ],
    ):
        with patch("langchain_cloudflare._native_models.requests.post") as post:
            with pytest.raises(ValueError, match="source text|embedded"):
                model.invoke(messages)
        post.assert_not_called()
