"""Live agent-level structured output through native REST model adapters."""

# MARK: - IMPORTS
import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage

from langchain_cloudflare import (
    CloudflareWorkersAIDecisionModel,
    CloudflareWorkersAIExtractionModel,
)
from tests.native_structured_helpers import (
    NATIVE_AGENT_CASES,
    assert_native_agent_result,
)


# MARK: - REST AGENT TESTS
@pytest.mark.parametrize(
    "case_name,asynchronous,strategy",
    [
        ("invoice", False, "auto"),
        ("invoice", True, "auto"),
        ("decision", False, "auto"),
        ("decision", True, "auto"),
        ("mixed", False, "auto"),
        ("mixed", True, "auto"),
        ("mixed", False, "provider"),
        ("mixed", False, "tool"),
        ("image_data_url", False, "auto"),
        ("image_base64", True, "auto"),
    ],
    ids=[
        "invoice-sync",
        "invoice-async",
        "decision-sync",
        "decision-async",
        "mixed-sync",
        "mixed-async",
        "mixed-provider",
        "mixed-tool",
        "image-url-sync",
        "image-base64-async",
    ],
)
async def test_native_agent(case_name, asynchronous, strategy, account_id, api_token):
    if not account_id or not api_token:
        pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")
    case = NATIVE_AGENT_CASES[case_name]
    params = {"account_id": account_id, "api_token": api_token, "ai_gateway": None}
    decision = CloudflareWorkersAIDecisionModel(**params)
    model = (
        decision
        if case_name == "decision"
        else CloudflareWorkersAIExtractionModel(
            decision_model=decision if case_name == "mixed" else None, **params
        )
    )
    schema = case["schema"]
    response_format = (
        ProviderStrategy(schema)
        if strategy == "provider"
        else ToolStrategy(schema, handle_errors=False)
        if strategy == "tool"
        else schema
    )
    agent = create_agent(model=model, tools=[], response_format=response_format)
    inputs = {"messages": case["messages"]}
    config = {"recursion_limit": 2}
    result = (
        await agent.ainvoke(inputs, config=config)
        if asynchronous
        else agent.invoke(inputs, config=config)
    )
    assert isinstance(result["structured_response"], schema)
    message = next(
        message
        for message in reversed(result["messages"])
        if isinstance(message, AIMessage)
    )
    assert_native_agent_result(
        {
            "structured_response": result["structured_response"].model_dump(
                mode="json"
            ),
            "schema_type": type(result["structured_response"]).__name__,
            "raw": message.response_metadata["native"],
        },
        case,
    )


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_agent_batch(asynchronous, account_id, api_token):
    if not account_id or not api_token:
        pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")
    case = NATIVE_AGENT_CASES["invoice"]
    model = CloudflareWorkersAIExtractionModel(
        account_id=account_id, api_token=api_token, ai_gateway=None
    )
    agent = create_agent(model=model, tools=[], response_format=case["schema"])
    inputs = [{"messages": case["messages"]}] * 2
    results = (
        await agent.abatch(inputs, config={"max_concurrency": 1, "recursion_limit": 2})
        if asynchronous
        else agent.batch(inputs, config={"max_concurrency": 1, "recursion_limit": 2})
    )
    assert len(results) == 2
    for result in results:
        parsed = result["structured_response"]
        assert isinstance(parsed, case["schema"])
        assert parsed.model_dump(mode="json") == case["expected"]
