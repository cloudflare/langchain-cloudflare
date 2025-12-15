# ruff: noqa: T201, E501
# mypy: ignore-errors
"""
Integration tests for Cloudflare Workers AI models.

Tests cover:
- Basic invoke and batch
- Structured output (invoke and batch)
- Tool calling (invoke and batch)
- Agent pattern with create_agent (invoke and batch)

Models can be added/removed from the MODELS list to expand coverage.

Required environment variables:
    CF_ACCOUNT_ID: Cloudflare account ID
    CF_AI_API_TOKEN: Cloudflare AI API token

Optional environment variables:
    AI_GATEWAY: Cloudflare AI Gateway ID (if using a gateway)

Usage:
    # Set environment variables
    export CF_ACCOUNT_ID="your_account_id"
    export CF_AI_API_TOKEN="your_api_token"
    export AI_GATEWAY="your_gateway_id"  # optional

    # Run with pytest
    python -m pytest test_workersai_models.py -v -s

    # Or run directly
    python test_workersai_models.py
"""

import os
from typing import List, Optional

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_cloudflare import ChatCloudflareWorkersAI

# Agent imports
try:
    from langchain.agents import create_agent

    CREATE_AGENT_AVAILABLE = True
except ImportError:
    CREATE_AGENT_AVAILABLE = False


# Test models
MODELS = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/qwen/qwen3-30b-a3b-fp8",
]


# Pydantic schema for structured output
class Entity(BaseModel):
    """An entity mentioned in the announcement."""

    name: str = Field(description="Name of the entity")
    ticker: Optional[str] = Field(
        default=None, description="Stock ticker if applicable"
    )
    role: str = Field(description="Role of the entity in the announcement")


class Announcement(BaseModel):
    """A single announcement extracted from text."""

    type: str = Field(
        description="Type of announcement: partnership, investment, regulatory, milestone, event, m&a, none"
    )
    context: str = Field(description="Brief context of the announcement")
    entities: List[Entity] = Field(
        default_factory=list, description="Entities involved"
    )


class Data(BaseModel):
    """Extracted announcements from a press release."""

    announcements: List[Announcement] = Field(default_factory=list)


# Tool for tool calling tests
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72°F"


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    return f"The stock price of {ticker} is $150.25"


# Test fixtures
@pytest.fixture
def account_id():
    return os.environ.get("CF_ACCOUNT_ID")


@pytest.fixture
def api_token():
    return os.environ.get("CF_AI_API_TOKEN")


@pytest.fixture
def ai_gateway():
    return os.environ.get("AI_GATEWAY", None)


def create_llm(
    model: str, account_id: str, api_token: str, ai_gateway: Optional[str] = None
):
    """Create a ChatCloudflareWorkersAI instance."""
    return ChatCloudflareWorkersAI(
        account_id=account_id,
        api_token=api_token,
        model=model,
        temperature=0.0,
        ai_gateway=ai_gateway,
    )


class TestStructuredOutput:
    """Test structured output across Workers AI models."""

    SAMPLE_TEXT = """
    Acme Corp (NYSE: ACME) today announced a strategic partnership with
    TechGiant Inc to jointly develop next-generation AI solutions.
    The partnership will combine Acme's expertise in cloud infrastructure
    with TechGiant's machine learning capabilities.
    """

    @pytest.mark.parametrize("model", MODELS)
    def test_structured_output_invoke(self, model, account_id, api_token, ai_gateway):
        """Test structured output with invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        structured_llm = llm.with_structured_output(Data)

        result = structured_llm.invoke(
            f"Extract announcements from this text:\n\n{self.SAMPLE_TEXT}"
        )

        print(f"\n[{model}] Structured Output (invoke):")
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"
        assert isinstance(result, (dict, Data)), (
            f"Unexpected type {type(result)} for {model}"
        )

        # Check structure
        if isinstance(result, dict):
            assert "announcements" in result, f"Missing 'announcements' key for {model}"
        else:
            assert hasattr(result, "announcements"), (
                f"Missing 'announcements' attr for {model}"
            )

    @pytest.mark.parametrize("model", MODELS)
    def test_structured_output_batch(self, model, account_id, api_token, ai_gateway):
        """Test structured output with batch()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        structured_llm = llm.with_structured_output(Data)

        texts = [
            f"Extract announcements from this text:\n\n{self.SAMPLE_TEXT}",
            "Extract announcements from this text:\n\nApple Inc announced record Q4 earnings, beating analyst expectations.",
        ]

        results = structured_llm.batch(texts, config={"max_concurrency": 2})

        print(f"\n[{model}] Structured Output (batch):")
        for i, result in enumerate(results):
            print(f"  Result {i} type: {type(result)}")
            print(f"  Result {i}: {result}")

        assert len(results) == 2, f"Expected 2 results, got {len(results)} for {model}"

        for i, result in enumerate(results):
            assert result is not None, f"Result {i} is None for {model}"


class TestToolCalling:
    """Test tool calling across Workers AI models."""

    @pytest.mark.parametrize("model", MODELS)
    def test_tool_calling_invoke(self, model, account_id, api_token, ai_gateway):
        """Test tool calling with invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        llm_with_tools = llm.bind_tools([get_weather, get_stock_price])

        result = llm_with_tools.invoke("What's the weather in San Francisco?")

        print(f"\n[{model}] Tool Calling (invoke):")
        print(f"  Result type: {type(result)}")
        print(f"  Content: {result.content}")
        print(f"  Tool calls: {result.tool_calls}")

        # Model should either call the tool or respond with content
        assert result is not None, f"Result is None for {model}"

        # Check if tool was called
        if result.tool_calls:
            assert len(result.tool_calls) > 0, f"Empty tool_calls for {model}"
            tool_call = result.tool_calls[0]
            assert "name" in tool_call, f"Missing 'name' in tool_call for {model}"
            assert tool_call["name"] == "get_weather", f"Wrong tool called for {model}"
            assert "args" in tool_call, f"Missing 'args' in tool_call for {model}"
            print(f"  Tool call successful: {tool_call}")
        else:
            print(
                f"  No tool call made, content: {result.content[:200] if result.content else 'empty'}"
            )

    @pytest.mark.parametrize("model", MODELS)
    def test_tool_calling_multi_turn(self, model, account_id, api_token, ai_gateway):
        """Test multi-turn tool calling conversation.

        This tests the full flow:
        1. User asks a question
        2. Model responds with a tool call
        3. We execute the tool and send the result back
        4. Model responds with final answer

        This exercises the is_llama_model logic in _create_message_dicts()
        which formats tool call history when sending back to the API.
        """
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        from langchain_core.messages import HumanMessage, ToolMessage

        llm = create_llm(model, account_id, api_token, ai_gateway)
        llm_with_tools = llm.bind_tools([get_weather, get_stock_price])

        # Step 1: Initial user message
        messages = [HumanMessage(content="What's the weather in San Francisco?")]

        # Step 2: Get model response (should be a tool call)
        response1 = llm_with_tools.invoke(messages)

        print(f"\n[{model}] Multi-turn Tool Calling:")
        print("  Step 1 - Initial response:")
        print(
            f"    Content: {response1.content[:100] if response1.content else 'empty'}"
        )
        print(f"    Tool calls: {response1.tool_calls}")

        assert response1 is not None, f"Response 1 is None for {model}"

        if not response1.tool_calls:
            print("  WARN: No tool call made, skipping multi-turn test")
            return

        # Step 3: Execute the tool and add messages to history
        tool_call = response1.tool_calls[0]
        tool_result = get_weather.invoke(tool_call["args"])

        messages.append(response1)  # Add AI message with tool call
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )

        print("  Step 2 - Tool executed:")
        print(f"    Tool: {tool_call['name']}")
        print(f"    Args: {tool_call['args']}")
        print(f"    Result: {tool_result}")

        # Step 4: Get final response from model
        response2 = llm_with_tools.invoke(messages)

        print("  Step 3 - Final response:")
        print(
            f"    Content: {response2.content[:200] if response2.content else 'empty'}"
        )
        print(f"    Tool calls: {response2.tool_calls}")

        assert response2 is not None, f"Response 2 is None for {model}"
        # Final response should have content (not another tool call)
        assert response2.content, f"Final response has no content for {model}"
        print("  Status: PASS")

    @pytest.mark.parametrize("model", MODELS)
    def test_tool_calling_batch(self, model, account_id, api_token, ai_gateway):
        """Test tool calling with batch()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        llm_with_tools = llm.bind_tools([get_weather, get_stock_price])

        queries = [
            "What's the weather in New York?",
            "What's the stock price of AAPL?",
        ]

        results = llm_with_tools.batch(queries, config={"max_concurrency": 2})

        print(f"\n[{model}] Tool Calling (batch):")
        for i, result in enumerate(results):
            print(f"  Result {i}:")
            print(
                f"    Content: {result.content[:100] if result.content else 'empty'}..."
            )
            print(f"    Tool calls: {result.tool_calls}")

        assert len(results) == 2, f"Expected 2 results, got {len(results)} for {model}"

        for i, result in enumerate(results):
            assert result is not None, f"Result {i} is None for {model}"


class TestCreateAgent:
    """Test create_agent pattern across Workers AI models."""

    SYSTEM_PROMPT = """You are a press release analyst. Extract announcements from the given text.
    Classify each announcement as one of: partnership, investment, regulatory, milestone, event, m&a, none.
    Return the results in the structured format."""

    @pytest.mark.skipif(
        not CREATE_AGENT_AVAILABLE, reason="langchain.agents.create_agent not available"
    )
    @pytest.mark.parametrize("model", MODELS)
    def test_create_agent_structured_output_invoke(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test create_agent with structured output using invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        agent = create_agent(
            model=llm,
            response_format=Data,
            system_prompt=self.SYSTEM_PROMPT,
            tools=[],
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Text: Acme Corp announced a partnership with TechGiant Inc.",
                    }
                ]
            }
        )

        print(f"\n[{model}] create_agent Structured Output (invoke):")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"

    @pytest.mark.skipif(
        not CREATE_AGENT_AVAILABLE, reason="langchain.agents.create_agent not available"
    )
    @pytest.mark.parametrize("model", MODELS)
    def test_create_agent_structured_output_batch(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test create_agent with structured output using batch()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        agent = create_agent(
            model=llm,
            response_format=Data,
            system_prompt=self.SYSTEM_PROMPT,
            tools=[],
        )

        batch_inputs = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Text: Acme Corp announced a partnership with TechGiant Inc.",
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Text: Apple reported record Q4 earnings.",
                    }
                ]
            },
        ]

        results = agent.batch(batch_inputs, config={"max_concurrency": 2})

        print(f"\n[{model}] create_agent Structured Output (batch):")
        for i, r in enumerate(results):
            print(f"  Result {i}: {r}")

        assert len(results) == 2, f"Expected 2 results, got {len(results)} for {model}"
        for i, result in enumerate(results):
            assert result is not None, f"Result {i} is None for {model}"

    @pytest.mark.skipif(
        not CREATE_AGENT_AVAILABLE, reason="langchain.agents.create_agent not available"
    )
    @pytest.mark.parametrize("model", MODELS)
    def test_create_agent_tools_invoke(self, model, account_id, api_token, ai_gateway):
        """Test create_agent with tools using invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        agent = create_agent(
            model=llm,
            tools=[get_weather, get_stock_price],
        )

        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "What's the weather in San Francisco?"}
                ]
            }
        )

        print(f"\n[{model}] create_agent Tools (invoke):")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"

    @pytest.mark.skipif(
        not CREATE_AGENT_AVAILABLE, reason="langchain.agents.create_agent not available"
    )
    @pytest.mark.parametrize("model", MODELS)
    def test_create_agent_tools_batch(self, model, account_id, api_token, ai_gateway):
        """Test create_agent with tools using batch()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        agent = create_agent(
            model=llm,
            tools=[get_weather, get_stock_price],
        )

        batch_inputs = [
            {"messages": [{"role": "user", "content": "What's the weather in NYC?"}]},
            {
                "messages": [
                    {"role": "user", "content": "What's the stock price of MSFT?"}
                ]
            },
        ]

        results = agent.batch(batch_inputs, config={"max_concurrency": 2})

        print(f"\n[{model}] create_agent Tools (batch):")
        for i, r in enumerate(results):
            print(f"  Result {i}: {r}")

        assert len(results) == 2, f"Expected 2 results, got {len(results)} for {model}"
        for i, result in enumerate(results):
            assert result is not None, f"Result {i} is None for {model}"


class TestBasicInvoke:
    """Test basic invoke/batch across Workers AI models."""

    @pytest.mark.parametrize("model", MODELS)
    def test_basic_invoke(self, model, account_id, api_token, ai_gateway):
        """Test basic invoke returns content."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        result = llm.invoke("Say 'Hello World' and nothing else.")

        print(f"\n[{model}] Basic Invoke:")
        print(f"  Content: {result.content}")

        assert result is not None, f"Result is None for {model}"
        assert result.content, f"Empty content for {model}"
        assert "hello" in result.content.lower(), f"Unexpected response for {model}"

    @pytest.mark.parametrize("model", MODELS)
    def test_basic_batch(self, model, account_id, api_token, ai_gateway):
        """Test basic batch returns content."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        queries = [
            "Say 'Hello' and nothing else.",
            "Say 'World' and nothing else.",
        ]

        results = llm.batch(queries, config={"max_concurrency": 2})

        print(f"\n[{model}] Basic Batch:")
        for i, result in enumerate(results):
            print(f"  Result {i}: {result.content}")

        assert len(results) == 2, f"Expected 2 results, got {len(results)} for {model}"

        for i, result in enumerate(results):
            assert result is not None, f"Result {i} is None for {model}"
            assert result.content, f"Empty content for result {i} for {model}"


if __name__ == "__main__":
    # Run with: python -m pytest test_workersai_models.py -v -s
    # Or directly: python test_workersai_models.py

    import sys

    # Check for env vars
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_AI_API_TOKEN")
    ai_gateway = os.environ.get("AI_GATEWAY")

    if not account_id or not api_token:
        print("Please set CF_ACCOUNT_ID and CF_AI_API_TOKEN environment variables")
        sys.exit(1)

    print("=" * 60)
    print("Testing Cloudflare Workers AI Models")
    print("=" * 60)

    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Model: {model}")
        print("=" * 60)

        llm = create_llm(model, account_id, api_token, ai_gateway)

        # Test 1: Basic invoke
        print("\n[Test 1] Basic Invoke:")
        try:
            result = llm.invoke("Say 'Hello World' and nothing else.")
            print(f"  Content: {result.content[:200] if result.content else 'EMPTY'}")
            print(
                "  Status: PASS" if result.content else "  Status: FAIL - empty content"
            )
        except Exception as e:
            print(f"  Status: FAIL - {e}")

        # Test 2: Structured output invoke
        print("\n[Test 2] Structured Output (invoke):")
        try:
            structured_llm = llm.with_structured_output(Data)
            result = structured_llm.invoke(
                "Extract announcements: Acme Corp announced a partnership with TechGiant Inc."
            )
            print(f"  Result: {result}")
            print(f"  Type: {type(result)}")
            print("  Status: PASS" if result else "  Status: FAIL - None result")
        except Exception as e:
            print(f"  Status: FAIL - {e}")

        # Test 3: Structured output batch
        print("\n[Test 3] Structured Output (batch):")
        try:
            structured_llm = llm.with_structured_output(Data)
            results = structured_llm.batch(
                [
                    "Extract announcements: Acme Corp announced a partnership.",
                    "Extract announcements: Apple reported record earnings.",
                ],
                config={"max_concurrency": 2},
            )
            print(f"  Results count: {len(results)}")
            for i, r in enumerate(results):
                print(f"  Result {i}: {r}")
            all_valid = all(r is not None for r in results)
            print(
                "  Status: PASS" if all_valid else "  Status: FAIL - some None results"
            )
        except Exception as e:
            print(f"  Status: FAIL - {e}")

        # Test 4: Tool calling invoke
        print("\n[Test 4] Tool Calling (invoke):")
        try:
            llm_with_tools = llm.bind_tools([get_weather, get_stock_price])
            result = llm_with_tools.invoke("What's the weather in San Francisco?")
            print(
                f"  Content: {result.content[:100] if result.content else 'empty'}..."
            )
            print(f"  Tool calls: {result.tool_calls}")
            has_tool_call = len(result.tool_calls) > 0 if result.tool_calls else False
            print(
                "  Status: PASS (tool called)"
                if has_tool_call
                else "  Status: WARN - no tool call"
            )
        except Exception as e:
            print(f"  Status: FAIL - {e}")

        # Test 5: Tool calling batch
        print("\n[Test 5] Tool Calling (batch):")
        try:
            llm_with_tools = llm.bind_tools([get_weather, get_stock_price])
            results = llm_with_tools.batch(
                [
                    "What's the weather in NYC?",
                    "What's the stock price of MSFT?",
                ],
                config={"max_concurrency": 2},
            )
            print(f"  Results count: {len(results)}")
            for i, r in enumerate(results):
                print(f"  Result {i} tool_calls: {r.tool_calls}")
            print("  Status: PASS" if len(results) == 2 else "  Status: FAIL")
        except Exception as e:
            print(f"  Status: FAIL - {e}")

    # Test 6: create_agent with structured output (invoke)
    if CREATE_AGENT_AVAILABLE:
        print("\n[Test 6] create_agent with Structured Output (invoke):")
        try:
            system_prompt = """You are a press release analyst. Extract announcements from the given text.
            Classify each announcement as one of: partnership, investment, regulatory, milestone, event, m&a, none.
            Return the results in the structured format."""

            agent = create_agent(
                model=llm,
                response_format=Data,
                system_prompt=system_prompt,
                tools=[],
            )

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Text: Acme Corp announced a partnership with TechGiant Inc.",
                        }
                    ]
                }
            )

            print(
                f"  Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}"
            )
            if isinstance(result, dict) and "structured_response" in result:
                print(f"  Structured response: {result['structured_response']}")
                print("  Status: PASS")
            else:
                print(f"  Result: {result}")
                print("  Status: WARN - unexpected format")
        except Exception as e:
            print(f"  Status: FAIL - {e}")

        # Test 7: create_agent with structured output (batch)
        print("\n[Test 7] create_agent with Structured Output (batch):")
        try:
            system_prompt = """You are a press release analyst. Extract announcements from the given text.
            Classify each announcement as one of: partnership, investment, regulatory, milestone, event, m&a, none.
            Return the results in the structured format."""

            agent = create_agent(
                model=llm,
                response_format=Data,
                system_prompt=system_prompt,
                tools=[],
            )

            batch_inputs = [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Text: Acme Corp announced a partnership with TechGiant Inc.",
                        }
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Text: Apple reported record Q4 earnings.",
                        }
                    ]
                },
            ]

            results = agent.batch(batch_inputs, config={"max_concurrency": 2})

            print(f"  Results count: {len(results)}")
            for i, r in enumerate(results):
                if isinstance(r, dict) and "structured_response" in r:
                    print(f"  Result {i}: {r['structured_response']}")
                else:
                    print(f"  Result {i}: {r}")
            all_valid = all(r is not None for r in results)
            print(
                "  Status: PASS" if all_valid else "  Status: FAIL - some None results"
            )
        except Exception as e:
            print(f"  Status: FAIL - {e}")

        # Test 8: create_agent with tools (invoke)
        print("\n[Test 8] create_agent with Tools (invoke):")
        try:
            agent = create_agent(
                model=llm,
                tools=[get_weather, get_stock_price],
            )

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What's the weather in San Francisco?",
                        }
                    ]
                }
            )

            print(
                f"  Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}"
            )
            print(f"  Result: {result}")
            print("  Status: PASS")
        except Exception as e:
            print(f"  Status: FAIL - {e}")

        # Test 9: create_agent with tools (batch)
        print("\n[Test 9] create_agent with Tools (batch):")
        try:
            agent = create_agent(
                model=llm,
                tools=[get_weather, get_stock_price],
            )

            batch_inputs = [
                {
                    "messages": [
                        {"role": "user", "content": "What's the weather in NYC?"}
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "What's the stock price of MSFT?"}
                    ]
                },
            ]

            results = agent.batch(batch_inputs, config={"max_concurrency": 2})

            print(f"  Results count: {len(results)}")
            for i, r in enumerate(results):
                print(f"  Result {i}: {r}")
            print("  Status: PASS" if len(results) == 2 else "  Status: FAIL")
        except Exception as e:
            print(f"  Status: FAIL - {e}")
    else:
        print(
            "\n[Test 6-9] create_agent tests skipped - langchain.agents.create_agent not available"
        )

    print("\n" + "=" * 60)
    print("Tests Complete")
    print("=" * 60)
