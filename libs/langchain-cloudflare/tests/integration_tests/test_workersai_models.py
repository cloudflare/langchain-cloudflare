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

import base64
import os
import uuid
from typing import List, Optional

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_cloudflare.rerankers import CloudflareWorkersAIReranker

# Agent imports
try:
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    CREATE_AGENT_AVAILABLE = True
except ImportError:
    CREATE_AGENT_AVAILABLE = False


# Test models
MODELS = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/moonshotai/kimi-k2.5",
    "@cf/moonshotai/kimi-k2.6",
    "@cf/google/gemma-4-26b-a4b-it",
]

# Models live-validated in this suite for method='json_schema'.
# Excluded families stay out of this list until their runtime behavior is
# verified end-to-end in integration tests.
JSON_SCHEMA_MODELS = [m for m in MODELS if "mistral" not in m and "gpt-oss" not in m]

# Models confirmed to support vision (image input). Per CF docs and live testing.
VISION_MODELS = [
    "@cf/moonshotai/kimi-k2.5",
    "@cf/moonshotai/kimi-k2.6",
    "@cf/google/gemma-4-26b-a4b-it",
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


def get_text_content(content):
    """Extract text from content that may be a string or list of content blocks.

    When reasoning models return content blocks, this extracts only the text
    blocks and joins them. For plain string content, returns as-is.
    """
    if isinstance(content, list):
        text_parts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return " ".join(text_parts)
    return content or ""


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

    @pytest.mark.parametrize("model", JSON_SCHEMA_MODELS)
    def test_structured_output_json_schema_method_invoke(
        self, model, account_id, api_token, ai_gateway
    ):
        """method='json_schema' should work for models that support json_object mode."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        structured_llm = llm.with_structured_output(Data, method="json_schema")

        result = structured_llm.invoke(
            f"Extract announcements from this text:\n\n{self.SAMPLE_TEXT}"
        )

        print(f"\n[{model}] Structured Output json_schema (invoke):")
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"
        assert isinstance(result, (dict, Data)), (
            f"Unexpected type {type(result)} for {model}"
        )

        if isinstance(result, dict):
            assert "announcements" in result, f"Missing 'announcements' key for {model}"
        else:
            assert hasattr(result, "announcements"), (
                f"Missing 'announcements' attr for {model}"
            )


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
                f"  No tool call made, content: {get_text_content(result.content)[:200] if result.content else 'empty'}"
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
            f"    Content: {get_text_content(response1.content)[:100] if response1.content else 'empty'}"
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
            f"    Content: {get_text_content(response2.content)[:200] if response2.content else 'empty'}"
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
                f"    Content: {get_text_content(result.content)[:100] if result.content else 'empty'}..."
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


# MARK: - ToolStrategy JSON Schema Tests


class TestToolStrategyJsonSchema:
    """Test create_agent with ToolStrategy using a JSON schema dict via REST API."""

    SYSTEM_PROMPT = (
        "You are a press release analyst. Extract announcements from the "
        "given text. Classify each announcement as one of: partnership, "
        "investment, regulatory, milestone, event, m&a, none. "
        "Return the results in the structured format."
    )

    @pytest.mark.skipif(
        not CREATE_AGENT_AVAILABLE,
        reason="langchain.agents.create_agent not available",
    )
    @pytest.mark.parametrize("model", MODELS)
    def test_tool_strategy_json_schema_invoke(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test create_agent with ToolStrategy(json_schema_dict) via REST API."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        # Use JSON schema dict instead of Pydantic model
        json_schema = Data.model_json_schema()

        agent = create_agent(
            model=llm,
            response_format=ToolStrategy(json_schema),
            system_prompt=self.SYSTEM_PROMPT,
            tools=[],
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Text: Acme Corp announced a "
                            "partnership with TechGiant Inc."
                        ),
                    }
                ]
            }
        )

        print(f"\n[{model}] ToolStrategy JSON Schema (invoke):")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"
        # ToolStrategy with json_schema kind returns raw dict
        if isinstance(result, dict):
            structured = result.get("structured_response", result)
            assert structured is not None


# MARK: - Reranker Tests


class TestReranker:
    """Test CloudflareWorkersAIReranker via REST API."""

    def test_rerank_basic(self, account_id, api_token):
        """Test reranker returns ranked results with scores."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        reranker = CloudflareWorkersAIReranker(
            model_name="@cf/baai/bge-reranker-base",
            account_id=account_id,
            api_token=api_token,
        )

        results = reranker.rerank(
            query="What is the capital of France?",
            documents=[
                "Paris is the capital and largest city of France.",
                "Berlin is the capital of Germany.",
                "The Eiffel Tower is located in Paris, France.",
                "London is the capital of the United Kingdom.",
            ],
            top_k=3,
        )

        assert len(results) > 0, "Reranker returned no results"
        assert len(results) <= 3
        # Results should have index and score
        for r in results:
            assert hasattr(r, "index")
            assert hasattr(r, "score")
            assert r.score >= 0.0

    @pytest.mark.asyncio
    async def test_arerank_basic(self, account_id, api_token):
        """Test async reranker returns ranked results with scores."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        reranker = CloudflareWorkersAIReranker(
            model_name="@cf/baai/bge-reranker-base",
            account_id=account_id,
            api_token=api_token,
        )

        results = await reranker.arerank(
            query="What is the capital of France?",
            documents=[
                "Paris is the capital and largest city of France.",
                "Berlin is the capital of Germany.",
                "The Eiffel Tower is located in Paris, France.",
                "London is the capital of the United Kingdom.",
            ],
            top_k=3,
        )

        assert len(results) > 0, "Reranker returned no results"
        assert len(results) <= 3
        for r in results:
            assert hasattr(r, "index")
            assert hasattr(r, "score")
            assert r.score >= 0.0


# MARK: - Basic Invoke Tests


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
        text = get_text_content(result.content)
        assert "hello" in text.lower(), f"Unexpected response for {model}"

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


# MARK: - Reasoning Content Tests


class TestReasoningContent:
    """Test reasoning_content extraction from models that support it."""

    REASONING_MODELS = [
        "@cf/qwen/qwen3-30b-a3b-fp8",
        "@cf/zai-org/glm-4.7-flash",
        "@cf/openai/gpt-oss-120b",
        "@cf/openai/gpt-oss-20b",
        "@cf/moonshotai/kimi-k2.5",
        "@cf/moonshotai/kimi-k2.6",
        "@cf/google/gemma-4-26b-a4b-it",
        "@cf/nvidia/nemotron-3-120b-a12b",
    ]

    @staticmethod
    def _extract_reasoning(result):
        """Extract reasoning from content blocks."""
        if isinstance(result.content, list):
            thinking_blocks = [
                b
                for b in result.content
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            return thinking_blocks[0]["thinking"] if thinking_blocks else ""
        return ""

    @pytest.mark.parametrize("model", REASONING_MODELS)
    def test_reasoning_content_sync(self, model, account_id, api_token, ai_gateway):
        """Test that reasoning_content appears as content blocks."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        result = llm.invoke("What is 25 * 37? Think step by step.")

        reasoning = self._extract_reasoning(result)

        print(f"\n[{model}] Reasoning Content (sync):")
        print(f"  Content: {str(result.content)[:200]}")
        print(f"  Reasoning: {reasoning[:200]}")
        print(f"  Content type: {type(result.content).__name__}")

        assert isinstance(result.content, list), (
            f"Expected list content blocks for {model}, got {type(result.content)}"
        )
        thinking_blocks = [
            b
            for b in result.content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) > 0, (
            f"Expected thinking block in content for {model}"
        )
        assert len(reasoning) > 0, f"Expected non-empty reasoning_content for {model}"

    @pytest.mark.parametrize("model", REASONING_MODELS)
    @pytest.mark.asyncio
    async def test_reasoning_content_async(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test that reasoning_content appears as content blocks (async)."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        result = await llm.ainvoke("What is 25 * 37? Think step by step.")

        reasoning = self._extract_reasoning(result)

        print(f"\n[{model}] Reasoning Content (async):")
        print(f"  Content: {str(result.content)[:200]}")
        print(f"  Reasoning: {reasoning[:200]}")
        print(f"  Content type: {type(result.content).__name__}")

        assert isinstance(result.content, list), (
            f"Expected list content blocks for {model}, got {type(result.content)}"
        )
        thinking_blocks = [
            b
            for b in result.content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) > 0, (
            f"Expected thinking block in content for {model}"
        )
        assert len(reasoning) > 0, f"Expected non-empty reasoning_content for {model}"

    @pytest.mark.parametrize("model", REASONING_MODELS)
    def test_reasoning_content_with_tool_calls(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test that reasoning_content is preserved when tool calls are also present."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        llm_with_tools = llm.bind_tools([get_weather])

        result = llm_with_tools.invoke("What's the weather in San Francisco?")

        print(f"\n[{model}] Reasoning + Tool Calls:")
        print(f"  Content type: {type(result.content).__name__}")
        print(f"  Content: {str(result.content)[:200]}")
        print(f"  Tool calls: {result.tool_calls}")

        assert result is not None, f"Result is None for {model}"

        # If the model made a tool call AND has reasoning, both should be present
        if result.tool_calls and isinstance(result.content, list):
            thinking_blocks = [
                b
                for b in result.content
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert len(thinking_blocks) > 0, (
                f"Expected thinking block alongside tool_calls for {model}"
            )
            assert len(thinking_blocks[0]["thinking"]) > 0, (
                f"Expected non-empty reasoning for {model}"
            )
            print(f"  Reasoning: {thinking_blocks[0]['thinking'][:200]}")
            print("  Status: PASS - reasoning preserved with tool calls")
        elif result.tool_calls:
            # Tool call made but no reasoning - content should be empty string
            print("  Status: WARN - tool call without reasoning content")
        else:
            print("  Status: WARN - no tool call made")

    def test_no_reasoning_content_for_llama(self, account_id, api_token, ai_gateway):
        """Test that Llama content is a plain string, not content blocks."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            account_id,
            api_token,
            ai_gateway,
        )
        result = llm.invoke("Say hello.")

        print("\n[llama] Reasoning Content check:")
        print(f"  Content type: {type(result.content).__name__}")

        assert isinstance(result.content, str), (
            "Llama should return plain string content, not content blocks"
        )


# MARK: - Multi-Modal Tests


def create_test_image_base64() -> str:
    """Create a minimal 1x1 red pixel PNG and return as base64.

    Uses raw PNG bytes to avoid requiring PIL in the test environment.
    """
    import struct
    import zlib

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + chunk
            + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        )

    width, height = 1, 1
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_row = b"\x00" + b"\xff\x00\x00"
    idat_data = zlib.compress(raw_row)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr_data)
    png += _png_chunk(b"IDAT", idat_data)
    png += _png_chunk(b"IEND", b"")

    return base64.standard_b64encode(png).decode("utf-8")


class TestMultiModal:
    """Test multi-modal image input across Workers AI models via REST API.

    Discovery test: Which Workers AI models accept image content blocks
    when invoked via the REST API (/v1/chat/completions)?
    """

    @pytest.mark.parametrize("model", MODELS)
    def test_image_base64(self, model, account_id, api_token, ai_gateway):
        """Test image input via base64-encoded PNG."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        image_b64 = create_test_image_base64()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe this image in one sentence. What color is it?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]
        )

        try:
            result = llm.invoke([message])
            text = get_text_content(result.content)

            print(f"\n[{model}] Multi-Modal Image (base64):")
            print("  Status: PASS")
            print(f"  Response: {text[:200]}")

            assert len(text) > 0, f"Expected non-empty response from {model}"
        except Exception as e:
            error_msg = str(e)
            print(f"\n[{model}] Multi-Modal Image (base64):")
            print("  Status: FAIL")
            print(f"  Error: {error_msg[:200]}")

            # Skip rather than fail — this is a discovery test
            pytest.skip(
                f"Model {model} does not support multi-modal: {error_msg[:100]}"
            )


# MARK: - Vision Model Regression Tests


class TestVisionModels:
    """Regression tests for confirmed vision-capable models.

    Unlike TestMultiModal (which skips failures), these tests assert that
    VISION_MODELS must successfully process image input. A failure here means
    vision support regressed for a model we know should work.
    """

    @pytest.mark.parametrize("model", VISION_MODELS)
    def test_vision_invoke(self, model, account_id, api_token, ai_gateway):
        """Vision models must return a non-empty response for image input."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        image_b64 = create_test_image_base64()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe this image in one sentence. What color is it?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        )

        result = llm.invoke([message])
        text = get_text_content(result.content)

        print(f"\n[{model}] Vision invoke:")
        print(f"  Response: {text[:200]}")

        assert len(text) > 0, f"Expected non-empty vision response from {model}"


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
            print(
                f"  Content: {get_text_content(result.content)[:200] if result.content else 'EMPTY'}"
            )
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
                f"  Content: {get_text_content(result.content)[:100] if result.content else 'empty'}..."
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


# MARK: - Session Affinity (Prompt Caching) Tests
class TestSessionAffinity:
    """Test prompt caching via x-session-affinity header.

    Currently only kimi-k2.5 is known to support prompt caching.
    """

    @pytest.mark.parametrize("model", ["@cf/moonshotai/kimi-k2.5"])
    def test_session_affinity_basic_invoke(self, model: str):
        """Verify that requests with session_id succeed and produce responses."""
        llm = ChatCloudflareWorkersAI(
            model=model,
            session_id="test-session-integration",
        )
        result = llm.invoke("Say hello in exactly 3 words.")
        text = get_text_content(result)
        assert text, f"Empty response from {model} with session_id"

    @pytest.mark.parametrize("model", ["@cf/moonshotai/kimi-k2.5"])
    def test_session_affinity_cached_tokens(self, model: str):
        """Two calls with same session_id should succeed; cache hits are best-effort.

        Prompt caching depends on Cloudflare routing both requests to the same
        machine, which session affinity makes *likely* but not guaranteed.
        We verify the plumbing works and log cache metrics without asserting
        on cached_tokens, since it's infrastructure-dependent.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Long system prompt to ensure enough tokens to cache
        system_prompt = "You are an expert assistant. " * 50
        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Say hi in 3 words."),
        ]

        session = f"test-cache-{uuid.uuid4().hex[:8]}"
        llm = ChatCloudflareWorkersAI(model=model, session_id=session)

        # First call primes the cache
        r1 = llm.invoke(msgs)
        assert get_text_content(r1), "First call produced empty response"
        usage1 = r1.response_metadata.get("token_usage", {})
        cached1 = usage1.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"  Call 1 cached_tokens: {cached1}")

        # Second call may hit the cache (best-effort)
        r2 = llm.invoke(msgs)
        assert get_text_content(r2), "Second call produced empty response"
        usage2 = r2.response_metadata.get("token_usage", {})
        cached2 = usage2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"  Call 2 cached_tokens: {cached2}")
        # Cache hits are best-effort; log but don't assert
        if cached2 > 0:
            print(f"  Cache HIT: {cached2} tokens cached")
        else:
            print("  Cache MISS: caching is best-effort, not guaranteed")

    @pytest.mark.parametrize("model", ["@cf/moonshotai/kimi-k2.5"])
    @pytest.mark.asyncio
    async def test_session_affinity_cached_tokens_async(self, model: str):
        """Async variant: two calls with same session_id should succeed."""
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = "You are an expert assistant. " * 50
        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Say hi in 3 words."),
        ]

        session = f"test-cache-async-{uuid.uuid4().hex[:8]}"
        llm = ChatCloudflareWorkersAI(model=model, session_id=session)

        r1 = await llm.ainvoke(msgs)
        assert get_text_content(r1), "First async call produced empty response"

        r2 = await llm.ainvoke(msgs)
        assert get_text_content(r2), "Second async call produced empty response"
        usage2 = r2.response_metadata.get("token_usage", {})
        cached2 = usage2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"  Async call 2 cached_tokens: {cached2}")
        if cached2 > 0:
            print(f"  Async cache HIT: {cached2} tokens cached")
        else:
            print("  Async cache MISS: caching is best-effort, not guaranteed")


# MARK: - AI Gateway Request Handling Tests
@pytest.mark.skipif(
    not os.environ.get("AI_GATEWAY"),
    reason="AI_GATEWAY env var not set",
)
class TestAIGatewayHeaders:
    """Test AI Gateway timeout and retry headers.

    Requires AI_GATEWAY environment variable to be set.
    """

    def test_aig_timeout_invoke(self):
        """Request with AI Gateway timeout header should succeed."""
        llm = ChatCloudflareWorkersAI(
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ai_gateway=os.environ["AI_GATEWAY"],
            aig_request_timeout=30000,
        )
        result = llm.invoke("Say hello in one word.")
        text = get_text_content(result)
        assert text, "Empty response with aig_request_timeout"

    def test_aig_retries_invoke(self):
        """Request with AI Gateway retry headers should succeed."""
        llm = ChatCloudflareWorkersAI(
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ai_gateway=os.environ["AI_GATEWAY"],
            aig_max_attempts=2,
            aig_retry_delay=1000,
            aig_backoff="exponential",
        )
        result = llm.invoke("Say hello in one word.")
        text = get_text_content(result)
        assert text, "Empty response with aig retry headers"

    def test_aig_timeout_with_session_id(self):
        """AI Gateway headers and session_id should work together."""
        llm = ChatCloudflareWorkersAI(
            model="@cf/moonshotai/kimi-k2.5",
            ai_gateway=os.environ["AI_GATEWAY"],
            session_id="test-aig-session",
            aig_request_timeout=30000,
        )
        result = llm.invoke("Say hello.")
        text = get_text_content(result)
        assert text, "Empty response with combined headers"
