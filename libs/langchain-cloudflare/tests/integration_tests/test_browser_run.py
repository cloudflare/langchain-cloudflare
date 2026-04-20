# ruff: noqa: T201
"""Integration tests for CloudflareBrowserRunLoader and CloudflareBrowserRunTool.

Tests cover:
- Loader: markdown, content, scrape, crawl modes (sync + async)
- Tool: markdown, links, json, screenshot modes (sync + async)

Required environment variables:
    CF_ACCOUNT_ID: Cloudflare account ID
    CF_API_TOKEN: API token with Browser Rendering – Edit permission

Usage:
    # Set environment variables
    export CF_ACCOUNT_ID="your_account_id"
    export CF_API_TOKEN="your_api_token"

    # Run with pytest
    python -m pytest tests/integration_tests/test_browser_run.py -v -s

    # Or via Makefile
    make integration_tests TEST_FILE=tests/integration_tests/test_browser_run.py
"""

import os

import pytest

from langchain_cloudflare.browser_run import (
    CloudflareBrowserRunLoader,
    CloudflareBrowserRunTool,
)

# A lightweight, stable public URL for testing
TEST_URL = "https://example.com"


# MARK: - Fixtures


@pytest.fixture
def account_id():
    """Return the Cloudflare account ID or skip."""
    val = os.environ.get("CF_ACCOUNT_ID", "")
    if not val:
        pytest.skip("CF_ACCOUNT_ID not set")
    return val


@pytest.fixture
def api_token():
    """Return the Cloudflare API token or skip."""
    val = (
        os.environ.get("TEST_CF_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
        or os.environ.get("CF_AI_API_TOKEN")
        or ""
    )
    if not val:
        pytest.skip("CF_API_TOKEN not set")
    return val


# MARK: - Loader Integration Tests


class TestBrowserRunLoader:
    """Integration tests for CloudflareBrowserRunLoader."""

    def test_markdown_single_url(self, account_id, api_token):
        """Load a single URL in markdown mode."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="markdown",
            account_id=account_id,
            api_token=api_token,
        )
        docs = loader.load()

        print("\n[Loader] Markdown single URL:")
        print(f"  Docs count: {len(docs)}")
        print(f"  Content preview: {docs[0].page_content[:200]}")
        print(f"  Metadata: {docs[0].metadata}")

        assert len(docs) == 1
        assert "Example Domain" in docs[0].page_content
        assert docs[0].metadata["source"] == TEST_URL
        assert docs[0].metadata["mode"] == "markdown"

    def test_markdown_multiple_urls(self, account_id, api_token):
        """Load multiple URLs in markdown mode."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL, "https://httpbin.org/html"],
            mode="markdown",
            account_id=account_id,
            api_token=api_token,
        )
        docs = loader.load()

        print("\n[Loader] Markdown multiple URLs:")
        print(f"  Docs count: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"  Doc {i} source: {doc.metadata['source']}")
            print(f"  Doc {i} preview: {doc.page_content[:100]}")

        assert len(docs) == 2

    def test_content_mode(self, account_id, api_token):
        """Load a URL in content (raw HTML) mode."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="content",
            account_id=account_id,
            api_token=api_token,
        )
        docs = loader.load()

        print("\n[Loader] Content mode:")
        print(f"  Docs count: {len(docs)}")
        print(f"  Content preview: {docs[0].page_content[:200]}")

        assert len(docs) == 1
        assert "<html" in docs[0].page_content.lower()
        assert docs[0].metadata["mode"] == "content"

    def test_scrape_with_selectors(self, account_id, api_token):
        """Scrape specific elements from a page."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="scrape",
            elements=[{"selector": "h1"}, {"selector": "a"}],
            account_id=account_id,
            api_token=api_token,
        )
        docs = loader.load()

        print("\n[Loader] Scrape with selectors:")
        print(f"  Docs count: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"  Doc {i} selector: {doc.metadata.get('selector')}")
            print(f"  Doc {i} content: {doc.page_content[:100]}")

        assert len(docs) >= 1
        selectors = [d.metadata.get("selector") for d in docs]
        assert "h1" in selectors

    def test_crawl_basic(self, account_id, api_token):
        """Crawl a small site."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="crawl",
            crawl_limit=3,
            crawl_depth=1,
            crawl_timeout=60.0,
            account_id=account_id,
            api_token=api_token,
        )
        docs = loader.load()

        print("\n[Loader] Crawl basic:")
        print(f"  Docs count: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"  Doc {i} source: {doc.metadata.get('source')}")
            print(f"  Doc {i} title: {doc.metadata.get('title', 'N/A')}")

        assert len(docs) >= 1
        assert docs[0].metadata["mode"] == "crawl"

    def test_lazy_load(self, account_id, api_token):
        """lazy_load yields documents one at a time."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="markdown",
            account_id=account_id,
            api_token=api_token,
        )
        docs = list(loader.lazy_load())

        print("\n[Loader] Lazy load:")
        print(f"  Docs count: {len(docs)}")

        assert len(docs) == 1
        assert "Example Domain" in docs[0].page_content

    @pytest.mark.asyncio
    async def test_async_markdown(self, account_id, api_token):
        """Async load a single URL in markdown mode."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="markdown",
            account_id=account_id,
            api_token=api_token,
        )
        docs = await loader.aload()

        print("\n[Loader] Async markdown:")
        print(f"  Docs count: {len(docs)}")
        print(f"  Content preview: {docs[0].page_content[:200]}")

        assert len(docs) == 1
        assert "Example Domain" in docs[0].page_content

    @pytest.mark.asyncio
    async def test_async_crawl(self, account_id, api_token):
        """Async crawl a small site."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="crawl",
            crawl_limit=3,
            crawl_depth=1,
            crawl_timeout=60.0,
            account_id=account_id,
            api_token=api_token,
        )
        docs = await loader.aload()

        print("\n[Loader] Async crawl:")
        print(f"  Docs count: {len(docs)}")

        assert len(docs) >= 1


# MARK: - Tool Integration Tests


class TestBrowserRunTool:
    """Integration tests for CloudflareBrowserRunTool."""

    def test_markdown_tool(self, account_id, api_token):
        """Markdown tool returns page content."""
        tool = CloudflareBrowserRunTool(
            mode="markdown",
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] Markdown:")
        print(f"  Result type: {type(result)}")
        print(f"  Result preview: {result[:200]}")

        assert isinstance(result, str)
        assert "Example Domain" in result

    def test_links_tool(self, account_id, api_token):
        """Links tool returns discovered URLs."""
        tool = CloudflareBrowserRunTool(
            mode="links",
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] Links:")
        print(f"  Result: {result}")

        assert isinstance(result, str)
        assert "iana.org" in result

    def test_json_extraction_with_prompt(self, account_id, api_token):
        """JSON tool extracts structured data using an AI prompt."""
        tool = CloudflareBrowserRunTool(
            mode="json",
            json_prompt="Extract the page title and any links on the page.",
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] JSON with prompt:")
        print(f"  Result: {result[:500]}")

        assert isinstance(result, str)
        assert len(result) > 10

    def test_json_extraction_with_schema(self, account_id, api_token):
        """JSON tool extracts structured data using a JSON schema."""
        tool = CloudflareBrowserRunTool(
            mode="json",
            json_response_format={
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "links": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] JSON with schema:")
        print(f"  Result: {result[:500]}")

        assert isinstance(result, str)
        assert "title" in result.lower() or "links" in result.lower()

    def test_screenshot_returns_base64(self, account_id, api_token):
        """Screenshot tool returns base64-encoded image data."""
        tool = CloudflareBrowserRunTool(
            mode="screenshot",
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] Screenshot:")
        print(f"  Result length: {len(result)} chars")
        print(f"  Starts with: {result[:20]}...")

        assert isinstance(result, str)
        assert len(result) > 100

    @pytest.mark.asyncio
    async def test_async_markdown_tool(self, account_id, api_token):
        """Async markdown tool returns page content."""
        tool = CloudflareBrowserRunTool(
            mode="markdown",
            account_id=account_id,
            api_token=api_token,
        )
        result = await tool.ainvoke({"url": TEST_URL})

        print("\n[Tool] Async markdown:")
        print(f"  Result preview: {result[:200]}")

        assert isinstance(result, str)
        assert "Example Domain" in result

    @pytest.mark.asyncio
    async def test_async_links_tool(self, account_id, api_token):
        """Async links tool returns discovered URLs."""
        tool = CloudflareBrowserRunTool(
            mode="links",
            account_id=account_id,
            api_token=api_token,
        )
        result = await tool.ainvoke({"url": TEST_URL})

        print("\n[Tool] Async links:")
        print(f"  Result: {result}")

        assert isinstance(result, str)
        assert "iana.org" in result


# MARK: - LangGraph Integration Tests


class TestBrowserRunLangGraph:
    """Integration tests verifying Browser Run works in LangGraph patterns."""

    def test_loader_as_custom_node(self, account_id, api_token):
        """Loader works as a custom node in a LangGraph StateGraph."""
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class ResearchState(TypedDict):
            url: str
            page_content: str

        def fetch_page(state: ResearchState) -> dict:
            loader = CloudflareBrowserRunLoader(
                urls=[state["url"]],
                mode="markdown",
                account_id=account_id,
                api_token=api_token,
            )
            docs = loader.load()
            return {"page_content": docs[0].page_content}

        graph = StateGraph(ResearchState)
        graph.add_node("fetch_page", fetch_page)
        graph.add_edge(START, "fetch_page")
        graph.add_edge("fetch_page", END)
        app = graph.compile()

        result = app.invoke({"url": TEST_URL, "page_content": ""})

        print("\n[LangGraph] Loader as custom node:")
        print(f"  Content: {len(result['page_content'])} chars")

        assert "Example Domain" in result["page_content"]

    def test_tool_in_toolnode(self, account_id, api_token):
        """Tools work inside LangGraph ToolNode with simulated tool calls."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition

        md_tool = CloudflareBrowserRunTool(
            mode="markdown",
            account_id=account_id,
            api_token=api_token,
        )
        tool_node = ToolNode([md_tool])

        def fake_model(state: MessagesState) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "cloudflare_browser_run_markdown",
                                "args": {"url": TEST_URL},
                                "id": "call_test_001",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }

        graph = StateGraph(MessagesState)
        graph.add_node("model", fake_model)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", tools_condition)
        graph.add_edge("tools", END)
        app = graph.compile()

        result = app.invoke({"messages": [HumanMessage(content="test")]})
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]

        print("\n[LangGraph] Tool in ToolNode:")
        print(f"  Tool messages: {len(tool_msgs)}")
        print(f"  Content: {tool_msgs[0].content[:100]}")

        assert len(tool_msgs) == 1
        assert "Example Domain" in tool_msgs[0].content

    def test_parallel_tool_calls(self, account_id, api_token):
        """Multiple tools execute in parallel via ToolNode."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition

        tools = [
            CloudflareBrowserRunTool(
                mode="markdown",
                account_id=account_id,
                api_token=api_token,
            ),
            CloudflareBrowserRunTool(
                mode="links",
                account_id=account_id,
                api_token=api_token,
            ),
        ]
        tool_node = ToolNode(tools)

        def fake_model(state: MessagesState) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "cloudflare_browser_run_markdown",
                                "args": {"url": TEST_URL},
                                "id": "call_p1",
                                "type": "tool_call",
                            },
                            {
                                "name": "cloudflare_browser_run_links",
                                "args": {"url": TEST_URL},
                                "id": "call_p2",
                                "type": "tool_call",
                            },
                        ],
                    )
                ]
            }

        graph = StateGraph(MessagesState)
        graph.add_node("model", fake_model)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", tools_condition)
        graph.add_edge("tools", END)
        app = graph.compile()

        result = app.invoke({"messages": [HumanMessage(content="test")]})
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]

        print("\n[LangGraph] Parallel tool calls:")
        for msg in tool_msgs:
            print(f"  - {msg.name}: {len(msg.content)} chars")

        assert len(tool_msgs) == 2
        names = {m.name for m in tool_msgs}
        assert "cloudflare_browser_run_markdown" in names
        assert "cloudflare_browser_run_links" in names

    def test_parallel_nodes_with_loader(self, account_id, api_token):
        """Loader and Tool run as parallel nodes in a DAG."""
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class ParallelState(TypedDict):
            url: str
            page_content: str
            links: list

        def fetch_page(state: ParallelState) -> dict:
            loader = CloudflareBrowserRunLoader(
                urls=[state["url"]],
                mode="markdown",
                account_id=account_id,
                api_token=api_token,
            )
            docs = loader.load()
            return {"page_content": docs[0].page_content}

        def extract_links(state: ParallelState) -> dict:
            tool = CloudflareBrowserRunTool(
                mode="links",
                account_id=account_id,
                api_token=api_token,
            )
            links = tool.invoke({"url": state["url"]}).strip().split("\n")
            return {"links": links}

        graph = StateGraph(ParallelState)
        graph.add_node("fetch_page", fetch_page)
        graph.add_node("extract_links", extract_links)
        graph.add_edge(START, "fetch_page")
        graph.add_edge(START, "extract_links")
        graph.add_edge("fetch_page", END)
        graph.add_edge("extract_links", END)
        app = graph.compile()

        result = app.invoke({"url": TEST_URL, "page_content": "", "links": []})

        print("\n[LangGraph] Parallel nodes (Loader + Tool):")
        print(f"  Content: {len(result['page_content'])} chars")
        print(f"  Links: {result['links']}")

        assert "Example Domain" in result["page_content"]
        assert len(result["links"]) >= 1
