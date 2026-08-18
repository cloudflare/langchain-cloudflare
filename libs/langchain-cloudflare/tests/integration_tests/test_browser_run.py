# ruff: noqa: T201
"""Integration tests for CloudflareBrowserRunLoader and CloudflareBrowserRunTool.

Tests cover:
- Loader: markdown, content, scrape, crawl modes (sync + async)
- Tool: markdown, links, json, screenshot, snapshot, accessibility_tree modes
  (sync + async)

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

from langchain_cloudflare.loaders import (
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
            urls=[TEST_URL, TEST_URL],
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

    def test_crawl_with_options_passthrough(self, account_id, api_token):
        """Crawl with crawl_options passthrough (e.g. source)."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="crawl",
            crawl_limit=2,
            crawl_depth=1,
            crawl_timeout=60.0,
            crawl_options={"source": "sitemaps"},
            account_id=account_id,
            api_token=api_token,
        )
        docs = loader.load()

        print("\n[Loader] Crawl with crawl_options:")
        print(f"  Docs count: {len(docs)}")

        assert len(docs) >= 1

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
        """JSON tool extracts structured data using a JSON schema.

        The nested key is ``json_schema`` (matching the API's OpenAI-style
        ``response_format`` shape), not ``schema`` -- confirmed live: the
        latter returns a 400 with "response_format.json_schema is required
        and must be non-empty when response_format.type is 'json_schema'".
        """
        tool = CloudflareBrowserRunTool(
            mode="json",
            json_response_format={
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "links": {"type": "array", "items": {"type": "string"}},
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

    def test_pdf_returns_base64(self, account_id, api_token):
        """PDF tool returns base64-encoded PDF data."""
        import base64

        tool = CloudflareBrowserRunTool(
            mode="pdf",
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] PDF:")
        print(f"  Result length: {len(result)} chars")
        print(f"  Starts with: {result[:20]}...")

        assert isinstance(result, str)
        assert len(result) > 100
        assert base64.b64decode(result).startswith(b"%PDF-")

    def test_snapshot_combines_formats(self, account_id, api_token):
        """Snapshot tool returns multiple requested formats in one call."""
        tool = CloudflareBrowserRunTool(
            mode="snapshot",
            snapshot_formats=["markdown", "screenshot"],
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] Snapshot:")
        print(f"  Result preview: {result[:300]}")

        assert isinstance(result, str)
        assert "markdown" in result.lower() or "screenshot" in result.lower()

    def test_accessibility_tree(self, account_id, api_token):
        """Accessibility tree tool returns a structured tree as JSON."""
        tool = CloudflareBrowserRunTool(
            mode="accessibility_tree",
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] Accessibility tree:")
        print(f"  Result preview: {result[:300]}")

        assert isinstance(result, str)
        assert len(result) > 2

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


# MARK: - Kitesurf Tests


class TestBrowserRunKitesurf:
    """Integration tests for the browser="kitesurf" alternate runtime (REST-only)."""

    def test_tool_markdown_with_kitesurf(self, account_id, api_token):
        """Markdown tool works with browser='kitesurf'."""
        tool = CloudflareBrowserRunTool(
            mode="markdown",
            browser="kitesurf",
            account_id=account_id,
            api_token=api_token,
        )
        result = tool.invoke({"url": TEST_URL})

        print("\n[Tool] Markdown with kitesurf:")
        print(f"  Result preview: {result[:200]}")

        assert isinstance(result, str)
        assert "Example Domain" in result

    def test_loader_crawl_with_kitesurf(self, account_id, api_token):
        """Crawl loader works with browser='kitesurf' (start-of-job option only)."""
        loader = CloudflareBrowserRunLoader(
            urls=[TEST_URL],
            mode="crawl",
            crawl_limit=1,
            crawl_timeout=60.0,
            browser="kitesurf",
            account_id=account_id,
            api_token=api_token,
        )
        docs = loader.load()

        print("\n[Loader] Crawl with kitesurf:")
        print(f"  Docs count: {len(docs)}")

        assert len(docs) >= 1
