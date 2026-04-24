"""Cloudflare Browser Run integration for LangChain.

This module provides a document loader and an agent tool backed by
Cloudflare Browser Run (formerly Browser Rendering).  Browser Run
offers serverless headless Chrome on Cloudflare's global network
via a simple REST API, supporting markdown extraction, crawling,
scraping, AI-powered structured data extraction, screenshots, PDFs,
and link discovery.

    * ``CloudflareBrowserRunLoader`` – a ``BaseLoader`` for document
      ingestion (RAG pipelines, knowledge-base construction).
    * ``CloudflareBrowserRunTool`` – a ``BaseTool`` for LangGraph
      agent workflows (research agents, data extraction, live web
      context).

Note:
    Browser Run Quick Actions are REST-only.  Unlike the other modules
    in this library there is no Workers binding path.
"""

# MARK: - Imports
from __future__ import annotations

import base64
import logging
import time
import warnings
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional

import requests
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from ._errors import TokenErrors

logger = logging.getLogger(__name__)

# MARK: - Constants
BROWSER_RUN_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_CRAWL_POLL_INTERVAL = 2.0  # seconds between /crawl status checks
DEFAULT_CRAWL_TIMEOUT = 300.0  # max seconds to wait for a crawl job
DEFAULT_CRAWL_LIMIT = 10
DEFAULT_CRAWL_DEPTH = 2
DEFAULT_REQUEST_TIMEOUT = 60.0  # seconds for individual HTTP requests


# MARK: - Helpers


def _build_browser_run_url(account_id: str, endpoint: str) -> str:
    """Build the full Browser Run REST API URL for a given endpoint.

    Args:
        account_id: Cloudflare account ID.
        endpoint: The Quick Action path, e.g. ``"markdown"`` or ``"crawl"``.

    Returns:
        Fully-qualified URL string.
    """
    return f"{BROWSER_RUN_BASE_URL}/{account_id}/browser-rendering/{endpoint}"


def _build_headers(api_token: str) -> Dict[str, str]:
    """Build authorization headers for Browser Run requests.

    Args:
        api_token: The plaintext API token value.

    Returns:
        Dict with ``Authorization`` and ``Content-Type`` headers.
    """
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }


def _build_shared_options(
    goto_options: Optional[Dict[str, Any]] = None,
    viewport: Optional[Dict[str, Any]] = None,
    wait_for_selector: Optional[Dict[str, Any]] = None,
    cookies: Optional[List[Dict[str, Any]]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    reject_resource_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the shared optional parameters accepted by most Browser Run endpoints.

    Args:
        goto_options: Page navigation options (waitUntil, timeout).
        viewport: Viewport dimensions ``{width, height}``.
        wait_for_selector: Wait for a CSS selector before returning.
        cookies: List of cookie dicts to set before navigation.
        extra_headers: Extra HTTP headers to set on the request.
        reject_resource_types: Resource types to block (e.g. ``["image"]``).

    Returns:
        Dict of non-None options ready to merge into the request body.
    """
    opts: Dict[str, Any] = {}
    if goto_options is not None:
        opts["gotoOptions"] = goto_options
    if viewport is not None:
        opts["viewport"] = viewport
    if wait_for_selector is not None:
        opts["waitForSelector"] = wait_for_selector
    if cookies is not None:
        opts["cookies"] = cookies
    if extra_headers is not None:
        opts["setExtraHTTPHeaders"] = extra_headers
    if reject_resource_types is not None:
        opts["rejectResourceTypes"] = reject_resource_types
    return opts


def _check_api_response(data: Any) -> None:
    """Raise if the Cloudflare API returned a success=false envelope.

    Some Browser Run endpoints return ``200 OK`` with an error body
    like ``{"success": false, "errors": [...]}``.  This helper ensures
    such responses are surfaced as exceptions rather than silently
    producing empty results.

    Args:
        data: The parsed JSON response body.

    Raises:
        RuntimeError: When the API indicates failure.
    """
    if isinstance(data, dict) and not data.get("success", True):
        errors = data.get("errors", [])
        raise RuntimeError(f"Browser Run API error: {errors}")


# MARK: - CloudflareBrowserRunLoader


class CloudflareBrowserRunLoader(BaseLoader, BaseModel):  # type: ignore[misc]
    """Load documents from web pages using Cloudflare Browser Run.

    Browser Run renders JavaScript-heavy pages on Cloudflare's global
    network and returns the content via a simple REST API.  This loader
    converts web pages into LangChain ``Document`` objects suitable for
    RAG pipelines and knowledge-base construction.

    Example (markdown mode):
        .. code-block:: python

            from langchain_cloudflare import CloudflareBrowserRunLoader

            loader = CloudflareBrowserRunLoader(
                urls=["https://developers.cloudflare.com/workers-ai/"],
                mode="markdown",
            )
            docs = loader.load()

    Example (crawl mode):
        .. code-block:: python

            loader = CloudflareBrowserRunLoader(
                urls=["https://developers.cloudflare.com/cloudflare-one/"],
                mode="crawl",
                crawl_limit=50,
                crawl_depth=2,
            )
            docs = loader.load()

    Key init args:
        urls: list[str]
            URLs to load.

        mode: str
            One of ``"markdown"``, ``"crawl"``, ``"scrape"``, ``"content"``.

        account_id: str
            Cloudflare account ID.  Falls back to ``CF_ACCOUNT_ID`` env var.

        api_token: str
            Cloudflare API token with *Browser Rendering – Edit* permission.
            Falls back to ``CF_API_TOKEN`` then ``CF_AI_API_TOKEN`` env var.
    """

    # MARK: - Fields
    urls: List[str] = Field(default_factory=list)
    """URLs to load."""

    mode: Literal["markdown", "crawl", "scrape", "content"] = "markdown"
    """Loader mode: ``markdown``, ``crawl``, ``scrape``, or ``content``."""

    account_id: str = Field(default_factory=from_env("CF_ACCOUNT_ID", default=""))
    """Cloudflare account ID."""

    api_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env(
            ["CF_API_TOKEN", "CF_AI_API_TOKEN"], default=None
        )
    )
    """API token with Browser Rendering – Edit permission."""

    # Crawl-specific
    crawl_limit: int = DEFAULT_CRAWL_LIMIT
    """Maximum number of pages to crawl (``/crawl`` mode only)."""

    crawl_depth: int = DEFAULT_CRAWL_DEPTH
    """Maximum link depth from seed URL (``/crawl`` mode only)."""

    crawl_poll_interval: float = DEFAULT_CRAWL_POLL_INTERVAL
    """Seconds between ``/crawl`` status polls."""

    crawl_timeout: float = DEFAULT_CRAWL_TIMEOUT
    """Maximum seconds to wait for a crawl job to finish."""

    # Scrape-specific
    elements: Optional[List[Dict[str, Any]]] = None
    """CSS selectors for ``/scrape`` mode, e.g. ``[{"selector": "h1"}]``."""

    # Shared Browser Run options
    goto_options: Optional[Dict[str, Any]] = None
    """Page navigation options (``waitUntil``, ``timeout``)."""

    viewport: Optional[Dict[str, Any]] = None
    """Viewport dimensions ``{width, height}``."""

    wait_for_selector: Optional[Dict[str, Any]] = None
    """Wait for a CSS selector before returning content."""

    cookies: Optional[List[Dict[str, Any]]] = None
    """Cookies to set before navigation."""

    extra_headers: Optional[Dict[str, str]] = None
    """Extra HTTP headers sent with the browser request."""

    reject_resource_types: Optional[List[str]] = None
    """Resource types to block (e.g. ``["image", "stylesheet"]``)."""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    """Timeout in seconds for individual HTTP requests."""

    # Internal
    _headers: Dict[str, str] = PrivateAttr()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the loader and validate credentials."""
        super().__init__(**kwargs)

        if not self.account_id:
            raise ValueError(TokenErrors.NO_ACCOUNT_ID_SET)
        if not self.api_token or not self.api_token.get_secret_value():
            raise ValueError(TokenErrors.INSUFFICIENT_BROWSER_RUN_TOKEN)

        self._headers = _build_headers(self.api_token.get_secret_value())

    # MARK: - Private helpers

    def _shared_body(self) -> Dict[str, Any]:
        """Return the shared optional params for the request body."""
        return _build_shared_options(
            goto_options=self.goto_options,
            viewport=self.viewport,
            wait_for_selector=self.wait_for_selector,
            cookies=self.cookies,
            extra_headers=self.extra_headers,
            reject_resource_types=self.reject_resource_types,
        )

    def _fetch_markdown(self, url: str) -> Document:
        """Fetch a URL via the ``/markdown`` endpoint.

        Args:
            url: The URL to render and convert to markdown.

        Returns:
            A single ``Document`` with markdown content.
        """
        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        resp = requests.post(
            _build_browser_run_url(self.account_id, "markdown"),
            headers=self._headers,
            json=body,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        _check_api_response(data)
        content = data.get("result", "")
        return Document(
            page_content=content,
            metadata={"source": url, "mode": "markdown"},
        )

    def _fetch_content(self, url: str) -> Document:
        """Fetch a URL via the ``/content`` endpoint (raw rendered HTML).

        Args:
            url: The URL to render.

        Returns:
            A single ``Document`` with HTML content.
        """
        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        resp = requests.post(
            _build_browser_run_url(self.account_id, "content"),
            headers=self._headers,
            json=body,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        _check_api_response(data)
        content = data.get("result", resp.text)
        return Document(
            page_content=content,
            metadata={"source": url, "mode": "content"},
        )

    def _fetch_scrape(self, url: str) -> List[Document]:
        """Fetch a URL via the ``/scrape`` endpoint.

        Args:
            url: The URL to scrape.

        Returns:
            List of ``Document`` objects, one per matched element group.
        """
        elements = self.elements or [{"selector": "body"}]
        body: Dict[str, Any] = {
            "url": url,
            "elements": elements,
            **self._shared_body(),
        }
        resp = requests.post(
            _build_browser_run_url(self.account_id, "scrape"),
            headers=self._headers,
            json=body,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        _check_api_response(data)

        docs: List[Document] = []
        for group in data.get("result", []):
            selector = group.get("selector", "")
            texts = [r.get("text", "") for r in group.get("results", [])]
            combined = "\n".join(t for t in texts if t)
            if combined:
                docs.append(
                    Document(
                        page_content=combined,
                        metadata={
                            "source": url,
                            "mode": "scrape",
                            "selector": selector,
                        },
                    )
                )
        return docs

    def _fetch_crawl(self, url: str) -> List[Document]:
        """Crawl a URL via the ``/crawl`` async endpoint.

        Initiates a crawl job, polls until complete, and returns all
        crawled pages as ``Document`` objects.

        Args:
            url: The seed URL to crawl.

        Returns:
            List of ``Document`` objects, one per crawled page.
        """
        crawl_url = _build_browser_run_url(self.account_id, "crawl")

        # Step 1: Initiate crawl
        body: Dict[str, Any] = {
            "url": url,
            "limit": self.crawl_limit,
            "depth": self.crawl_depth,
            "formats": ["markdown"],
            **self._shared_body(),
        }
        resp = requests.post(
            crawl_url, headers=self._headers, json=body, timeout=self.request_timeout
        )
        resp.raise_for_status()
        job_id = resp.json().get("result", "")

        if not job_id:
            return []

        # Step 2: Poll for results
        results_url = f"{crawl_url}/{job_id}"
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.crawl_timeout:
                warnings.warn(
                    f"Crawl for {url} timed out after {self.crawl_timeout}s. "
                    "Returning partial results.",
                    stacklevel=2,
                )
                break

            poll = requests.get(
                results_url, headers=self._headers, timeout=self.request_timeout
            )
            poll.raise_for_status()
            poll_data = poll.json().get("result", {})
            status = poll_data.get("status", "")

            if status in (
                "completed",
                "errored",
                "cancelled_by_user",
                "cancelled_due_to_timeout",
                "cancelled_due_to_limits",
            ):
                break

            time.sleep(self.crawl_poll_interval)

        # Step 3: Collect records (may need pagination)
        docs: List[Document] = []
        cursor: Optional[int] = None

        while True:
            params: Dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor

            page_resp = requests.get(
                results_url,
                headers=self._headers,
                params=params,
                timeout=self.request_timeout,
            )
            page_resp.raise_for_status()
            page_data = page_resp.json().get("result", {})

            for record in page_data.get("records", []):
                if record.get("status") != "completed":
                    continue
                content = record.get("markdown", record.get("html", ""))
                meta = record.get("metadata", {})
                if content:
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": record.get("url", url),
                                "mode": "crawl",
                                "title": meta.get("title", ""),
                                "status_code": meta.get("status", 0),
                            },
                        )
                    )

            next_cursor = page_data.get("cursor")
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor

        return docs

    # MARK: - Async private helpers

    async def _afetch_markdown(self, url: str) -> Document:
        """Async variant of ``_fetch_markdown``.

        Args:
            url: The URL to render and convert to markdown.

        Returns:
            A single ``Document`` with markdown content.
        """
        import httpx

        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                _build_browser_run_url(self.account_id, "markdown"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        _check_api_response(data)
        content = data.get("result", "")
        return Document(
            page_content=content,
            metadata={"source": url, "mode": "markdown"},
        )

    async def _afetch_content(self, url: str) -> Document:
        """Async variant of ``_fetch_content``.

        Args:
            url: The URL to render.

        Returns:
            A single ``Document`` with HTML content.
        """
        import httpx

        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                _build_browser_run_url(self.account_id, "content"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        _check_api_response(data)
        return Document(
            page_content=data.get("result", resp.text),
            metadata={"source": url, "mode": "content"},
        )

    async def _afetch_scrape(self, url: str) -> List[Document]:
        """Async variant of ``_fetch_scrape``.

        Args:
            url: The URL to scrape.

        Returns:
            List of ``Document`` objects.
        """
        import httpx

        elements = self.elements or [{"selector": "body"}]
        body: Dict[str, Any] = {
            "url": url,
            "elements": elements,
            **self._shared_body(),
        }
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                _build_browser_run_url(self.account_id, "scrape"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        _check_api_response(data)
        docs: List[Document] = []
        for group in data.get("result", []):
            selector = group.get("selector", "")
            texts = [r.get("text", "") for r in group.get("results", [])]
            combined = "\n".join(t for t in texts if t)
            if combined:
                docs.append(
                    Document(
                        page_content=combined,
                        metadata={
                            "source": url,
                            "mode": "scrape",
                            "selector": selector,
                        },
                    )
                )
        return docs

    async def _afetch_crawl(self, url: str) -> List[Document]:
        """Async variant of ``_fetch_crawl``.

        Args:
            url: The seed URL to crawl.

        Returns:
            List of ``Document`` objects.
        """
        import asyncio

        import httpx

        crawl_url = _build_browser_run_url(self.account_id, "crawl")

        body: Dict[str, Any] = {
            "url": url,
            "limit": self.crawl_limit,
            "depth": self.crawl_depth,
            "formats": ["markdown"],
            **self._shared_body(),
        }

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(crawl_url, headers=self._headers, json=body)
            resp.raise_for_status()
            job_id = resp.json().get("result", "")

            if not job_id:
                return []

            results_url = f"{crawl_url}/{job_id}"
            start_time = time.time()

            while True:
                elapsed = time.time() - start_time
                if elapsed > self.crawl_timeout:
                    warnings.warn(
                        f"Crawl for {url} timed out after {self.crawl_timeout}s. "
                        "Returning partial results.",
                        stacklevel=2,
                    )
                    break

                poll = await client.get(results_url, headers=self._headers)
                poll.raise_for_status()
                poll_data = poll.json().get("result", {})
                status = poll_data.get("status", "")

                if status in (
                    "completed",
                    "errored",
                    "cancelled_by_user",
                    "cancelled_due_to_timeout",
                    "cancelled_due_to_limits",
                ):
                    break

                await asyncio.sleep(self.crawl_poll_interval)

            # Collect records
            docs: List[Document] = []
            cursor: Optional[int] = None

            while True:
                params: Dict[str, Any] = {}
                if cursor is not None:
                    params["cursor"] = cursor

                page_resp = await client.get(
                    results_url, headers=self._headers, params=params
                )
                page_resp.raise_for_status()
                page_data = page_resp.json().get("result", {})

                for record in page_data.get("records", []):
                    if record.get("status") != "completed":
                        continue
                    content = record.get("markdown", record.get("html", ""))
                    meta = record.get("metadata", {})
                    if content:
                        docs.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": record.get("url", url),
                                    "mode": "crawl",
                                    "title": meta.get("title", ""),
                                    "status_code": meta.get("status", 0),
                                },
                            )
                        )

                next_cursor = page_data.get("cursor")
                if next_cursor is None or next_cursor == cursor:
                    break
                cursor = next_cursor

        return docs

    # MARK: - Public API

    def load(self) -> List[Document]:
        """Load all URLs and return a list of ``Document`` objects.

        Returns:
            List of ``Document`` objects, one per page (or more for scrape/crawl).
        """
        docs: List[Document] = []
        for url in self.urls:
            if self.mode == "markdown":
                docs.append(self._fetch_markdown(url))
            elif self.mode == "content":
                docs.append(self._fetch_content(url))
            elif self.mode == "scrape":
                docs.extend(self._fetch_scrape(url))
            elif self.mode == "crawl":
                docs.extend(self._fetch_crawl(url))
        return docs

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load URLs, yielding one ``Document`` at a time.

        Yields:
            ``Document`` objects.
        """
        for url in self.urls:
            if self.mode == "markdown":
                yield self._fetch_markdown(url)
            elif self.mode == "content":
                yield self._fetch_content(url)
            elif self.mode == "scrape":
                yield from self._fetch_scrape(url)
            elif self.mode == "crawl":
                yield from self._fetch_crawl(url)

    async def aload(self) -> List[Document]:
        """Async variant of ``load()``.

        Returns:
            List of ``Document`` objects.
        """
        docs: List[Document] = []
        for url in self.urls:
            if self.mode == "markdown":
                docs.append(await self._afetch_markdown(url))
            elif self.mode == "content":
                docs.append(await self._afetch_content(url))
            elif self.mode == "scrape":
                docs.extend(await self._afetch_scrape(url))
            elif self.mode == "crawl":
                docs.extend(await self._afetch_crawl(url))
        return docs

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy variant of ``load()``.

        Yields:
            ``Document`` objects.
        """
        for url in self.urls:
            if self.mode == "markdown":
                yield await self._afetch_markdown(url)
            elif self.mode == "content":
                yield await self._afetch_content(url)
            elif self.mode == "scrape":
                for doc in await self._afetch_scrape(url):
                    yield doc
            elif self.mode == "crawl":
                for doc in await self._afetch_crawl(url):
                    yield doc


# MARK: - CloudflareBrowserRunTool


class CloudflareBrowserRunTool(BaseTool):
    """LangGraph agent tool for interacting with web pages via Browser Run.

    Gives agents the ability to fetch web page content, extract structured
    data using AI, discover links, take screenshots, and generate PDFs.

    Example (markdown):
        .. code-block:: python

            from langchain_cloudflare import CloudflareBrowserRunTool

            tool = CloudflareBrowserRunTool(mode="markdown")
            result = tool.invoke({"url": "https://example.com"})

    Example (AI-powered JSON extraction):
        .. code-block:: python

            tool = CloudflareBrowserRunTool(
                mode="json",
                json_prompt="Extract the company name, industry, and employee count.",
            )
            result = tool.invoke({"url": "https://example.com/about"})

    Example (in a LangGraph agent):
        .. code-block:: python

            from langgraph.prebuilt import create_react_agent
            from langchain_cloudflare import ChatCloudflareWorkersAI

            llm = ChatCloudflareWorkersAI()
            tools = [
                CloudflareBrowserRunTool(mode="markdown"),
                CloudflareBrowserRunTool(mode="json", json_prompt="Extract key facts."),
                CloudflareBrowserRunTool(mode="links"),
            ]
            agent = create_react_agent(llm, tools)

    Key init args:
        mode: str
            One of ``"markdown"``, ``"json"``, ``"links"``, ``"screenshot"``, ``"pdf"``.

        account_id: str
            Cloudflare account ID.  Falls back to ``CF_ACCOUNT_ID`` env var.

        api_token: str
            Cloudflare API token with *Browser Rendering – Edit* permission.
            Falls back to ``CF_API_TOKEN`` then ``CF_AI_API_TOKEN`` env var.
    """

    # BaseTool fields
    name: str = "cloudflare_browser_run"
    description: str = (
        "Fetch and extract content from a web page using Cloudflare Browser Run. "
        "Input must be a URL string. "
        "Returns rendered page content (markdown, structured JSON, links, etc.) "
        "depending on the configured mode."
    )

    # MARK: - Fields
    mode: Literal["markdown", "json", "links", "screenshot", "pdf"] = "markdown"
    """Tool mode: determines which Browser Run endpoint to call."""

    account_id: str = Field(default_factory=from_env("CF_ACCOUNT_ID", default=""))
    """Cloudflare account ID."""

    api_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env(
            ["CF_API_TOKEN", "CF_AI_API_TOKEN"], default=None
        )
    )
    """API token with Browser Rendering – Edit permission."""

    # JSON mode options
    json_prompt: Optional[str] = None
    """Natural language prompt for AI extraction (``/json`` mode)."""

    json_response_format: Optional[Dict[str, Any]] = None
    """JSON schema for structured extraction (``/json`` mode)."""

    # Shared Browser Run options
    goto_options: Optional[Dict[str, Any]] = None
    """Page navigation options."""

    viewport: Optional[Dict[str, Any]] = None
    """Viewport dimensions."""

    wait_for_selector: Optional[Dict[str, Any]] = None
    """Wait for a CSS selector before returning."""

    cookies: Optional[List[Dict[str, Any]]] = None
    """Cookies to set before navigation."""

    extra_headers: Optional[Dict[str, str]] = None
    """Extra HTTP headers sent with the browser request."""

    reject_resource_types: Optional[List[str]] = None
    """Resource types to block."""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    """Timeout in seconds for individual HTTP requests."""

    # Internal
    _headers: Dict[str, str] = PrivateAttr()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool and validate credentials."""
        super().__init__(**kwargs)

        if not self.account_id:
            raise ValueError(TokenErrors.NO_ACCOUNT_ID_SET)
        if not self.api_token or not self.api_token.get_secret_value():
            raise ValueError(TokenErrors.INSUFFICIENT_BROWSER_RUN_TOKEN)

        self._headers = _build_headers(self.api_token.get_secret_value())

        # Set descriptive name and description per mode
        self.name = f"cloudflare_browser_run_{self.mode}"
        _mode_descriptions = {
            "markdown": (
                "Fetch a web page and return its content as clean markdown. "
                "Input must be a URL string."
            ),
            "json": (
                "Extract structured JSON data from a web page using AI. "
                "Input must be a URL string."
            ),
            "links": (
                "Discover and return all links found on a web page. "
                "Input must be a URL string."
            ),
            "screenshot": (
                "Capture a screenshot of a web page and return it as "
                "base64-encoded PNG. Input must be a URL string."
            ),
            "pdf": (
                "Generate a PDF of a web page and return it as base64-encoded "
                "data. Input must be a URL string."
            ),
        }
        if self.mode in _mode_descriptions:
            self.description = _mode_descriptions[self.mode]

    # MARK: - Private helpers

    def _shared_body(self) -> Dict[str, Any]:
        """Return shared optional params for the request body."""
        return _build_shared_options(
            goto_options=self.goto_options,
            viewport=self.viewport,
            wait_for_selector=self.wait_for_selector,
            cookies=self.cookies,
            extra_headers=self.extra_headers,
            reject_resource_types=self.reject_resource_types,
        )

    # MARK: - Tool execution

    def _run(self, url: str) -> str:
        """Execute the tool synchronously.

        Args:
            url: The URL to process.

        Returns:
            String result (markdown, JSON string, link list, or base64 bytes
            description for binary endpoints).
        """
        import json as json_mod

        base = _build_browser_run_url(self.account_id, self.mode)
        body: Dict[str, Any] = {"url": url, **self._shared_body()}

        if self.mode == "json":
            if self.json_prompt:
                body["prompt"] = self.json_prompt
            if self.json_response_format:
                body["response_format"] = self.json_response_format

        resp = requests.post(
            base, headers=self._headers, json=body, timeout=self.request_timeout
        )
        resp.raise_for_status()

        if self.mode in ("screenshot", "pdf"):
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type or "text/html" in content_type:
                data = resp.json()
                _check_api_response(data)
                raise RuntimeError(
                    f"Browser Run returned {content_type} instead of binary "
                    f"data for /{self.mode}: {data}"
                )
            return base64.b64encode(resp.content).decode("utf-8")

        data = resp.json()
        _check_api_response(data)

        if self.mode == "markdown":
            return str(data.get("result", ""))
        elif self.mode == "json":
            result = data.get("result", {})
            return (
                json_mod.dumps(result, indent=2)
                if isinstance(result, dict)
                else str(result)
            )
        elif self.mode == "links":
            links = data.get("result", [])
            return "\n".join(links)
        else:
            return str(data.get("result", ""))

    async def _arun(self, url: str) -> str:
        """Execute the tool asynchronously.

        Args:
            url: The URL to process.

        Returns:
            String result.
        """
        import json as json_mod

        import httpx

        base = _build_browser_run_url(self.account_id, self.mode)
        body: Dict[str, Any] = {"url": url, **self._shared_body()}

        if self.mode == "json":
            if self.json_prompt:
                body["prompt"] = self.json_prompt
            if self.json_response_format:
                body["response_format"] = self.json_response_format

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(base, headers=self._headers, json=body)
            resp.raise_for_status()

        if self.mode in ("screenshot", "pdf"):
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type or "text/html" in content_type:
                data = resp.json()
                _check_api_response(data)
                raise RuntimeError(
                    f"Browser Run returned {content_type} instead of binary "
                    f"data for /{self.mode}: {data}"
                )
            encoded = base64.b64encode(resp.content).decode("utf-8")
            return encoded

        data = resp.json()
        _check_api_response(data)

        if self.mode == "markdown":
            return str(data.get("result", ""))
        elif self.mode == "json":
            result = data.get("result", {})
            return (
                json_mod.dumps(result, indent=2)
                if isinstance(result, dict)
                else str(result)
            )
        elif self.mode == "links":
            links = data.get("result", [])
            return "\n".join(links)
        else:
            return str(data.get("result", ""))
