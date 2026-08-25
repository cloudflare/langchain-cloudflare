"""Cloudflare Browser Run integration for LangChain.

This module provides a document loader and an agent tool backed by
Cloudflare Browser Run (formerly Browser Rendering). Browser Run offers
serverless headless Chrome on Cloudflare's global network, reachable either
over a REST API or, inside a Python Worker, via the ``browser`` binding's
``quickAction()`` RPC method (``env.BROWSER.quickAction(action, params)``,
GA since 2026-05-28). It supports markdown extraction, crawling, scraping,
AI-powered structured data extraction, screenshots, PDFs, link discovery,
combined-format snapshots, and accessibility trees.

    * ``CloudflareBrowserRunLoader`` -- a ``BaseLoader`` for document
      ingestion (RAG pipelines, knowledge-base construction).
    * ``CloudflareBrowserRunTool`` -- a ``BaseTool`` for LangGraph
      agent workflows (research agents, data extraction, live web
      context).

Note:
    ``crawl`` is an async, job-polling endpoint with no ``quickAction()``
    equivalent, so it is REST-only. Every other mode works over both REST
    and the ``binding`` parameter; the binding path is async-only (see each
    class's docstring).

Credit:
    Originally proposed as REST-only in
    https://github.com/cloudflare/langchain-cloudflare/pull/41 by
    Vamshi Mugala (vamshi694).
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

# Python-side mode name -> Quick Action path segment / quickAction() action
# name. Identical for every mode except accessibility_tree, whose REST path
# and binding action name are both the camelCase "accessibilityTree".
QUICK_ACTION_NAMES: Dict[str, str] = {
    "markdown": "markdown",
    "content": "content",
    "scrape": "scrape",
    "crawl": "crawl",
    "json": "json",
    "links": "links",
    "screenshot": "screenshot",
    "pdf": "pdf",
    "snapshot": "snapshot",
    "accessibility_tree": "accessibilityTree",
}

# Modes reachable via the Worker binding's quickAction(). Excludes crawl,
# which is an async job-polling endpoint with no quickAction() equivalent.
BINDING_SUPPORTED_MODES = frozenset(
    mode for mode in QUICK_ACTION_NAMES if mode != "crawl"
)


# MARK: - Helpers


def _build_browser_run_url(
    account_id: str, endpoint: str, browser: Optional[str] = None
) -> str:
    """Build the full Browser Run REST API URL for a given endpoint.

    Args:
        account_id: Cloudflare account ID.
        endpoint: The Quick Action path, e.g. ``"markdown"`` or ``"crawl"``.
        browser: Optional alternate browser runtime, e.g. ``"kitesurf"``
            (REST-only -- appended as a ``?browser=`` query parameter, per
            https://developers.cloudflare.com/changelog/post/2026-08-06-kitesurf/).

    Returns:
        Fully-qualified URL string.
    """
    url = f"{BROWSER_RUN_BASE_URL}/{account_id}/browser-rendering/{endpoint}"
    if browser:
        url = f"{url}?browser={browser}"
    return url


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
    like ``{"success": false, "errors": [...]}``. This helper ensures
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


# MARK: - Binding Helpers


async def _quickaction_call(binding: Any, action: str, params: Dict[str, Any]) -> Any:
    """Call ``binding.quickAction(action, params)`` and return the raw Response.

    ``env.BROWSER.quickAction()`` resolves to ``workers._workers.Response``,
    the Python-native Workers runtime wrapper (unlike ``env.AI.run()``/
    ``env.VECTORIZE.query()``, which resolve directly to parsed data) --
    callers still need to await ``.json()`` or ``.bytes()`` on the result.
    Confirmed live inside a running Python Worker.

    Args:
        binding: The Browser Run Worker binding (``env.BROWSER``).
        action: The quickAction() action name (see ``QUICK_ACTION_NAMES``).
        params: The action parameters, as a plain Python dict.

    Returns:
        The ``workers._workers.Response`` object.

    Raises:
        RuntimeError: If the response status indicates failure.
    """
    from .bindings import convert_payload_for_binding

    js_params = convert_payload_for_binding(params)
    response = await binding.quickAction(action, js_params)

    if not response.ok:
        try:
            text = await response.text()
        except Exception:
            text = ""
        raise RuntimeError(
            f"Browser Run binding error for {action}: HTTP {response.status} {text}"
        )
    return response


async def _quickaction_json(
    binding: Any, action: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Call the binding's quickAction() and return the parsed JSON envelope."""
    from .bindings import convert_quickaction_response

    response = await _quickaction_call(binding, action, params)
    data = await response.json()
    return convert_quickaction_response(data)


async def _quickaction_bytes(
    binding: Any, action: str, params: Dict[str, Any]
) -> bytes:
    """Call the binding's quickAction() and return the raw response bytes.

    The binding's ``quickAction()`` resolves to ``workers._workers.Response``,
    the Python-native Workers runtime wrapper (not a raw JS ``Response``
    proxy) -- confirmed live inside a Python Worker: its ``bytes()`` method
    already returns plain Python ``bytes`` directly, no JS ``ArrayBuffer``
    or ``.to_py()`` conversion involved.
    """
    response = await _quickaction_call(binding, action, params)
    return await response.bytes()  # type: ignore[no-any-return]


# MARK: - CloudflareBrowserRunLoader


class CloudflareBrowserRunLoader(BaseLoader, BaseModel):  # type: ignore[misc]
    """Load documents from web pages using Cloudflare Browser Run.

    Browser Run renders JavaScript-heavy pages on Cloudflare's global
    network and returns the content either via a REST API or, inside a
    Python Worker, via the ``browser`` binding's ``quickAction()`` method.
    This loader converts web pages into LangChain ``Document`` objects
    suitable for RAG pipelines and knowledge-base construction.

    Example (markdown mode, REST):
        .. code-block:: python

            from langchain_cloudflare import CloudflareBrowserRunLoader

            loader = CloudflareBrowserRunLoader(
                urls=["https://developers.cloudflare.com/workers-ai/"],
                mode="markdown",
            )
            docs = loader.load()

    Example (crawl mode, REST):
        .. code-block:: python

            loader = CloudflareBrowserRunLoader(
                urls=["https://developers.cloudflare.com/cloudflare-one/"],
                mode="crawl",
                crawl_limit=50,
                crawl_depth=2,
            )
            docs = loader.load()

    Example (Worker binding, async-only):
        .. code-block:: python

            loader = CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                mode="markdown",
                binding=self.env.BROWSER,
            )
            docs = await loader.aload()

    Key init args:
        urls: list[str]
            URLs to load.

        mode: str
            One of ``"markdown"``, ``"crawl"``, ``"scrape"``, ``"content"``.

        account_id: str
            Cloudflare account ID. Falls back to ``CF_ACCOUNT_ID`` env var.
            Not needed when ``binding`` is provided.

        api_token: str
            Cloudflare API token with *Browser Rendering – Edit* permission.
            Falls back to ``CF_API_TOKEN`` then ``CF_AI_API_TOKEN`` env var.
            Not needed when ``binding`` is provided.

        binding: Any
            Browser Run binding (``env.BROWSER``) for use in Python Workers.
            Async-only (``aload``/``alazy_load``); does not support
            ``mode="crawl"`` (REST-only).
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

    binding: Any = Field(default=None, exclude=True)
    """Browser Run binding (``env.BROWSER``) for use in Python Workers.

    Async-only; does not support ``mode="crawl"``, which is REST-only."""

    # Crawl-specific
    crawl_limit: int = DEFAULT_CRAWL_LIMIT
    """Maximum number of pages to crawl (``/crawl`` mode only)."""

    crawl_depth: int = DEFAULT_CRAWL_DEPTH
    """Maximum link depth from seed URL (``/crawl`` mode only)."""

    crawl_poll_interval: float = DEFAULT_CRAWL_POLL_INTERVAL
    """Seconds between ``/crawl`` status polls."""

    crawl_timeout: float = DEFAULT_CRAWL_TIMEOUT
    """Maximum seconds to wait for a crawl job to finish."""

    crawl_formats: List[str] = Field(default_factory=lambda: ["markdown"])
    """Response formats to request from ``/crawl`` (e.g. ``["markdown", "html"]``)."""

    crawl_options: Optional[Dict[str, Any]] = None
    """Additional raw ``/crawl`` body options not otherwise exposed as
    dedicated fields (e.g. ``source`` -- ``"all"``, ``"sitemaps"``, or
    ``"links"`` -- or ``render``, to skip JS execution for a faster crawl).
    See the `/crawl` endpoint reference for the full set of accepted keys.
    Merged last, so these win over ``crawl_limit``/``crawl_depth``/
    ``crawl_formats`` on key conflicts."""

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

    browser: Optional[Literal["kitesurf"]] = None
    """Alternate browser runtime (REST-only, via ``?browser=`` query param).
    ``"kitesurf"`` is Cloudflare's stateless, agent-optimized browser (GA
    2026-08-06). Not reachable via ``quickAction()`` -- setting this together
    with ``binding`` raises ``ValueError``."""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    """Timeout in seconds for individual HTTP requests."""

    # Internal
    _headers: Dict[str, str] = PrivateAttr()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the loader and validate credentials."""
        super().__init__(**kwargs)

        if self.binding is not None:
            if self.mode == "crawl":
                raise ValueError(
                    "The Browser Run Worker binding does not support "
                    "mode='crawl' -- quickAction() has no crawl action "
                    "since crawl is an async job with polling, not a "
                    "single request/response call. Use account_id/"
                    "api_token (REST) for crawl mode instead."
                )
            if self.browser is not None:
                raise ValueError(
                    "The 'browser' parameter (alternate browser runtimes "
                    "like 'kitesurf') is REST-only -- it's a URL query "
                    "parameter, with no equivalent in the quickAction() "
                    "binding's params object. Use account_id/api_token "
                    "(REST) instead of binding."
                )
            self._headers = {}
            return

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
            _build_browser_run_url(self.account_id, "markdown", browser=self.browser),
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
            _build_browser_run_url(self.account_id, "content", browser=self.browser),
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
            _build_browser_run_url(self.account_id, "scrape", browser=self.browser),
            headers=self._headers,
            json=body,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        _check_api_response(data)
        return self._parse_scrape_result(data, url)

    def _fetch_crawl(self, url: str) -> List[Document]:
        """Crawl a URL via the ``/crawl`` async endpoint.

        Initiates a crawl job, polls until complete, and returns all
        crawled pages as ``Document`` objects.

        Args:
            url: The seed URL to crawl.

        Returns:
            List of ``Document`` objects, one per crawled page.
        """
        # `crawl_url` stays query-string-free -- it's also the base for
        # `results_url` below (job status/results are keyed on job ID only,
        # not on which browser rendered the pages, so `browser` is a
        # start-of-job option, not something to carry into every poll GET).
        crawl_url = _build_browser_run_url(self.account_id, "crawl")

        resp = requests.post(
            crawl_url,
            headers=self._headers,
            json=self._crawl_body(url),
            params={"browser": self.browser} if self.browser else None,
            timeout=self.request_timeout,
        )
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

            poll = requests.get(
                results_url, headers=self._headers, timeout=self.request_timeout
            )
            poll.raise_for_status()
            poll_data = poll.json().get("result", {})
            status = poll_data.get("status", "")

            if status in _CRAWL_TERMINAL_STATUSES:
                break

            time.sleep(self.crawl_poll_interval)

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

            docs.extend(self._parse_crawl_records(page_data, url))

            next_cursor = page_data.get("cursor")
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor

        return docs

    # MARK: - Async private helpers

    async def _afetch_markdown(self, url: str) -> Document:
        """Async variant of ``_fetch_markdown`` (REST)."""
        import httpx

        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                _build_browser_run_url(
                    self.account_id, "markdown", browser=self.browser
                ),
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
        """Async variant of ``_fetch_content`` (REST)."""
        import httpx

        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                _build_browser_run_url(
                    self.account_id, "content", browser=self.browser
                ),
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
        """Async variant of ``_fetch_scrape`` (REST)."""
        import httpx

        elements = self.elements or [{"selector": "body"}]
        body: Dict[str, Any] = {
            "url": url,
            "elements": elements,
            **self._shared_body(),
        }
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                _build_browser_run_url(self.account_id, "scrape", browser=self.browser),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        _check_api_response(data)
        return self._parse_scrape_result(data, url)

    async def _afetch_crawl(self, url: str) -> List[Document]:
        """Async variant of ``_fetch_crawl`` (REST)."""
        import asyncio

        import httpx

        # See _fetch_crawl: crawl_url stays query-string-free since it's
        # also the base for results_url below.
        crawl_url = _build_browser_run_url(self.account_id, "crawl")

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                crawl_url,
                headers=self._headers,
                json=self._crawl_body(url),
                params={"browser": self.browser} if self.browser else None,
            )
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

                if status in _CRAWL_TERMINAL_STATUSES:
                    break

                await asyncio.sleep(self.crawl_poll_interval)

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

                docs.extend(self._parse_crawl_records(page_data, url))

                next_cursor = page_data.get("cursor")
                if next_cursor is None or next_cursor == cursor:
                    break
                cursor = next_cursor

        return docs

    # MARK: - Binding private helpers

    async def _afetch_markdown_binding(self, url: str) -> Document:
        """Async variant of ``_fetch_markdown`` using the Worker binding."""
        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        data = await _quickaction_json(self.binding, "markdown", body)
        _check_api_response(data)
        return Document(
            page_content=data.get("result", ""),
            metadata={"source": url, "mode": "markdown"},
        )

    async def _afetch_content_binding(self, url: str) -> Document:
        """Async variant of ``_fetch_content`` using the Worker binding."""
        body: Dict[str, Any] = {"url": url, **self._shared_body()}
        data = await _quickaction_json(self.binding, "content", body)
        _check_api_response(data)
        return Document(
            page_content=data.get("result", ""),
            metadata={"source": url, "mode": "content"},
        )

    async def _afetch_scrape_binding(self, url: str) -> List[Document]:
        """Async variant of ``_fetch_scrape`` using the Worker binding."""
        elements = self.elements or [{"selector": "body"}]
        body: Dict[str, Any] = {
            "url": url,
            "elements": elements,
            **self._shared_body(),
        }
        data = await _quickaction_json(self.binding, "scrape", body)
        _check_api_response(data)
        return self._parse_scrape_result(data, url)

    # MARK: - Shared parsing helpers

    def _crawl_body(self, url: str) -> Dict[str, Any]:
        """Build the ``/crawl`` request body (REST-only)."""
        return {
            "url": url,
            "limit": self.crawl_limit,
            "depth": self.crawl_depth,
            "formats": self.crawl_formats,
            **self._shared_body(),
            **(self.crawl_options or {}),
        }

    @staticmethod
    def _parse_scrape_result(data: Dict[str, Any], url: str) -> List[Document]:
        """Parse a ``/scrape`` response into one ``Document`` per selector group."""
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

    @staticmethod
    def _parse_crawl_records(
        page_data: Dict[str, Any], seed_url: str
    ) -> List[Document]:
        """Parse a ``/crawl`` results page into ``Document`` objects."""
        docs: List[Document] = []
        for record in page_data.get("records", []):
            if record.get("status") != "completed":
                continue
            content = record.get("markdown") or record.get("html") or ""
            if not content and record.get("json") is not None:
                import json as json_mod

                content = json_mod.dumps(record["json"])
            meta = record.get("metadata", {})
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": record.get("url", seed_url),
                            "mode": "crawl",
                            "title": meta.get("title", ""),
                            "status_code": meta.get("status", 0),
                        },
                    )
                )
        return docs

    # MARK: - Fetch dispatch

    def _fetch_one(self, url: str) -> List[Document]:
        """Fetch a single URL synchronously (REST only)."""
        if self.binding is not None:
            raise NotImplementedError(
                "The Browser Run Worker binding is async-only. Use "
                "aload()/alazy_load() instead of load()/lazy_load() when "
                "a binding is set."
            )
        if self.mode == "markdown":
            return [self._fetch_markdown(url)]
        elif self.mode == "content":
            return [self._fetch_content(url)]
        elif self.mode == "scrape":
            return self._fetch_scrape(url)
        elif self.mode == "crawl":
            return self._fetch_crawl(url)
        return []

    async def _afetch_one(self, url: str) -> List[Document]:
        """Fetch a single URL asynchronously (REST or binding)."""
        if self.binding is not None:
            if self.mode == "markdown":
                return [await self._afetch_markdown_binding(url)]
            elif self.mode == "content":
                return [await self._afetch_content_binding(url)]
            elif self.mode == "scrape":
                return await self._afetch_scrape_binding(url)
            # mode == "crawl" is rejected in __init__ when binding is set.
            return []

        if self.mode == "markdown":
            return [await self._afetch_markdown(url)]
        elif self.mode == "content":
            return [await self._afetch_content(url)]
        elif self.mode == "scrape":
            return await self._afetch_scrape(url)
        elif self.mode == "crawl":
            return await self._afetch_crawl(url)
        return []

    # MARK: - Public API

    def load(self) -> List[Document]:
        """Load all URLs and return a list of ``Document`` objects.

        Returns:
            List of ``Document`` objects, one per page (or more for scrape/crawl).
        """
        docs: List[Document] = []
        for url in self.urls:
            docs.extend(self._fetch_one(url))
        return docs

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load URLs, yielding one ``Document`` at a time.

        Yields:
            ``Document`` objects.
        """
        for url in self.urls:
            yield from self._fetch_one(url)

    async def aload(self) -> List[Document]:
        """Async variant of ``load()``.

        Returns:
            List of ``Document`` objects.
        """
        docs: List[Document] = []
        for url in self.urls:
            docs.extend(await self._afetch_one(url))
        return docs

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy variant of ``load()``.

        Yields:
            ``Document`` objects.
        """
        for url in self.urls:
            for doc in await self._afetch_one(url):
                yield doc


_CRAWL_TERMINAL_STATUSES = (
    "completed",
    "errored",
    "cancelled_by_user",
    "cancelled_due_to_timeout",
    "cancelled_due_to_limits",
)


# MARK: - CloudflareBrowserRunTool


class CloudflareBrowserRunTool(BaseTool):
    """LangGraph agent tool for interacting with web pages via Browser Run.

    Gives agents the ability to fetch web page content, extract structured
    data using AI, discover links, take screenshots, generate PDFs, capture
    combined-format snapshots, and read accessibility trees.

    Example (markdown, REST):
        .. code-block:: python

            from langchain_cloudflare import CloudflareBrowserRunTool

            tool = CloudflareBrowserRunTool(mode="markdown")
            result = tool.invoke({"url": "https://example.com"})

    Example (AI-powered JSON extraction, REST):
        .. code-block:: python

            tool = CloudflareBrowserRunTool(
                mode="json",
                json_prompt="Extract the company name, industry, and employee count.",
            )
            result = tool.invoke({"url": "https://example.com/about"})

    Example (Worker binding, async-only):
        .. code-block:: python

            tool = CloudflareBrowserRunTool(mode="markdown", binding=self.env.BROWSER)
            result = await tool.ainvoke({"url": "https://example.com"})

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
            One of ``"markdown"``, ``"json"``, ``"links"``, ``"screenshot"``,
            ``"pdf"``, ``"snapshot"``, ``"accessibility_tree"``.

        account_id: str
            Cloudflare account ID. Falls back to ``CF_ACCOUNT_ID`` env var.
            Not needed when ``binding`` is provided.

        api_token: str
            Cloudflare API token with *Browser Rendering – Edit* permission.
            Falls back to ``CF_API_TOKEN`` then ``CF_AI_API_TOKEN`` env var.
            Not needed when ``binding`` is provided.

        binding: Any
            Browser Run binding (``env.BROWSER``) for use in Python Workers.
            Async-only (``ainvoke``).
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
    mode: Literal[
        "markdown",
        "json",
        "links",
        "screenshot",
        "pdf",
        "snapshot",
        "accessibility_tree",
    ] = "markdown"
    """Tool mode: determines which Browser Run Quick Action to call."""

    account_id: str = Field(default_factory=from_env("CF_ACCOUNT_ID", default=""))
    """Cloudflare account ID."""

    api_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env(
            ["CF_API_TOKEN", "CF_AI_API_TOKEN"], default=None
        )
    )
    """API token with Browser Rendering – Edit permission."""

    binding: Any = Field(default=None, exclude=True)
    """Browser Run binding (``env.BROWSER``) for use in Python Workers.

    Async-only; use ``ainvoke()`` rather than ``invoke()`` when set."""

    # JSON mode options
    json_prompt: Optional[str] = None
    """Natural language prompt for AI extraction (``/json`` mode)."""

    json_response_format: Optional[Dict[str, Any]] = None
    """JSON schema for structured extraction (``/json`` mode)."""

    # Snapshot mode options
    snapshot_formats: List[str] = Field(
        default_factory=lambda: ["markdown", "screenshot"]
    )
    """Formats to capture in ``/snapshot`` mode -- at least two of
    ``"content"``, ``"screenshot"``, ``"markdown"``, ``"accessibilityTree"``."""

    # Accessibility tree mode options
    accessibility_tree_root: Optional[str] = None
    """Optional CSS selector to scope ``/accessibilityTree`` to a subtree."""

    accessibility_tree_interesting_only: Optional[bool] = None
    """Whether ``/accessibilityTree`` should only return "interesting" nodes."""

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

    browser: Optional[Literal["kitesurf"]] = None
    """Alternate browser runtime (REST-only, via ``?browser=`` query param).
    ``"kitesurf"`` is Cloudflare's stateless, agent-optimized browser (GA
    2026-08-06). Not reachable via ``quickAction()`` -- setting this together
    with ``binding`` raises ``ValueError``."""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    """Timeout in seconds for individual HTTP requests."""

    # Internal
    _headers: Dict[str, str] = PrivateAttr()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool and validate credentials."""
        super().__init__(**kwargs)

        if self.binding is None:
            if not self.account_id:
                raise ValueError(TokenErrors.NO_ACCOUNT_ID_SET)
            if not self.api_token or not self.api_token.get_secret_value():
                raise ValueError(TokenErrors.INSUFFICIENT_BROWSER_RUN_TOKEN)
            self._headers = _build_headers(self.api_token.get_secret_value())
        else:
            if self.browser is not None:
                raise ValueError(
                    "The 'browser' parameter (alternate browser runtimes "
                    "like 'kitesurf') is REST-only -- it's a URL query "
                    "parameter, with no equivalent in the quickAction() "
                    "binding's params object. Use account_id/api_token "
                    "(REST) instead of binding."
                )
            self._headers = {}

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
            "snapshot": (
                "Capture multiple page formats (markdown, screenshot, HTML, "
                "accessibility tree) in a single call and return them as "
                "JSON. Input must be a URL string."
            ),
            "accessibility_tree": (
                "Return the web page's accessibility tree (roles, names, "
                "states, hierarchy) as JSON. Input must be a URL string."
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

    def _build_action_body(self, url: str) -> Dict[str, Any]:
        """Build the mode-specific request body shared by REST and binding calls."""
        body: Dict[str, Any] = {"url": url, **self._shared_body()}

        if self.mode == "json":
            if self.json_prompt:
                body["prompt"] = self.json_prompt
            if self.json_response_format:
                body["response_format"] = self.json_response_format
        elif self.mode == "snapshot":
            body["formats"] = self.snapshot_formats
        elif self.mode == "accessibility_tree":
            if self.accessibility_tree_root is not None:
                body["root"] = self.accessibility_tree_root
            if self.accessibility_tree_interesting_only is not None:
                body["interestingOnly"] = self.accessibility_tree_interesting_only

        return body

    def _format_result(self, data: Dict[str, Any]) -> str:
        """Format a parsed JSON envelope's ``result`` field per mode."""
        import json as json_mod

        if self.mode == "markdown":
            return str(data.get("result", ""))
        elif self.mode == "links":
            links = data.get("result", [])
            return "\n".join(links)
        else:  # json, snapshot, accessibility_tree
            result = data.get("result", {})
            return (
                json_mod.dumps(result, indent=2)
                if isinstance(result, (dict, list))
                else str(result)
            )

    # MARK: - Tool execution

    def _run(self, url: str) -> str:
        """Execute the tool synchronously (REST only).

        Args:
            url: The URL to process.

        Returns:
            String result (markdown, JSON string, link list, or base64 bytes
            description for binary endpoints).
        """
        if self.binding is not None:
            raise NotImplementedError(
                "The Browser Run Worker binding is async-only. Use "
                "ainvoke() instead of invoke() when a binding is set."
            )

        action = QUICK_ACTION_NAMES[self.mode]
        base = _build_browser_run_url(self.account_id, action, browser=self.browser)
        body = self._build_action_body(url)

        resp = requests.post(
            base, headers=self._headers, json=body, timeout=self.request_timeout
        )
        resp.raise_for_status()

        if self.mode in ("screenshot", "pdf"):
            return self._encode_binary_response(
                resp.headers.get("content-type", ""), resp.content, resp.json
            )

        data = resp.json()
        _check_api_response(data)
        return self._format_result(data)

    async def _arun(self, url: str) -> str:
        """Execute the tool asynchronously (REST or binding).

        Args:
            url: The URL to process.

        Returns:
            String result.
        """
        if self.binding is not None:
            return await self._arun_binding(url)

        import httpx

        action = QUICK_ACTION_NAMES[self.mode]
        base = _build_browser_run_url(self.account_id, action, browser=self.browser)
        body = self._build_action_body(url)

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(base, headers=self._headers, json=body)
            resp.raise_for_status()

        if self.mode in ("screenshot", "pdf"):
            return self._encode_binary_response(
                resp.headers.get("content-type", ""), resp.content, resp.json
            )

        data = resp.json()
        _check_api_response(data)
        return self._format_result(data)

    async def _arun_binding(self, url: str) -> str:
        """Execute the tool asynchronously using the Worker binding."""
        action = QUICK_ACTION_NAMES[self.mode]
        body = self._build_action_body(url)

        if self.mode in ("screenshot", "pdf"):
            raw = await _quickaction_bytes(self.binding, action, body)
            return base64.b64encode(raw).decode("utf-8")

        data = await _quickaction_json(self.binding, action, body)
        _check_api_response(data)
        return self._format_result(data)

    @staticmethod
    def _encode_binary_response(content_type: str, content: bytes, json_fn: Any) -> str:
        """Base64-encode a binary REST response, raising on JSON/HTML error bodies."""
        if "application/json" in content_type or "text/html" in content_type:
            data = json_fn()
            _check_api_response(data)
            raise RuntimeError(
                f"Browser Run returned {content_type} instead of binary data: {data}"
            )
        return base64.b64encode(content).decode("utf-8")
