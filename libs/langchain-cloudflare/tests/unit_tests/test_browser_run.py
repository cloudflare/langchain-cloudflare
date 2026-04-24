# ruff: noqa: T201
"""Unit tests for CloudflareBrowserRunLoader and CloudflareBrowserRunTool."""

import pytest

from langchain_cloudflare._errors import TokenErrors
from langchain_cloudflare.loaders import (
    CloudflareBrowserRunLoader,
    CloudflareBrowserRunTool,
    _build_browser_run_url,
    _build_headers,
    _build_shared_options,
)

# MARK: - URL Construction Tests


class TestURLConstruction:
    """Tests for Browser Run URL building helpers."""

    def test_build_url_markdown(self):
        """URL for the /markdown endpoint."""
        url = _build_browser_run_url("abc123", "markdown")
        assert url == (
            "https://api.cloudflare.com/client/v4/accounts/abc123"
            "/browser-rendering/markdown"
        )

    def test_build_url_crawl(self):
        """URL for the /crawl endpoint."""
        url = _build_browser_run_url("abc123", "crawl")
        assert url == (
            "https://api.cloudflare.com/client/v4/accounts/abc123"
            "/browser-rendering/crawl"
        )

    def test_build_url_json(self):
        """URL for the /json endpoint."""
        url = _build_browser_run_url("my-acct", "json")
        assert url == (
            "https://api.cloudflare.com/client/v4/accounts/my-acct"
            "/browser-rendering/json"
        )

    def test_build_url_screenshot(self):
        """URL for the /screenshot endpoint."""
        url = _build_browser_run_url("x", "screenshot")
        assert url.endswith("/browser-rendering/screenshot")


# MARK: - Header Tests


class TestHeaders:
    """Tests for header construction."""

    def test_build_headers(self):
        """Headers contain Authorization and Content-Type."""
        headers = _build_headers("my-token")
        assert headers["Authorization"] == "Bearer my-token"
        assert headers["Content-Type"] == "application/json"


# MARK: - Shared Options Tests


class TestSharedOptions:
    """Tests for the shared Browser Run options builder."""

    def test_empty_options(self):
        """No options returns empty dict."""
        assert _build_shared_options() == {}

    def test_goto_options(self):
        """gotoOptions is passed through."""
        opts = _build_shared_options(goto_options={"waitUntil": "networkidle0"})
        assert opts == {"gotoOptions": {"waitUntil": "networkidle0"}}

    def test_viewport(self):
        """viewport is passed through."""
        opts = _build_shared_options(viewport={"width": 1280, "height": 720})
        assert opts == {"viewport": {"width": 1280, "height": 720}}

    def test_multiple_options(self):
        """Multiple options are combined."""
        opts = _build_shared_options(
            viewport={"width": 800, "height": 600},
            reject_resource_types=["image", "font"],
            cookies=[{"name": "a", "value": "b"}],
        )
        assert "viewport" in opts
        assert "rejectResourceTypes" in opts
        assert "cookies" in opts
        assert len(opts) == 3

    def test_none_values_excluded(self):
        """None values are not included in the output."""
        opts = _build_shared_options(
            goto_options=None,
            viewport={"width": 100, "height": 100},
            cookies=None,
        )
        assert "gotoOptions" not in opts
        assert "cookies" not in opts
        assert "viewport" in opts


# MARK: - Loader Token Validation Tests


class TestLoaderTokenValidation:
    """Ensure token validation raises ValueError for bad inputs."""

    def test_no_account_id_raises(self):
        """Missing account_id should raise ValueError."""
        with pytest.raises(ValueError, match="account ID"):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                account_id="",
                api_token="some-token",
            )

    def test_no_api_token_raises(self):
        """Empty api_token should raise ValueError."""
        with pytest.raises(ValueError, match="API token"):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                account_id="abc123",
                api_token="",
            )

    def test_none_env_defaults_raises(self, monkeypatch: pytest.MonkeyPatch):
        """When env vars are unset, default empty strings should raise ValueError."""
        monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CF_API_TOKEN", raising=False)
        monkeypatch.delenv("CF_AI_API_TOKEN", raising=False)
        with pytest.raises(ValueError):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                account_id="",
                api_token="",
            )

    def test_valid_credentials_no_error(self):
        """Valid account_id + api_token should not raise."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            account_id="abc123",
            api_token="valid-token",
        )
        assert loader.account_id == "abc123"

    def test_default_mode_is_markdown(self):
        """Default mode should be markdown."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            account_id="abc123",
            api_token="valid-token",
        )
        assert loader.mode == "markdown"

    def test_error_message_is_token_errors_enum(self):
        """Verify the error message matches our centralized TokenErrors."""
        import re

        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.NO_ACCOUNT_ID_SET))
        ):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                account_id="",
                api_token="tok",
            )

        with pytest.raises(
            ValueError,
            match=re.escape(str(TokenErrors.INSUFFICIENT_BROWSER_RUN_TOKEN)),
        ):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                account_id="abc",
                api_token="",
            )


# MARK: - Loader Configuration Tests


class TestLoaderConfiguration:
    """Tests for loader field defaults and configuration."""

    def test_crawl_defaults(self):
        """Crawl parameters have sensible defaults."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            account_id="abc123",
            api_token="tok",
        )
        assert loader.crawl_limit == 10
        assert loader.crawl_depth == 2
        assert loader.crawl_poll_interval == 2.0
        assert loader.crawl_timeout == 300.0

    def test_custom_crawl_params(self):
        """Custom crawl parameters are stored correctly."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            crawl_limit=100,
            crawl_depth=5,
            crawl_poll_interval=1.0,
            crawl_timeout=600.0,
            account_id="abc123",
            api_token="tok",
        )
        assert loader.crawl_limit == 100
        assert loader.crawl_depth == 5

    def test_scrape_elements_default(self):
        """Elements default to None."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="scrape",
            account_id="abc123",
            api_token="tok",
        )
        assert loader.elements is None

    def test_shared_options_stored(self):
        """Shared browser options are stored on the instance."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            account_id="abc123",
            api_token="tok",
            viewport={"width": 1920, "height": 1080},
            reject_resource_types=["image"],
        )
        assert loader.viewport == {"width": 1920, "height": 1080}
        assert loader.reject_resource_types == ["image"]

    def test_extra_fields_forbidden(self):
        """Extra fields should raise a validation error."""
        with pytest.raises(Exception):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                account_id="abc123",
                api_token="tok",
                unknown_field="bad",
            )


# MARK: - Tool Token Validation Tests


class TestToolTokenValidation:
    """Ensure token validation raises ValueError for bad inputs."""

    def test_no_account_id_raises(self):
        """Missing account_id should raise ValueError."""
        with pytest.raises(ValueError, match="account ID"):
            CloudflareBrowserRunTool(
                account_id="",
                api_token="some-token",
            )

    def test_no_api_token_raises(self):
        """Empty api_token should raise ValueError."""
        with pytest.raises(ValueError, match="API token"):
            CloudflareBrowserRunTool(
                account_id="abc123",
                api_token="",
            )

    def test_none_env_defaults_raises(self, monkeypatch: pytest.MonkeyPatch):
        """When env vars are unset, default empty strings should raise ValueError."""
        monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CF_API_TOKEN", raising=False)
        monkeypatch.delenv("CF_AI_API_TOKEN", raising=False)
        with pytest.raises(ValueError):
            CloudflareBrowserRunTool(account_id="", api_token="")

    def test_valid_credentials_no_error(self):
        """Valid account_id + api_token should not raise."""
        tool = CloudflareBrowserRunTool(
            account_id="abc123",
            api_token="valid-token",
        )
        assert tool.account_id == "abc123"

    def test_default_mode_is_markdown(self):
        """Default mode should be markdown."""
        tool = CloudflareBrowserRunTool(
            account_id="abc123",
            api_token="valid-token",
        )
        assert tool.mode == "markdown"

    def test_error_message_is_token_errors_enum(self):
        """Verify the error message matches our centralized TokenErrors."""
        import re

        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.NO_ACCOUNT_ID_SET))
        ):
            CloudflareBrowserRunTool(account_id="", api_token="tok")

        with pytest.raises(
            ValueError,
            match=re.escape(str(TokenErrors.INSUFFICIENT_BROWSER_RUN_TOKEN)),
        ):
            CloudflareBrowserRunTool(account_id="abc", api_token="")


# MARK: - Tool Configuration Tests


class TestToolConfiguration:
    """Tests for tool field defaults and configuration."""

    def test_name_includes_mode(self):
        """Tool name should include the mode for agent disambiguation."""
        tool = CloudflareBrowserRunTool(
            mode="json",
            account_id="abc123",
            api_token="tok",
        )
        assert tool.name == "cloudflare_browser_run_json"

    def test_markdown_tool_name(self):
        """Markdown mode tool name."""
        tool = CloudflareBrowserRunTool(
            mode="markdown",
            account_id="abc123",
            api_token="tok",
        )
        assert tool.name == "cloudflare_browser_run_markdown"

    def test_json_prompt_stored(self):
        """JSON prompt is stored on the instance."""
        tool = CloudflareBrowserRunTool(
            mode="json",
            json_prompt="Extract the main heading.",
            account_id="abc123",
            api_token="tok",
        )
        assert tool.json_prompt == "Extract the main heading."

    def test_json_response_format_stored(self):
        """JSON response format is stored on the instance."""
        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
            },
        }
        tool = CloudflareBrowserRunTool(
            mode="json",
            json_response_format=schema,
            account_id="abc123",
            api_token="tok",
        )
        assert tool.json_response_format == schema

    def test_description_is_set(self):
        """Tool description should be non-empty."""
        tool = CloudflareBrowserRunTool(
            account_id="abc123",
            api_token="tok",
        )
        assert len(tool.description) > 0

    def test_extra_fields_forbidden(self):
        """Extra fields should raise a validation error."""
        with pytest.raises(Exception):
            CloudflareBrowserRunTool(
                account_id="abc123",
                api_token="tok",
                unknown_field="bad",
            )


# MARK: - Mocked HTTP Behavior Tests


class TestErrorEnvelopes:
    """Verify _check_api_response raises on success=false envelopes."""

    def test_success_false_raises(self):
        """API error envelope should raise RuntimeError."""
        from langchain_cloudflare.loaders import _check_api_response

        with pytest.raises(RuntimeError, match="Browser Run API error"):
            _check_api_response(
                {"success": False, "errors": [{"message": "bad request"}]}
            )

    def test_success_true_passes(self):
        """Normal response should not raise."""
        from langchain_cloudflare.loaders import _check_api_response

        _check_api_response({"success": True, "result": "ok"})

    def test_non_dict_passes(self):
        """Non-dict response should not raise."""
        from langchain_cloudflare.loaders import _check_api_response

        _check_api_response("plain string")
        _check_api_response(["a", "list"])


class TestBinaryEndpointErrorHandling:
    """Verify screenshot/pdf detect JSON error responses instead of blindly encoding."""

    def test_screenshot_json_error_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Screenshot mode should raise when API returns JSON error."""
        from unittest.mock import MagicMock, patch

        tool = CloudflareBrowserRunTool(
            mode="screenshot",
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "success": False,
            "errors": [{"message": "invalid URL"}],
        }

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ):
            with pytest.raises(RuntimeError, match="Browser Run"):
                tool._run("https://example.com")

    def test_screenshot_html_error_raises(self):
        """Screenshot mode should raise when API returns HTML error page."""
        from unittest.mock import MagicMock, patch

        tool = CloudflareBrowserRunTool(
            mode="screenshot",
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.json.return_value = {"success": True, "result": "error page"}

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ):
            with pytest.raises(RuntimeError, match="instead of binary"):
                tool._run("https://example.com")

    def test_screenshot_binary_success(self):
        """Screenshot mode should return base64 when API returns image."""
        from unittest.mock import MagicMock, patch

        tool = CloudflareBrowserRunTool(
            mode="screenshot",
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.content = b"\x89PNG\r\n\x1a\nfake"

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ):
            result = tool._run("https://example.com")
            assert isinstance(result, str)
            assert len(result) > 0


class TestCrawlPolling:
    """Verify crawl timeout, error status, and pagination handling."""

    def test_crawl_timeout_warns(self):
        """Crawl should warn and return partial results on timeout."""
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            crawl_timeout=0.1,
            crawl_poll_interval=0.05,
            account_id="abc123",
            api_token="tok",
        )

        # Mock: POST /crawl returns job_id, GET always returns "processing"
        mock_post = MagicMock()
        mock_post.raise_for_status = MagicMock()
        mock_post.json.return_value = {"result": "job-123"}

        mock_get = MagicMock()
        mock_get.raise_for_status = MagicMock()
        mock_get.json.return_value = {"result": {"status": "processing", "records": []}}

        with (
            patch("langchain_cloudflare.loaders.requests.post", return_value=mock_post),
            patch("langchain_cloudflare.loaders.requests.get", return_value=mock_get),
        ):
            with pytest.warns(UserWarning, match="timed out"):
                docs = loader.load()

        assert docs == []

    def test_crawl_errored_status_stops(self):
        """Crawl should stop polling when job status is errored."""
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            account_id="abc123",
            api_token="tok",
        )

        mock_post = MagicMock()
        mock_post.raise_for_status = MagicMock()
        mock_post.json.return_value = {"result": "job-456"}

        mock_get = MagicMock()
        mock_get.raise_for_status = MagicMock()
        mock_get.json.return_value = {"result": {"status": "errored", "records": []}}

        with (
            patch("langchain_cloudflare.loaders.requests.post", return_value=mock_post),
            patch("langchain_cloudflare.loaders.requests.get", return_value=mock_get),
        ):
            docs = loader.load()

        assert docs == []

    def test_crawl_completed_with_records(self):
        """Crawl should return Documents from completed records."""
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            account_id="abc123",
            api_token="tok",
        )

        mock_post = MagicMock()
        mock_post.raise_for_status = MagicMock()
        mock_post.json.return_value = {"result": "job-789"}

        mock_get = MagicMock()
        mock_get.raise_for_status = MagicMock()
        mock_get.json.return_value = {
            "result": {
                "status": "completed",
                "records": [
                    {
                        "url": "https://example.com",
                        "status": "completed",
                        "markdown": "# Example\nHello world",
                        "metadata": {"title": "Example", "status": 200},
                    },
                    {
                        "url": "https://example.com/about",
                        "status": "completed",
                        "markdown": "# About\nAbout us",
                        "metadata": {"title": "About", "status": 200},
                    },
                ],
            }
        }

        with (
            patch("langchain_cloudflare.loaders.requests.post", return_value=mock_post),
            patch("langchain_cloudflare.loaders.requests.get", return_value=mock_get),
        ):
            docs = loader.load()

        assert len(docs) == 2
        assert docs[0].page_content == "# Example\nHello world"
        assert docs[0].metadata["source"] == "https://example.com"
        assert docs[0].metadata["title"] == "Example"
        assert docs[1].metadata["source"] == "https://example.com/about"


class TestRequestBodyConstruction:
    """Verify request bodies are constructed correctly per mode."""

    def test_markdown_body(self):
        """Markdown mode sends url + shared options."""
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="markdown",
            viewport={"width": 1920, "height": 1080},
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": True, "result": "# Hello"}

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ) as mock_post:
            loader.load()

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["url"] == "https://example.com"
        assert body["viewport"] == {"width": 1920, "height": 1080}

    def test_scrape_body_includes_elements(self):
        """Scrape mode sends elements in the request body."""
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="scrape",
            elements=[{"selector": "h1"}, {"selector": ".price"}],
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": True, "result": []}

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ) as mock_post:
            loader.load()

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["elements"] == [{"selector": "h1"}, {"selector": ".price"}]

    def test_json_tool_body_includes_prompt_and_schema(self):
        """JSON tool sends prompt and response_format in the body."""
        from unittest.mock import MagicMock, patch

        schema = {"type": "json_schema", "schema": {"type": "object"}}
        tool = CloudflareBrowserRunTool(
            mode="json",
            json_prompt="Extract facts.",
            json_response_format=schema,
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": True, "result": {"key": "val"}}

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ) as mock_post:
            tool._run("https://example.com")

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["prompt"] == "Extract facts."
        assert body["response_format"] == schema

    def test_loader_sends_timeout(self):
        """All loader requests include the configured timeout."""
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="markdown",
            request_timeout=30.0,
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": True, "result": "# Hello"}

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ) as mock_post:
            loader.load()

        call_kwargs = mock_post.call_args
        timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert timeout == 30.0
