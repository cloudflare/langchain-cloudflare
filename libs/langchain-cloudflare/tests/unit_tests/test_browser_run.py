# ruff: noqa: T201
"""Unit tests for CloudflareBrowserRunLoader and CloudflareBrowserRunTool."""

import pytest

from langchain_cloudflare._errors import TokenErrors
from langchain_cloudflare.loaders import (
    QUICK_ACTION_NAMES,
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

    def test_build_url_accessibility_tree(self):
        """URL for the /accessibilityTree endpoint uses the camelCase segment."""
        url = _build_browser_run_url("x", QUICK_ACTION_NAMES["accessibility_tree"])
        assert url.endswith("/browser-rendering/accessibilityTree")

    def test_build_url_with_kitesurf(self):
        """browser='kitesurf' is appended as a ?browser= query parameter."""
        url = _build_browser_run_url("abc123", "markdown", browser="kitesurf")
        assert url == (
            "https://api.cloudflare.com/client/v4/accounts/abc123"
            "/browser-rendering/markdown?browser=kitesurf"
        )

    def test_build_url_without_browser_has_no_query_string(self):
        """No browser argument means no query string at all."""
        url = _build_browser_run_url("abc123", "markdown")
        assert "?" not in url


# MARK: - Quick Action Name Mapping Tests


class TestQuickActionNames:
    """Tests for the mode -> Quick Action / quickAction() name mapping."""

    def test_most_modes_map_to_themselves(self):
        """Every mode except accessibility_tree maps to its own name."""
        for mode, name in QUICK_ACTION_NAMES.items():
            if mode != "accessibility_tree":
                assert mode == name

    def test_accessibility_tree_maps_to_camel_case(self):
        """accessibility_tree maps to the camelCase accessibilityTree action."""
        assert QUICK_ACTION_NAMES["accessibility_tree"] == "accessibilityTree"


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

    def test_binding_skips_rest_credential_validation(self):
        """Providing a binding should skip account_id/api_token validation."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            account_id="",
            api_token="",
            binding=object(),
        )
        assert loader.binding is not None

    def test_binding_with_crawl_mode_raises(self):
        """crawl mode has no quickAction() equivalent, so binding + crawl errors."""
        with pytest.raises(ValueError, match="crawl"):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                mode="crawl",
                binding=object(),
            )

    def test_binding_with_kitesurf_raises(self):
        """browser='kitesurf' is REST-only, so binding + browser errors."""
        with pytest.raises(ValueError, match="browser"):
            CloudflareBrowserRunLoader(
                urls=["https://example.com"],
                binding=object(),
                browser="kitesurf",
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
        assert loader.crawl_formats == ["markdown"]
        assert loader.crawl_options is None

    def test_custom_crawl_params(self):
        """Custom crawl parameters are stored correctly."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            crawl_limit=100,
            crawl_depth=5,
            crawl_poll_interval=1.0,
            crawl_timeout=600.0,
            crawl_formats=["markdown", "html"],
            crawl_options={"includePatterns": ["/docs/*"], "source": "sitemaps"},
            account_id="abc123",
            api_token="tok",
        )
        assert loader.crawl_limit == 100
        assert loader.crawl_depth == 5
        assert loader.crawl_formats == ["markdown", "html"]
        assert loader.crawl_options == {
            "includePatterns": ["/docs/*"],
            "source": "sitemaps",
        }

    def test_crawl_body_merges_options_last(self):
        """crawl_options should be able to override limit/depth/formats."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            crawl_limit=10,
            crawl_options={"limit": 999},
            account_id="abc123",
            api_token="tok",
        )
        body = loader._crawl_body("https://example.com")
        assert body["limit"] == 999

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

    def test_sync_load_with_binding_raises(self):
        """Sync load() should raise NotImplementedError when binding is set."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            binding=object(),
        )
        with pytest.raises(NotImplementedError, match="async-only"):
            loader.load()

    def test_sync_lazy_load_with_binding_raises(self):
        """Sync lazy_load() should raise NotImplementedError when binding is set."""
        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            binding=object(),
        )
        with pytest.raises(NotImplementedError, match="async-only"):
            list(loader.lazy_load())


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

    def test_binding_skips_rest_credential_validation(self):
        """Providing a binding should skip account_id/api_token validation."""
        tool = CloudflareBrowserRunTool(account_id="", api_token="", binding=object())
        assert tool.binding is not None

    def test_sync_run_with_binding_raises(self):
        """Sync invoke() should raise NotImplementedError when binding is set."""
        tool = CloudflareBrowserRunTool(binding=object())
        with pytest.raises(NotImplementedError, match="async-only"):
            tool._run("https://example.com")

    def test_binding_with_kitesurf_raises(self):
        """browser='kitesurf' is REST-only, so binding + browser errors."""
        with pytest.raises(ValueError, match="browser"):
            CloudflareBrowserRunTool(binding=object(), browser="kitesurf")


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

    def test_snapshot_tool_name_and_defaults(self):
        """Snapshot mode tool name and default formats."""
        tool = CloudflareBrowserRunTool(
            mode="snapshot",
            account_id="abc123",
            api_token="tok",
        )
        assert tool.name == "cloudflare_browser_run_snapshot"
        assert tool.snapshot_formats == ["markdown", "screenshot"]

    def test_accessibility_tree_tool_name(self):
        """Accessibility tree mode tool name."""
        tool = CloudflareBrowserRunTool(
            mode="accessibility_tree",
            account_id="abc123",
            api_token="tok",
        )
        assert tool.name == "cloudflare_browser_run_accessibility_tree"

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
            "json_schema": {
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

    def test_screenshot_json_error_raises(self):
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

    def test_kitesurf_appends_query_param_to_url(self):
        """browser='kitesurf' is appended to the request URL, not the body."""
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="markdown",
            browser="kitesurf",
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

        call_args = mock_post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert url.endswith("?browser=kitesurf")
        assert "browser" not in body

    def test_crawl_kitesurf_does_not_corrupt_results_url(self):
        """browser='kitesurf' on crawl must not leak into the job-status poll URL.

        crawl_url is reused as the base for results_url (f"{crawl_url}/{job_id}");
        if the query string were baked into crawl_url itself, this would produce
        a broken URL like ".../crawl?browser=kitesurf/{job_id}".
        """
        from unittest.mock import MagicMock, patch

        loader = CloudflareBrowserRunLoader(
            urls=["https://example.com"],
            mode="crawl",
            browser="kitesurf",
            account_id="abc123",
            api_token="tok",
        )

        mock_post = MagicMock()
        mock_post.raise_for_status = MagicMock()
        mock_post.json.return_value = {"result": "job-123"}

        mock_get = MagicMock()
        mock_get.raise_for_status = MagicMock()
        mock_get.json.return_value = {"result": {"status": "completed", "records": []}}

        with (
            patch(
                "langchain_cloudflare.loaders.requests.post", return_value=mock_post
            ) as mp,
            patch(
                "langchain_cloudflare.loaders.requests.get", return_value=mock_get
            ) as mg,
        ):
            loader.load()

        post_args = mp.call_args
        post_url = post_args.args[0] if post_args.args else post_args.kwargs.get("url")
        assert "?" not in post_url
        assert post_args.kwargs.get("params") == {"browser": "kitesurf"}

        get_args = mg.call_args
        get_url = get_args.args[0] if get_args.args else get_args.kwargs.get("url")
        assert get_url.endswith("/job-123")
        assert "?" not in get_url

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

        schema = {"type": "json_schema", "json_schema": {"type": "object"}}
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

    def test_snapshot_tool_body_includes_formats(self):
        """Snapshot tool sends the configured formats list."""
        from unittest.mock import MagicMock, patch

        tool = CloudflareBrowserRunTool(
            mode="snapshot",
            snapshot_formats=["markdown", "accessibilityTree"],
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": {"markdown": "# Hi", "accessibilityTree": {}},
        }

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ) as mock_post:
            result = tool._run("https://example.com")

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["formats"] == ["markdown", "accessibilityTree"]
        assert "markdown" in result

    def test_accessibility_tree_tool_body_includes_root(self):
        """Accessibility tree tool sends root/interestingOnly when configured."""
        from unittest.mock import MagicMock, patch

        tool = CloudflareBrowserRunTool(
            mode="accessibility_tree",
            accessibility_tree_root="#main",
            accessibility_tree_interesting_only=True,
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": True, "result": {"role": "main"}}

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ) as mock_post:
            tool._run("https://example.com")

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["root"] == "#main"
        assert body["interestingOnly"] is True

    def test_accessibility_tree_uses_camel_case_url(self):
        """Accessibility tree tool posts to the /accessibilityTree REST path."""
        from unittest.mock import MagicMock, patch

        tool = CloudflareBrowserRunTool(
            mode="accessibility_tree",
            account_id="abc123",
            api_token="tok",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"success": True, "result": {}}

        with patch(
            "langchain_cloudflare.loaders.requests.post", return_value=mock_resp
        ) as mock_post:
            tool._run("https://example.com")

        call_args = mock_post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert url.endswith("/browser-rendering/accessibilityTree")

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


# MARK: - Worker Binding Tests


class TestQuickActionBindingHelpers:
    """Verify the quickAction() binding call/parse helpers with a mocked binding."""

    @pytest.mark.asyncio
    async def test_quickaction_json_success(self):
        """A successful quickAction() JSON response is parsed via response.json()."""
        from unittest.mock import AsyncMock, MagicMock

        from langchain_cloudflare.loaders import _quickaction_json

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(
            return_value={"success": True, "result": "# Hello"}
        )

        mock_binding = MagicMock()
        mock_binding.quickAction = AsyncMock(return_value=mock_response)

        data = await _quickaction_json(mock_binding, "markdown", {"url": "https://x"})

        assert data == {"success": True, "result": "# Hello"}
        mock_binding.quickAction.assert_awaited_once()
        action_arg = mock_binding.quickAction.call_args.args[0]
        assert action_arg == "markdown"

    @pytest.mark.asyncio
    async def test_quickaction_bytes_success(self):
        """A successful binary quickAction() response is read via bytes().

        The binding resolves to ``workers._workers.Response`` (the Python-
        native Workers runtime wrapper), confirmed live inside a Python
        Worker -- its ``bytes()`` method already returns plain Python
        ``bytes``, not a JS ``ArrayBuffer`` needing conversion.
        """
        from unittest.mock import AsyncMock, MagicMock

        from langchain_cloudflare.loaders import _quickaction_bytes

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.bytes = AsyncMock(return_value=b"\x89PNG\r\n\x1a\nfake")

        mock_binding = MagicMock()
        mock_binding.quickAction = AsyncMock(return_value=mock_response)

        raw = await _quickaction_bytes(mock_binding, "screenshot", {"url": "https://x"})
        assert raw == b"\x89PNG\r\n\x1a\nfake"

    @pytest.mark.asyncio
    async def test_quickaction_non_ok_raises(self):
        """A non-ok Response should raise RuntimeError with status + body text."""
        from unittest.mock import AsyncMock, MagicMock

        from langchain_cloudflare.loaders import _quickaction_json

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="bad request")

        mock_binding = MagicMock()
        mock_binding.quickAction = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="400"):
            await _quickaction_json(mock_binding, "markdown", {"url": "https://x"})


class TestToolBindingExecution:
    """Verify CloudflareBrowserRunTool._arun_binding dispatches correctly."""

    @pytest.mark.asyncio
    async def test_arun_binding_markdown(self):
        """Markdown mode via binding returns the result string."""
        from unittest.mock import AsyncMock, MagicMock

        tool = CloudflareBrowserRunTool(mode="markdown", binding=MagicMock())

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(
            return_value={"success": True, "result": "# Hello"}
        )
        tool.binding.quickAction = AsyncMock(return_value=mock_response)

        result = await tool._arun("https://example.com")
        assert result == "# Hello"

    @pytest.mark.asyncio
    async def test_arun_binding_screenshot_base64_encodes(self):
        """Screenshot mode via binding returns base64-encoded bytes."""
        from unittest.mock import AsyncMock, MagicMock

        tool = CloudflareBrowserRunTool(mode="screenshot", binding=MagicMock())

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.bytes = AsyncMock(return_value=b"\x89PNG\r\n\x1a\nfake")
        tool.binding.quickAction = AsyncMock(return_value=mock_response)

        result = await tool._arun("https://example.com")
        assert isinstance(result, str)
        assert len(result) > 0
