# ruff: noqa: T201
"""Unit tests for CloudflareWorkersAIReranker token validation."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_cloudflare._errors import TokenErrors
from langchain_cloudflare.rerankers import CloudflareWorkersAIReranker


# MARK: - Token Validation Tests
class TestTokenValidation:
    """Ensure token validation raises ValueError (not AttributeError) for bad inputs."""

    def test_no_account_id_raises(self):
        """Missing account_id should raise ValueError."""
        with pytest.raises(ValueError, match="account ID"):
            CloudflareWorkersAIReranker(
                account_id="",
                api_token="some-token",
            )

    def test_no_api_token_raises(self):
        """Empty api_token should raise ValueError, not AttributeError."""
        with pytest.raises(ValueError, match="API token"):
            CloudflareWorkersAIReranker(
                account_id="abc123",
                api_token="",
            )

    def test_none_env_defaults_raises(self, monkeypatch: pytest.MonkeyPatch):
        """When env vars are unset, default empty strings should raise ValueError."""
        monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CF_AI_API_TOKEN", raising=False)
        with pytest.raises(ValueError):
            CloudflareWorkersAIReranker(account_id="", api_token="")

    def test_valid_credentials_no_error(self):
        """Valid account_id + api_token should not raise."""
        reranker = CloudflareWorkersAIReranker(
            account_id="abc123",
            api_token="valid-token",
        )
        assert reranker.account_id == "abc123"

    def test_binding_skips_validation(self):
        """When binding is provided, no credentials are needed."""

        class FakeBinding:
            pass

        reranker = CloudflareWorkersAIReranker(
            binding=FakeBinding(),
        )
        assert reranker.binding is not None

    def test_error_message_is_token_errors_enum(self):
        """Verify the error message matches our centralized TokenErrors."""
        import re

        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.NO_ACCOUNT_ID_SET))
        ):
            CloudflareWorkersAIReranker(account_id="", api_token="tok")

        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.INSUFFICIENT_AI_TOKENS))
        ):
            CloudflareWorkersAIReranker(account_id="abc", api_token="")


# MARK: - Timeout Tests
class TestRequestTimeout:
    """Regression test: REST calls must pass an explicit timeout.

    rerank()'s requests.post() had no timeout at all -- requests has no
    default, so a stuck response would hang forever. arerank()'s
    httpx.AsyncClient() had none either, silently falling back to httpx's
    default 5.0s, which a real integration test exceeded reranking just 4
    short documents under load.
    """

    def test_rerank_passes_timeout_to_requests(self):
        reranker = CloudflareWorkersAIReranker(
            account_id="abc123", api_token="tok", timeout=45.0
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"response": []}}

        with patch(
            "langchain_cloudflare.rerankers.requests.post",
            return_value=mock_response,
        ) as mock_post:
            reranker.rerank(query="q", documents=["doc1"])

        assert mock_post.call_args.kwargs["timeout"] == 45.0

    async def test_arerank_passes_timeout_to_httpx_client(self):
        reranker = CloudflareWorkersAIReranker(
            account_id="abc123", api_token="tok", timeout=45.0
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": {"response": []}}

            async def _post(*args, **kwargs):
                return mock_response

            mock_client.post = _post
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            await reranker.arerank(query="q", documents=["doc1"])

        assert mock_client_cls.call_args.kwargs["timeout"] == 45.0

    def test_default_timeout_is_60_seconds(self):
        reranker = CloudflareWorkersAIReranker(account_id="abc123", api_token="tok")
        assert reranker.timeout == 60.0


# MARK: - AI Gateway Unified Endpoint Tests
class TestAIGatewayUnifiedEndpoint:
    """Regression test: AI Gateway must route via header, not a separate host.

    Since the Workers AI / AI Gateway unification
    (https://blog.cloudflare.com/workers-ai-gateway-unification/), AI Gateway
    no longer uses a separate gateway.ai.cloudflare.com host/path -- routing
    happens via the cf-aig-gateway-id header on the standard endpoint.
    """

    def test_ai_gateway_uses_standard_url_with_header(self):
        reranker = CloudflareWorkersAIReranker(
            account_id="test_account",
            api_token="test_token",
            ai_gateway="my-gateway",
        )

        assert reranker._inference_url == (
            "https://api.cloudflare.com/client/v4/accounts/test_account/ai/run/"
            f"{reranker.model_name}"
        )
        assert reranker.headers["cf-aig-gateway-id"] == "my-gateway"

    def test_no_gateway_header_without_ai_gateway(self):
        reranker = CloudflareWorkersAIReranker(
            account_id="test_account",
            api_token="test_token",
        )

        assert "cf-aig-gateway-id" not in reranker.headers
