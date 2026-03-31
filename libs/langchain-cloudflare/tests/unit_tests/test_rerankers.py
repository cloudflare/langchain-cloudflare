# ruff: noqa: T201
"""Unit tests for CloudflareWorkersAIReranker token validation."""

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
