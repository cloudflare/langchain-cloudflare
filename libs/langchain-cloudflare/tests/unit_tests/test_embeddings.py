# ruff: noqa: T201
"""Unit tests for CloudflareWorkersAIEmbeddings token validation."""

import pytest

from langchain_cloudflare._errors import TokenErrors
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings


# MARK: - Token Validation Tests
class TestTokenValidation:
    """Ensure token validation raises ValueError (not AttributeError) for bad inputs."""

    def test_no_account_id_raises(self):
        """Missing account_id should raise ValueError."""
        with pytest.raises(ValueError, match="account ID"):
            CloudflareWorkersAIEmbeddings(
                account_id="",
                api_token="some-token",
            )

    def test_no_api_token_raises(self):
        """Empty api_token should raise ValueError, not AttributeError."""
        with pytest.raises(ValueError, match="API token"):
            CloudflareWorkersAIEmbeddings(
                account_id="abc123",
                api_token="",
            )

    def test_none_env_defaults_raises(self, monkeypatch: pytest.MonkeyPatch):
        """When env vars are unset, default empty strings should raise ValueError."""
        monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CF_AI_API_TOKEN", raising=False)
        with pytest.raises(ValueError):
            CloudflareWorkersAIEmbeddings(account_id="", api_token="")

    def test_valid_credentials_no_error(self):
        """Valid account_id + api_token should not raise."""
        embeddings = CloudflareWorkersAIEmbeddings(
            account_id="abc123",
            api_token="valid-token",
        )
        assert embeddings.account_id == "abc123"

    def test_binding_skips_validation(self):
        """When binding is provided, no credentials are needed."""

        class FakeBinding:
            pass

        embeddings = CloudflareWorkersAIEmbeddings(
            binding=FakeBinding(),
        )
        assert embeddings.binding is not None

    def test_error_message_is_token_errors_enum(self):
        """Verify the error message matches our centralized TokenErrors."""
        import re

        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.NO_ACCOUNT_ID_SET))
        ):
            CloudflareWorkersAIEmbeddings(account_id="", api_token="tok")

        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.INSUFFICIENT_AI_TOKENS))
        ):
            CloudflareWorkersAIEmbeddings(account_id="abc", api_token="")
