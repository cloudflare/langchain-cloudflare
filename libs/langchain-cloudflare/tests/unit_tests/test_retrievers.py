# ruff: noqa: T201
"""Unit tests for CloudflareAISearchRetriever (offline: no HTTP is issued)."""

import re

import pytest
from langchain_core.documents import Document

from langchain_cloudflare._errors import TokenErrors
from langchain_cloudflare.retrievers import CloudflareAISearchRetriever


def _make_retriever(**overrides):
    """Construct a retriever with valid dummy credentials (no network)."""
    params = {
        "account_id": "abc123",
        "api_token": "valid-token",
        "instance_name": "test-instance",
    }
    params.update(overrides)
    return CloudflareAISearchRetriever(**params)


# MARK: - Token Validation Tests
class TestTokenValidation:
    """Ensure credential validation raises ValueError for bad inputs."""

    def test_no_account_id_raises(self):
        """Missing account_id should raise ValueError."""
        with pytest.raises(ValueError, match="account ID"):
            CloudflareAISearchRetriever(
                account_id="",
                api_token="some-token",
                instance_name="test-instance",
            )

    def test_no_api_token_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Empty api_token (and no CF_API_TOKEN fallback) should raise."""
        monkeypatch.delenv("CF_AI_SEARCH_API_TOKEN", raising=False)
        monkeypatch.delenv("CF_API_TOKEN", raising=False)
        with pytest.raises(ValueError, match="API token"):
            CloudflareAISearchRetriever(
                account_id="abc123",
                api_token="",
                instance_name="test-instance",
            )

    def test_falls_back_to_cf_api_token(self, monkeypatch: pytest.MonkeyPatch):
        """An empty api_token should fall back to the shared CF_API_TOKEN."""
        monkeypatch.delenv("CF_AI_SEARCH_API_TOKEN", raising=False)
        monkeypatch.setenv("CF_API_TOKEN", "global-token")
        retriever = CloudflareAISearchRetriever(
            account_id="abc123",
            api_token="",
            instance_name="test-instance",
        )
        assert retriever.headers["Authorization"] == "Bearer global-token"

    def test_missing_instance_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Missing instance_name should raise ValueError."""
        monkeypatch.delenv("CF_AI_SEARCH_INSTANCE_NAME", raising=False)
        with pytest.raises(ValueError, match="instance_name"):
            CloudflareAISearchRetriever(
                account_id="abc123",
                api_token="valid-token",
                instance_name="",
            )

    def test_none_env_defaults_raises(self, monkeypatch: pytest.MonkeyPatch):
        """When env vars are unset, default empty strings should raise."""
        monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CF_AI_SEARCH_API_TOKEN", raising=False)
        monkeypatch.delenv("CF_API_TOKEN", raising=False)
        monkeypatch.delenv("CF_AI_SEARCH_INSTANCE_NAME", raising=False)
        with pytest.raises(ValueError):
            CloudflareAISearchRetriever()

    def test_valid_credentials_no_error(self):
        """Valid credentials should not raise and should build the search URL."""
        retriever = _make_retriever()
        assert retriever.account_id == "abc123"
        assert retriever._search_url == (
            "https://api.cloudflare.com/client/v4/accounts/abc123/"
            "ai-search/instances/test-instance/search"
        )

    def test_namespace_url(self):
        """Setting a namespace should use the namespace-scoped endpoint."""
        retriever = _make_retriever(namespace="my-ns")
        assert retriever._search_url == (
            "https://api.cloudflare.com/client/v4/accounts/abc123/"
            "ai-search/namespaces/my-ns/instances/test-instance/search"
        )

    def test_namespace_defaults_to_default(self, monkeypatch: pytest.MonkeyPatch):
        """namespace defaults to 'default' when unset."""
        monkeypatch.delenv("CF_AI_SEARCH_NAMESPACE", raising=False)
        assert _make_retriever().namespace == "default"

    def test_default_namespace_uses_instance_endpoint(self):
        """The 'default' namespace uses the canonical bare-instance endpoint."""
        retriever = _make_retriever(namespace="default")
        assert "/namespaces/" not in retriever._search_url
        assert retriever._search_url.endswith(
            "/ai-search/instances/test-instance/search"
        )

    def test_binding_skips_validation(self):
        """When binding is provided, no credentials are needed."""

        class FakeBinding:
            pass

        retriever = CloudflareAISearchRetriever(binding=FakeBinding())
        assert retriever.binding is not None
        assert retriever._search_url == ""

    def test_error_message_is_token_errors_enum(self, monkeypatch: pytest.MonkeyPatch):
        """Verify the error messages match our centralized TokenErrors."""
        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.NO_ACCOUNT_ID_SET))
        ):
            CloudflareAISearchRetriever(account_id="", api_token="tok")

        monkeypatch.delenv("CF_AI_SEARCH_API_TOKEN", raising=False)
        monkeypatch.delenv("CF_API_TOKEN", raising=False)
        with pytest.raises(
            ValueError,
            match=re.escape(str(TokenErrors.INSUFFICIENT_AI_SEARCH_TOKENS)),
        ):
            CloudflareAISearchRetriever(account_id="abc", api_token="")

        with pytest.raises(
            ValueError, match=re.escape(str(TokenErrors.NO_AI_SEARCH_INSTANCE))
        ):
            CloudflareAISearchRetriever(
                account_id="abc", api_token="tok", instance_name=""
            )


# MARK: - Payload Builder Tests
class TestPayloadBuilder:
    """Test the offline request-body builder."""

    def test_default_body(self):
        """Default body uses `query` and only sets max_num_results."""
        body = _make_retriever()._build_body("hello")
        assert body == {
            "query": "hello",
            "ai_search_options": {"retrieval": {"max_num_results": 10}},
        }
        assert "messages" not in body

    def test_constructor_k(self):
        """The constructor `k` controls max_num_results."""
        body = _make_retriever(k=3)._build_body("q")
        assert body["ai_search_options"]["retrieval"]["max_num_results"] == 3

    def test_k_override_and_clamp(self):
        """An invoke-time k overrides the field and clamps to 1..50."""
        retriever = _make_retriever(k=10)
        assert (
            retriever._build_body("q", 999)["ai_search_options"]["retrieval"][
                "max_num_results"
            ]
            == 50
        )
        assert (
            retriever._build_body("q", 0)["ai_search_options"]["retrieval"][
                "max_num_results"
            ]
            == 1
        )

    def test_none_fields_omitted(self):
        """Unset options are omitted; only `retrieval` is present by default."""
        options = _make_retriever()._build_body("q")["ai_search_options"]
        assert set(options.keys()) == {"retrieval"}
        assert set(options["retrieval"].keys()) == {"max_num_results"}

    def test_retrieval_knobs(self):
        """Retrieval knobs are mapped under `retrieval`."""
        retriever = _make_retriever(
            retrieval_type="hybrid",
            match_threshold=0.5,
            filters={"folder": "docs/"},
            boost_by=[{"field": "timestamp", "direction": "desc"}],
        )
        retrieval = retriever._build_body("q")["ai_search_options"]["retrieval"]
        assert retrieval["retrieval_type"] == "hybrid"
        assert retrieval["match_threshold"] == 0.5
        assert retrieval["filters"] == {"folder": "docs/"}
        assert retrieval["boost_by"] == [{"field": "timestamp", "direction": "desc"}]

    def test_toggle_sections(self):
        """rewrite/rerank/cache toggles produce their own sections."""
        options = _make_retriever(
            rewrite_query=True,
            reranking=True,
            cache=True,
        )._build_body("q")["ai_search_options"]
        assert options["query_rewrite"] == {"enabled": True}
        assert options["reranking"] == {"enabled": True}
        assert options["cache"] == {"enabled": True}

    def test_escape_hatch_deep_merge(self):
        """ai_search_options overrides/augments the typed fields."""
        options = _make_retriever(
            ai_search_options={"retrieval": {"max_num_results": 3, "custom": 1}},
        )._build_body("q")["ai_search_options"]
        assert options["retrieval"]["max_num_results"] == 3
        assert options["retrieval"]["custom"] == 1


# MARK: - Response Mapping Tests
class TestResponseMapping:
    """Test chunk -> Document mapping and chunk extraction."""

    def test_chunk_to_document(self):
        """A full chunk maps to a Document with citation metadata."""
        chunk = {
            "id": "chunk-1",
            "score": 0.9,
            "text": "hello world",
            "type": "text",
            "item": {
                "key": "docs/a.md",
                "timestamp": 123.0,
                "metadata": {"foo": "bar"},
            },
            "scoring_details": {"vector_score": 0.8},
        }
        doc = _make_retriever()._chunk_to_document(chunk)
        assert isinstance(doc, Document)
        assert doc.page_content == "hello world"
        assert doc.metadata["id"] == "chunk-1"
        assert doc.metadata["score"] == 0.9
        assert doc.metadata["type"] == "text"
        assert doc.metadata["filename"] == "docs/a.md"
        assert doc.metadata["timestamp"] == 123.0
        assert doc.metadata["scoring_details"] == {"vector_score": 0.8}
        assert doc.metadata["instance_id"] == "test-instance"
        assert doc.metadata["foo"] == "bar"

    def test_chunk_metadata_collision_canonical_wins(self):
        """Canonical keys win over user metadata on collision."""
        chunk = {
            "id": "chunk-1",
            "score": 0.5,
            "text": "x",
            "item": {"key": "k", "metadata": {"id": "user-id", "foo": "bar"}},
        }
        doc = _make_retriever()._chunk_to_document(chunk)
        assert doc.metadata["id"] == "chunk-1"
        assert doc.metadata["foo"] == "bar"

    def test_chunk_to_document_guards(self):
        """Missing item/text should not crash."""
        doc = _make_retriever()._chunk_to_document({"id": "1", "score": 0.1})
        assert doc.page_content == ""
        assert doc.metadata["filename"] is None

    def test_extract_chunks(self):
        """_extract_chunks handles wrapped, unwrapped, and empty shapes."""
        extract = CloudflareAISearchRetriever._extract_chunks
        assert extract({"result": {"chunks": [{"id": "1"}]}}) == [{"id": "1"}]
        assert extract({"chunks": [{"id": "2"}]}) == [{"id": "2"}]
        assert extract({"success": False}) == []
        assert extract({}) == []
        assert extract(None) == []
        assert extract({"result": {"chunks": None}}) == []
