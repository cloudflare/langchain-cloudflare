# ruff: noqa: T201
"""Unit tests for bindings.py utilities."""

from langchain_cloudflare.bindings import (
    convert_aisearch_response,
    convert_quickaction_response,
    convert_reranker_response,
    create_binding_run_options,
)

# MARK: - convert_reranker_response Tests


class TestConvertRerankerResponse:
    """Test convert_reranker_response handles all known response formats."""

    def test_plain_list(self):
        """Direct list of {id, score} dicts should be returned as-is."""
        data = [{"id": 0, "score": 0.99}, {"id": 1, "score": 0.85}]
        assert convert_reranker_response(data) == data

    def test_result_key(self):
        """Dict with 'result' key wrapping a list should unwrap."""
        data = {"result": [{"id": 0, "score": 0.99}]}
        assert convert_reranker_response(data) == [{"id": 0, "score": 0.99}]

    def test_response_key(self):
        """Dict with 'response' key (native AI binding format) should unwrap."""
        data = {
            "response": [{"id": 2, "score": 0.999}, {"id": 0, "score": 0.998}],
            "usage": {
                "prompt_tokens": 8785,
                "completion_tokens": 0,
                "total_tokens": 8785,
            },
        }
        result = convert_reranker_response(data)
        assert len(result) == 2
        assert result[0]["id"] == 2
        assert result[0]["score"] == 0.999

    def test_data_key(self):
        """Dict with 'data' key should unwrap."""
        data = {"data": [{"id": 0, "score": 0.5}]}
        assert convert_reranker_response(data) == [{"id": 0, "score": 0.5}]

    def test_empty_response_list(self):
        """Empty list should return empty list."""
        assert convert_reranker_response([]) == []

    def test_empty_dict(self):
        """Dict with no recognized keys should return empty list."""
        assert convert_reranker_response({}) == []

    def test_none_returns_empty(self):
        """None should return empty list."""
        assert convert_reranker_response(None) == []

    def test_result_key_precedence_over_response(self):
        """When both 'result' and 'response' exist, 'result' takes precedence."""
        data = {
            "result": [{"id": 0, "score": 0.9}],
            "response": [{"id": 1, "score": 0.8}],
        }
        result = convert_reranker_response(data)
        assert result == [{"id": 0, "score": 0.9}]

    def test_response_key_precedence_over_data(self):
        """When both 'response' and 'data' exist, 'response' takes precedence."""
        data = {
            "response": [{"id": 0, "score": 0.9}],
            "data": [{"id": 1, "score": 0.8}],
        }
        result = convert_reranker_response(data)
        assert result == [{"id": 0, "score": 0.9}]


# MARK: - create_binding_run_options Tests
class TestCreateBindingRunOptions:
    """Test create_binding_run_options builds correct options objects."""

    def test_no_options_returns_none(self):
        """No gateway or session_id should return None."""
        assert create_binding_run_options() is None

    def test_gateway_only(self):
        """Only gateway_id should produce gateway options."""
        result = create_binding_run_options(gateway_id="my-gateway")
        assert result == {"gateway": {"id": "my-gateway"}}
        assert "headers" not in result

    def test_session_id_only(self):
        """Only session_id should produce headers options."""
        result = create_binding_run_options(session_id="sess-123")
        assert result == {"headers": {"x-session-affinity": "sess-123"}}
        assert "gateway" not in result

    def test_gateway_and_session_id(self):
        """Both gateway_id and session_id should be combined."""
        result = create_binding_run_options(
            gateway_id="my-gateway", session_id="sess-123"
        )
        assert result == {
            "gateway": {"id": "my-gateway"},
            "headers": {"x-session-affinity": "sess-123"},
        }


# MARK: - convert_aisearch_response Tests
class TestConvertAISearchResponse:
    """Test convert_aisearch_response normalizes all known response shapes."""

    def test_dict_passthrough(self):
        """A dict response (already Python) should be returned as-is."""
        data = {"result": {"chunks": [{"id": "1", "text": "x"}]}}
        assert convert_aisearch_response(data) == data

    def test_list_wrapped(self):
        """A bare list should be wrapped as a chunks result."""
        chunks = [{"id": "1"}, {"id": "2"}]
        assert convert_aisearch_response(chunks) == {"result": {"chunks": chunks}}

    def test_none_returns_empty_chunks(self):
        """None should return an empty chunks result."""
        assert convert_aisearch_response(None) == {"result": {"chunks": []}}

    def test_unknown_returns_empty_chunks(self):
        """An unexpected scalar should return an empty chunks result."""
        assert convert_aisearch_response("nope") == {"result": {"chunks": []}}


# MARK: - convert_quickaction_response Tests
class TestConvertQuickactionResponse:
    """Test convert_quickaction_response normalizes quickAction() JSON bodies."""

    def test_dict_passthrough(self):
        """A dict response (already Python) should be returned as-is."""
        data = {"success": True, "result": "# Hello"}
        assert convert_quickaction_response(data) == data

    def test_list_wrapped_as_result(self):
        """A bare list (e.g. /links) should be wrapped under 'result'."""
        links = ["https://a.example", "https://b.example"]
        assert convert_quickaction_response(links) == {"result": links}

    def test_scalar_wrapped_as_result(self):
        """A bare scalar should be wrapped under 'result'."""
        assert convert_quickaction_response("plain text") == {"result": "plain text"}

    def test_to_py_proxy_is_converted(self):
        """A JS proxy object exposing to_py() should be converted first."""

        class FakeJsProxy:
            def to_py(self):
                return {"success": True, "result": {"role": "main"}}

        result = convert_quickaction_response(FakeJsProxy())
        assert result == {"success": True, "result": {"role": "main"}}
