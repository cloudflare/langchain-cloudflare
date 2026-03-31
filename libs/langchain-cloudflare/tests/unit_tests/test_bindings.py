# ruff: noqa: T201
"""Unit tests for bindings.py utilities."""

from langchain_cloudflare.bindings import (
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
