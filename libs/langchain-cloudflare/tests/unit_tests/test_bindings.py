# ruff: noqa: T201
"""Unit tests for bindings.py utilities."""

import pytest

from langchain_cloudflare.bindings import (
    convert_aisearch_response,
    convert_quickaction_response,
    convert_reranker_response,
    create_binding_run_options,
    create_gateway_options,
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

    def test_reject_if_busy_only(self):
        """rejectIfBusy alone should produce a standalone options object.

        The binding reads it only from this third argument -- it is ignored
        inside the model input object.
        """
        assert create_binding_run_options(reject_if_busy=True) == {"rejectIfBusy": True}

    def test_reject_if_busy_false_is_omitted(self):
        """False/None must not emit the key at all."""
        assert create_binding_run_options(reject_if_busy=False) is None
        assert create_binding_run_options(reject_if_busy=None) is None

    def test_reject_if_busy_combines_with_gateway_and_session(self):
        """rejectIfBusy sits alongside gateway and headers, not inside them."""
        result = create_binding_run_options(
            gateway_id="my-gateway",
            session_id="sess-123",
            reject_if_busy=True,
        )
        assert result == {
            "gateway": {"id": "my-gateway"},
            "headers": {"x-session-affinity": "sess-123"},
            "rejectIfBusy": True,
        }


# MARK: - create_gateway_options Tests
class TestCreateGatewayOptions:
    """The legacy helper must keep behaving as a gateway-only special case."""

    def test_no_gateway_returns_none(self):
        assert create_gateway_options(None) is None
        assert create_gateway_options("") is None

    def test_gateway_matches_run_options(self):
        """It delegates, so it must agree with create_binding_run_options."""
        assert create_gateway_options("my-gateway") == create_binding_run_options(
            gateway_id="my-gateway"
        )
        assert create_gateway_options("my-gateway") == {"gateway": {"id": "my-gateway"}}


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


# MARK: - Binding Argument Placement Tests


class RecordingBinding:
    """Stand-in for env.AI that records exactly how it was called.

    Reconstructing what create_binding_run_options() *would* return does not
    prove what a class actually passes to binding.run(). This records the real
    arguments so the model input object can be checked directly.
    """

    def __init__(self, response):
        self._response = response
        self.model = None
        self.input_object = None
        self.run_options = None
        self.call_count = 0

    async def run(self, model, input_object, options=None):
        self.model = model
        self.input_object = input_object
        self.run_options = options
        self.call_count += 1
        return self._response


class TestRejectIfBusyArgumentPlacement:
    """rejectIfBusy must reach the third argument and never the input object.

    Cloudflare's docs: "Do not add rejectIfBusy to the model input object.
    The binding only applies this option from the third argument."
    """

    @staticmethod
    async def _run_chat(binding):
        from langchain_cloudflare import ChatCloudflareWorkersAI

        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/qwen/qwen3-30b-a3b-fp8",
            binding=binding,
            reject_if_busy=True,
        )
        await llm.ainvoke("hi")

    @staticmethod
    async def _run_embeddings(binding):
        from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings

        embeddings = CloudflareWorkersAIEmbeddings(binding=binding, reject_if_busy=True)
        await embeddings.aembed_query("hi")

    @staticmethod
    async def _run_reranker(binding):
        from langchain_cloudflare.rerankers import CloudflareWorkersAIReranker

        reranker = CloudflareWorkersAIReranker(binding=binding, reject_if_busy=True)
        await reranker.arerank(query="q", documents=["a", "b"])

    RESPONSES = {
        "chat": {"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        "embeddings": {"data": [[0.1, 0.2]]},
        "reranker": {"response": [{"id": 0, "score": 0.9}]},
    }

    @pytest.mark.parametrize("target", ["chat", "embeddings", "reranker"])
    async def test_option_only_in_third_argument(self, target):
        """Every class: options in arg 3, no options key in arg 2."""
        binding = RecordingBinding(self.RESPONSES[target])
        runner = {
            "chat": self._run_chat,
            "embeddings": self._run_embeddings,
            "reranker": self._run_reranker,
        }[target]

        await runner(binding)

        assert binding.call_count == 1
        assert binding.run_options is not None, f"{target} passed no run options"
        assert binding.run_options["rejectIfBusy"] is True
        assert "options" not in binding.input_object, (
            f"{target} leaked options into the model input object: "
            f"{sorted(binding.input_object)}"
        )

    @pytest.mark.parametrize("target", ["chat", "embeddings", "reranker"])
    async def test_no_reject_if_busy_when_flag_unset(self, target):
        """Without the flag, neither argument mentions rejectIfBusy.

        run_options may still be non-None: ai_gateway defaults from the
        AI_GATEWAY env var, so a gateway entry can legitimately be present.
        """
        binding = RecordingBinding(self.RESPONSES[target])

        from langchain_cloudflare import ChatCloudflareWorkersAI
        from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
        from langchain_cloudflare.rerankers import CloudflareWorkersAIReranker

        if target == "chat":
            llm = ChatCloudflareWorkersAI(
                account_id="test_account",
                api_token="test_token",
                model="@cf/qwen/qwen3-30b-a3b-fp8",
                binding=binding,
            )
            await llm.ainvoke("hi")
        elif target == "embeddings":
            await CloudflareWorkersAIEmbeddings(binding=binding).aembed_query("hi")
        else:
            await CloudflareWorkersAIReranker(binding=binding).arerank(
                query="q", documents=["a"]
            )

        assert "rejectIfBusy" not in (binding.run_options or {})
        assert "options" not in binding.input_object
