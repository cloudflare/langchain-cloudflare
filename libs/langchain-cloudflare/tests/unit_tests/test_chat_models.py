"""Test CloudflareWorkersAI Chat API wrapper."""

from typing import Any, Dict, List, Optional, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompt_values import ChatPromptValueConcrete
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel as PydanticBaseModel

from langchain_cloudflare.chat_models import (
    DEFAULT_MODEL_BEHAVIOR,
    MODEL_BEHAVIORS,
    ChatCloudflareWorkersAI,
    _convert_message_to_dict,
    get_model_behavior,
)


class TestChatCloudflareWorkersAI(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatCloudflareWorkersAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "account_id": "my_account_id",
            "api_token": "my_api_token",
            "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        }


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        # Test case with a single HumanMessage
        (
            [HumanMessage(content="Hello, AI!")],
            [{"role": "user", "content": "Hello, AI!"}],
        ),
        # Test case with SystemMessage, HumanMessage, and AIMessage without tool calls
        (
            [
                SystemMessage(content="System initialized."),
                HumanMessage(content="Hello, AI!"),
                AIMessage(content="Response from AI"),
            ],
            [
                {"role": "system", "content": "System initialized."},
                {"role": "user", "content": "Hello, AI!"},
                {"role": "assistant", "content": "Response from AI"},
            ],
        ),
        # Test case with ToolMessage and tool_call_id
        (
            [
                ToolMessage(
                    content="Tool message content", tool_call_id="tool_call_123"
                ),
            ],
            [
                {
                    "role": "tool",
                    "content": "Tool message content",
                    "tool_call_id": "tool_call_123",
                }
            ],
        ),
    ],
)
def test_convert_messages_to_cloudflare_format(
    messages: List[BaseMessage], expected: List[Dict[str, Any]]
) -> None:
    # Convert each message individually and collect results
    result = [_convert_message_to_dict(message) for message in messages]

    for i, item in enumerate(result):
        if item.get("role") == "tool" and "name" in item and item["name"] is None:
            del item["name"]

    assert result == expected


# MARK: - Reasoning Content Tests


class TestReasoningContent:
    """Test reasoning_content extraction in _create_chat_result."""

    def _create_llm(self, model: str = "@cf/qwen/qwen3-30b-a3b-fp8"):
        """Create a ChatCloudflareWorkersAI instance for testing.

        Dynamic routes only construct on the OpenAI-compatible endpoint over
        REST; the format is irrelevant to the response parsing under test.
        """
        kwargs = {}
        if model.lower().startswith("dynamic/"):
            kwargs["endpoint_format"] = "openai_compatible"

        return ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model=model,
            **kwargs,
        )

    def test_reasoning_content_extracted_for_qwen(self):
        """Qwen response with reasoning_content should surface as content blocks."""
        llm = self._create_llm("@cf/qwen/qwen3-30b-a3b-fp8")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "925",
                            "reasoning_content": "Let me calculate 25 * 37...",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, list)
        thinking_blocks = [b for b in msg.content if b["type"] == "thinking"]
        text_blocks = [b for b in msg.content if b["type"] == "text"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Let me calculate 25 * 37..."
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "925"

    def test_no_reasoning_content_when_absent(self):
        """Qwen response without reasoning_content should have plain string content."""
        llm = self._create_llm("@cf/qwen/qwen3-30b-a3b-fp8")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                        }
                    }
                ],
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, str)
        assert msg.content == "Hello!"

    @pytest.mark.parametrize(
        "model",
        [
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/acme/brand-new-model-with-no-registry-entry",
            "dynamic/rt-fallback",
        ],
        ids=["registered-non-reasoning", "unregistered", "dynamic-route"],
    )
    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
    def test_reasoning_detected_regardless_of_registry(self, model, field):
        """Reasoning is read off the response, not gated on the registry.

        A dynamic route resolves server-side and a newly released model has
        no registry entry, so neither can be looked up ahead of the call.
        """
        llm = self._create_llm(model)
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                            field: "Some reasoning",
                        }
                    }
                ],
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, list)
        thinking_blocks = [b for b in msg.content if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Some reasoning"

    def test_null_reasoning_key_stays_plain_string(self):
        """A null `reasoning` key must not produce content blocks.

        This is the live Mistral response shape: the key is always present
        and always null (confirmed 3/3), which is why dropping the registry
        gate is a no-op for it.
        """
        llm = self._create_llm("@cf/mistralai/mistral-small-3.1-24b-instruct")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                            "reasoning": None,
                            "reasoning_content": None,
                        }
                    }
                ],
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, str)
        assert msg.content == "Hello!"

    def test_reasoning_content_empty_string_not_added(self):
        """Empty reasoning_content should result in plain string content."""
        llm = self._create_llm("@cf/qwen/qwen3-30b-a3b-fp8")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                            "reasoning_content": "",
                        }
                    }
                ],
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, str)
        assert msg.content == "Hello!"

    def test_reasoning_content_extracted_for_glm(self):
        """GLM response with reasoning_content should surface as content blocks."""
        llm = self._create_llm("@cf/zai-org/glm-4.7-flash")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "925",
                            "reasoning_content": "25 * 37 = 925",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, list)
        thinking_blocks = [b for b in msg.content if b["type"] == "thinking"]
        text_blocks = [b for b in msg.content if b["type"] == "text"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "25 * 37 = 925"
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "925"

    def test_reasoning_content_with_tool_calls_qwen(self):
        """Qwen reasoning_content + tool_calls should preserve both."""
        llm = self._create_llm("@cf/qwen/qwen3-30b-a3b-fp8")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "I need to check the weather...",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "SF"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        # Reasoning should be surfaced as content blocks
        assert isinstance(msg.content, list), (
            "Expected list content blocks when both reasoning and tool_calls present"
        )
        thinking_blocks = [b for b in msg.content if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "I need to check the weather..."

        # Tool calls should also be present
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"city": "SF"}

    def test_reasoning_content_with_tool_calls_glm(self):
        """GLM reasoning_content + tool_calls should preserve both."""
        llm = self._create_llm("@cf/zai-org/glm-4.7-flash")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "The user wants weather data...",
                            "tool_calls": [
                                {
                                    "id": "call_def",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "NYC"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, list), (
            "Expected list content blocks when both reasoning and tool_calls present"
        )
        thinking_blocks = [b for b in msg.content if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "The user wants weather data..."

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"city": "NYC"}

    def test_reasoning_content_with_tool_calls_gpt_oss(self):
        """GPT-OSS reasoning_content + tool_calls should preserve both."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Let me look up the stock price...",
                            "tool_calls": [
                                {
                                    "id": "call_ghi",
                                    "type": "function",
                                    "function": {
                                        "name": "get_stock_price",
                                        "arguments": '{"ticker": "AAPL"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, list), (
            "Expected list content blocks when both reasoning and tool_calls present"
        )
        thinking_blocks = [b for b in msg.content if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Let me look up the stock price..."

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_stock_price"
        assert msg.tool_calls[0]["args"] == {"ticker": "AAPL"}

    def test_reasoning_content_with_tool_calls_and_dict_content(self):
        """Dict content should be normalized before creating content blocks."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": {"announcements": []},
                            "reasoning_content": "Need to check the schema.",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "SF"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, list)
        text_blocks = [b for b in msg.content if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == '{"announcements": []}'

    def test_tool_calls_without_reasoning_content_unchanged(self):
        """Tool calls without reasoning_content produce empty string."""
        llm = self._create_llm("@cf/qwen/qwen3-30b-a3b-fp8")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_xyz",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "LA"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        # No reasoning_content, so content should be empty string
        assert msg.content == ""
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"

    SAMPLING_PARAMS = {
        "max_tokens": 100,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "tool_choice": "required",
        "temperature": 0.7,
    }

    def test_glm_unsupported_params_removed(self):
        """glm-4.7-flash should only drop max_tokens and repetition_penalty.

        top_k and tool_choice are accepted by the model and must survive.
        """
        llm = self._create_llm("@cf/zai-org/glm-4.7-flash")

        translated = llm._translate_params_for_model(dict(self.SAMPLING_PARAMS))

        assert "max_tokens" not in translated
        assert "repetition_penalty" not in translated
        assert translated["top_k"] == 50
        assert translated["tool_choice"] == "required"
        assert translated["temperature"] == 0.7

    @pytest.mark.parametrize(
        "model",
        ["@cf/zai-org/glm-5.2", "@cf/zai-org/glm-5.3-flash"],
    )
    def test_modern_glm_preserves_supported_params(self, model):
        """Modern GLM models should keep every parameter their schemas accept."""
        llm = self._create_llm(model)

        translated = llm._translate_params_for_model(dict(self.SAMPLING_PARAMS))

        assert translated == self.SAMPLING_PARAMS

    def test_modern_glm_does_not_fall_back_to_legacy_entry(self):
        """glm-5.x keys must be matched before the legacy 'glm' entry.

        get_model_behavior() returns the first substring match in insertion
        order, so reordering MODEL_BEHAVIORS would silently route glm-5.x
        through the glm-4.7-flash restrictions.
        """
        families = list(MODEL_BEHAVIORS)
        assert families.index("glm-5.3-flash") < families.index("glm")
        assert families.index("glm-5.2") < families.index("glm")

        legacy = MODEL_BEHAVIORS["glm"]
        assert legacy.unsupported_params == ("max_tokens", "repetition_penalty")

        for model in ("@cf/zai-org/glm-5.2", "@cf/zai-org/glm-5.3-flash"):
            behavior = get_model_behavior(model)
            assert behavior is not legacy
            assert behavior.unsupported_params == ()

        assert get_model_behavior("@cf/zai-org/glm-4.7-flash") is legacy


# MARK: - Model Behavior Lookup Tests


class TestGetModelBehavior:
    """Test registry lookup, including AI Gateway dynamic routes."""

    # One representative model id per registry family, so a family that stops
    # resolving is caught here rather than in a live suite.
    FAMILY_MODELS = {
        "@cf/zai-org/glm-5.3-flash": "glm-5.3-flash",
        "@cf/zai-org/glm-5.2": "glm-5.2",
        "@cf/zai-org/glm-4.7-flash": "glm",
        "@cf/google/gemma-4-26b-a4b-it": "gemma",
        "@cf/openai/gpt-oss-120b": "gpt-oss",
        "@cf/deepseek-ai/deepseek-v4-pro-0813": "deepseek",
        "@cf/moonshotai/kimi-k2.6": "kimi",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast": "llama",
        "@cf/mistralai/mistral-small-3.1-24b-instruct": "mistral",
        "@cf/nvidia/nemotron-3-120b-a12b": "nemotron",
        "@cf/qwen/qwen3-30b-a3b-fp8": "qwen",
    }

    @pytest.mark.parametrize(("model", "family"), sorted(FAMILY_MODELS.items()))
    def test_model_ids_resolve_to_their_family(self, model, family):
        """Non-dynamic model ids keep resolving exactly as before."""
        assert get_model_behavior(model) is MODEL_BEHAVIORS[family]

    def test_unknown_model_falls_back_to_default(self):
        """A model with no registry entry gets the default behavior."""
        assert get_model_behavior("@cf/acme/brand-new-model") is DEFAULT_MODEL_BEHAVIOR

    @pytest.mark.parametrize(
        "model",
        [
            "dynamic/rt-fallback",
            "dynamic/rt-qwen",
            "dynamic/rt-glm",
            "dynamic/mistral-backup",
            "dynamic/my-llama-route",
            "DYNAMIC/My-Gemma-Route",
        ],
    )
    def test_dynamic_routes_never_family_match(self, model):
        """Route names are user-chosen, so a family substring means nothing.

        An AI Gateway dynamic route resolves server-side and can fall back to
        a different provider between calls, so no family can be inferred
        before the request is sent.
        """
        assert get_model_behavior(model) is DEFAULT_MODEL_BEHAVIOR


# MARK: - GPT-OSS Model Tests


class TestGptOss:
    """Test GPT-OSS model behavior in _create_chat_result and param translation."""

    def _create_llm(self, model: str = "@cf/openai/gpt-oss-120b"):
        """Create a ChatCloudflareWorkersAI instance for testing."""
        return ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model=model,
        )

    def test_gpt_oss_120b_basic_response(self):
        """GPT-OSS 120B should parse OpenAI-compatible chat completions response."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello World",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert msg.content == "Hello World"
        assert "reasoning_content" not in msg.additional_kwargs

    def test_gpt_oss_20b_basic_response(self):
        """GPT-OSS 20B should parse OpenAI-compatible chat completions response."""
        llm = self._create_llm("@cf/openai/gpt-oss-20b")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello World",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert msg.content == "Hello World"
        assert "reasoning_content" not in msg.additional_kwargs

    def test_gpt_oss_tool_calls_parsed(self):
        """GPT-OSS should parse tool calls from OpenAI-format response."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "NYC"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert msg.content == ""
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"city": "NYC"}
        assert msg.tool_calls[0]["id"] == "call_123"

    def test_gpt_oss_all_params_preserved(self):
        """GPT-OSS should not strip any standard params."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        params = {
            "max_tokens": 256,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "tool_choice": "auto",
        }

        translated = llm._translate_params_for_model(params)

        assert translated["max_tokens"] == 256
        assert translated["temperature"] == 0.6
        assert translated["top_p"] == 0.9
        assert translated["top_k"] == 40
        assert translated["repetition_penalty"] == 1.1
        assert translated["frequency_penalty"] == 0.5
        assert translated["presence_penalty"] == 0.5
        assert translated["tool_choice"] == "auto"

    def test_gpt_oss_reasoning_content_extracted(self):
        """GPT-OSS should extract reasoning_content as content blocks."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "925",
                            "reasoning_content": "25 * 37 = 925",
                        }
                    }
                ],
            }
        }

        result = llm._create_chat_result(response)
        msg = result.generations[0].message

        assert isinstance(msg.content, list)
        thinking_blocks = [b for b in msg.content if b["type"] == "thinking"]
        text_blocks = [b for b in msg.content if b["type"] == "text"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "25 * 37 = 925"
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "925"

    def test_gpt_oss_response_format_normalized(self):
        """OpenAI-style response_format should be normalized for Cloudflare."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        params = {
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Data",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                    "strict": True,
                },
            },
        }

        translated = llm._translate_params_for_model(params)

        rf = translated["response_format"]
        assert rf["type"] == "json_schema"
        # Should be flat schema, not nested under "name"/"schema"
        assert "name" not in rf["json_schema"]
        assert "schema" not in rf["json_schema"]
        assert rf["json_schema"]["type"] == "object"
        assert "x" in rf["json_schema"]["properties"]

    def test_gpt_oss_response_format_json_object_unchanged(self):
        """json_object response_format should pass through unchanged."""
        llm = self._create_llm("@cf/openai/gpt-oss-120b")
        params = {
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        translated = llm._translate_params_for_model(params)

        assert translated["response_format"] == {"type": "json_object"}


# MARK: - Session Affinity Tests
class TestSessionAffinity:
    """Tests for prompt caching via x-session-affinity header."""

    def test_session_id_sets_header(self):
        """session_id should set x-session-affinity header on the client."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            session_id="my-session-123",
        )
        assert llm.client.headers["x-session-affinity"] == "my-session-123"
        assert llm.async_client.headers["x-session-affinity"] == "my-session-123"

    def test_no_session_id_no_header(self):
        """Without session_id, x-session-affinity header should not be set."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        )
        assert "x-session-affinity" not in llm.client.headers

    def test_session_id_with_binding_skips_client(self):
        """With binding, session_id should be stored but no client created."""
        llm = ChatCloudflareWorkersAI(
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            binding=object(),
            session_id="my-session-123",
        )
        assert llm.session_id == "my-session-123"
        assert llm.client is None


# MARK: - AI Gateway Request Handling Tests
class TestAIGatewayHeaders:
    """Tests for AI Gateway timeout and retry headers."""

    def test_aig_headers_set_with_gateway(self):
        """AI Gateway headers should be set when ai_gateway is configured."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ai_gateway="my-gateway",
            aig_request_timeout=5000,
            aig_max_attempts=3,
            aig_retry_delay=1000,
            aig_backoff="exponential",
        )
        assert llm.client.headers["cf-aig-request-timeout"] == "5000"
        assert llm.client.headers["cf-aig-max-attempts"] == "3"
        assert llm.client.headers["cf-aig-retry-delay"] == "1000"
        assert llm.client.headers["cf-aig-backoff"] == "exponential"

    def test_aig_headers_not_set_without_gateway(self):
        """AI Gateway headers should NOT be set when ai_gateway is not configured."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            aig_request_timeout=5000,
            aig_max_attempts=3,
        )
        assert "cf-aig-request-timeout" not in llm.client.headers
        assert "cf-aig-max-attempts" not in llm.client.headers

    def test_aig_partial_headers(self):
        """Only specified AI Gateway headers should be set."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ai_gateway="my-gateway",
            aig_request_timeout=5000,
        )
        assert llm.client.headers["cf-aig-request-timeout"] == "5000"
        assert "cf-aig-max-attempts" not in llm.client.headers
        assert "cf-aig-retry-delay" not in llm.client.headers
        assert "cf-aig-backoff" not in llm.client.headers

    def test_session_id_with_aig_headers(self):
        """Session affinity and AI Gateway headers should coexist."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ai_gateway="my-gateway",
            session_id="session-456",
            aig_request_timeout=5000,
        )
        assert llm.client.headers["x-session-affinity"] == "session-456"
        assert llm.client.headers["cf-aig-request-timeout"] == "5000"


# MARK: - Endpoint Format Tests
class TestEndpointFormat:
    """Tests for native Workers AI vs OpenAI-compatible endpoint routing."""

    def test_workers_ai_endpoint_format_uses_native_run_url_and_payload(self):
        """Default endpoint format should preserve existing native run behavior."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        )

        messages, params = llm._create_message_dicts(
            [HumanMessage(content="Hello")],
            stop=None,
        )
        payload = llm._create_request_payload(messages, params)

        assert llm._get_api_url() == (
            "accounts/test_account/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        )
        assert "model" not in payload
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_openai_compatible_endpoint_format_uses_chat_completions_payload(self):
        """OpenAI-compatible format should route to chat completions."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/moonshotai/kimi-k2.6",
            endpoint_format="openai_compatible",
        )

        messages, params = llm._create_message_dicts(
            [HumanMessage(content="Hello")],
            stop=None,
        )
        payload = llm._create_request_payload(messages, params)

        assert llm._get_api_url() == ("accounts/test_account/ai/v1/chat/completions")
        assert payload["model"] == "@cf/moonshotai/kimi-k2.6"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_openai_compatible_endpoint_format_uses_gateway_chat_completions(self):
        """AI Gateway should route OpenAI-compatible requests through the
        unified Workers AI endpoint, gated via cf-aig-gateway-id.

        Since the Workers AI / AI Gateway unification
        (https://blog.cloudflare.com/workers-ai-gateway-unification/),
        AI Gateway no longer uses a separate gateway.ai.cloudflare.com
        host/path -- the URL is identical to the non-gateway case, and
        routing happens via the cf-aig-gateway-id header instead.
        """
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/moonshotai/kimi-k2.6",
            ai_gateway="my-gateway",
            endpoint_format="openai_compatible",
        )

        assert str(llm.client.base_url) == "https://api.cloudflare.com/client/v4/"
        assert llm._get_api_url() == "accounts/test_account/ai/v1/chat/completions"
        assert llm.client.headers["cf-aig-gateway-id"] == "my-gateway"

    def test_openai_compatible_endpoint_format_rejects_binding(self):
        """Bindings use env.AI.run() and cannot select chat completions."""
        with pytest.raises(ValueError, match="openai_compatible"):
            ChatCloudflareWorkersAI(
                model="@cf/moonshotai/kimi-k2.6",
                binding=object(),
                endpoint_format="openai_compatible",
            )

    def test_create_chat_result_accepts_top_level_openai_response(self):
        """OpenAI-compatible responses can arrive without a result wrapper."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/moonshotai/kimi-k2.6",
            endpoint_format="openai_compatible",
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done",
                        "reasoning_content": "Short reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        }

        result = llm._create_chat_result(response)
        message = result.generations[0].message

        assert isinstance(message.content, list)
        assert message.content[0]["type"] == "thinking"
        assert message.content[1] == {"type": "text", "text": "Done"}
        assert message.usage_metadata == {
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        }

    def test_openai_compatible_stream_chunk_parsing(self):
        """OpenAI-compatible streaming deltas should become message chunks."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            endpoint_format="openai_compatible",
        )

        chunk = llm._create_openai_stream_chunk(
            {"choices": [{"delta": {"content": "Hello"}}]}
        )

        assert chunk is not None
        assert chunk.message.content == "Hello"


# MARK: - LangSmith Params Tests
class TestLangSmithParams:
    """Tests for LangSmith tracing parameters."""

    def test_get_ls_params_uses_per_call_model_override(self):
        """LangSmith params should reflect per-call model overrides."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        )

        params = llm._get_ls_params(model="test-model-override-sentinel")

        assert params["ls_model_name"] == "test-model-override-sentinel"


# MARK: - with_structured_output Routing Tests


class _Announcement(PydanticBaseModel):
    title: str
    summary: Optional[str] = None


def _make_llm(model: str) -> ChatCloudflareWorkersAI:
    return ChatCloudflareWorkersAI(
        account_id="test_account",
        api_token="test_token",
        model=model,
    )


class TestWithStructuredOutputRouting:
    """Unit tests for with_structured_output method routing.

    All tests inspect the returned Runnable pipeline without calling the API.
    """

    # MARK: - json_schema method

    def test_json_schema_method_injects_schema_system_message(self):
        """method='json_schema' should prepend a schema system message."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        assert isinstance(chain, RunnableSequence)
        # First step must be the schema-injection lambda
        assert isinstance(chain.first, RunnableLambda)

        # Invoke the lambda with a plain string and confirm schema is injected
        result = chain.first.invoke("tell me something")
        assert isinstance(result, list)
        assert isinstance(result[0], SystemMessage)
        assert "title" in result[0].content

    def test_json_schema_method_sets_json_object_response_format(self):
        """method='json_schema' on llama should bind response_format=json_object."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        # pipeline is: RunnableLambda | bound_llm | output_parser
        # bound_llm is chain.steps[1]
        bound_llm = chain.steps[1]
        assert bound_llm.kwargs.get("response_format") == {"type": "json_object"}

    def test_json_schema_method_merges_existing_system_message(self):
        """Schema system message should merge with an existing system message."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement, method="json_schema")
        inject = chain.first.func

        messages = [SystemMessage(content="Be concise."), HumanMessage(content="hi")]
        result = inject(messages)

        assert isinstance(result[0], SystemMessage)
        assert "Be concise." in result[0].content
        assert "title" in result[0].content

    def test_json_schema_method_injects_schema_for_chat_prompt_value(self):
        """method='json_schema' should also rewrite prompt-value inputs."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        prompt_value = ChatPromptValueConcrete(
            messages=[HumanMessage(content="tell me something")]
        )
        result = chain.first.invoke(prompt_value)

        assert isinstance(result, list)
        assert isinstance(result[0], SystemMessage)
        assert "title" in result[0].content

    def test_json_schema_method_merges_system_message_for_chat_prompt_value(self):
        """Prompt-value inputs should preserve existing system instructions."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement, method="json_schema")
        inject = chain.first.func

        prompt_value = ChatPromptValueConcrete(
            messages=[SystemMessage(content="Be concise."), HumanMessage(content="hi")]
        )
        result = inject(prompt_value)

        assert isinstance(result, list)
        assert isinstance(result[0], SystemMessage)
        assert "Be concise." in result[0].content
        assert "title" in result[0].content

    def test_json_schema_method_works_on_gemma(self):
        """Explicit method='json_schema' on Gemma should follow same path."""
        llm = _make_llm("@cf/google/gemma-4-26b-a4b-it")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        assert isinstance(chain.first, RunnableLambda)
        bound_llm = chain.steps[1]
        assert bound_llm.kwargs.get("response_format") == {"type": "json_object"}

    # MARK: - Gemma auto-routing

    def test_gemma_function_calling_auto_routes_to_json_schema(self):
        """Gemma with method='function_calling' should auto-route to json_schema."""
        llm = _make_llm("@cf/google/gemma-4-26b-a4b-it")
        chain = llm.with_structured_output(_Announcement)

        # Same pipeline shape as json_schema: starts with injection lambda
        assert isinstance(chain.first, RunnableLambda)
        bound_llm = chain.steps[1]
        assert bound_llm.kwargs.get("response_format") == {"type": "json_object"}

    # MARK: - function_calling method (non-Gemma)

    def test_function_calling_on_llama_uses_tool_calling(self):
        """Default method='function_calling' on llama should NOT inject schema."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement)

        # Pipeline is: bound_llm | output_parser — no injection lambda
        assert not isinstance(chain.first, RunnableLambda)

    def test_function_calling_raises_without_schema(self):
        """method='function_calling' with schema=None should raise ValueError."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        with pytest.raises(ValueError, match="schema must be specified"):
            llm.with_structured_output(None, method="function_calling")

    # MARK: - json_mode method

    def test_json_mode_sets_json_object_without_injection(self):
        """method='json_mode' should set json_object but NOT inject a schema message."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement, method="json_mode")

        # Pipeline is: bound_llm | output_parser — no injection lambda
        assert not isinstance(chain.first, RunnableLambda)
        assert chain.first.kwargs.get("response_format") == {"type": "json_object"}

    # MARK: - Mistral guided_json mode

    def test_mistral_json_schema_uses_json_object_with_injection(self):
        """method='json_schema' on Mistral uses json_object + system message injection.

        Mistral's Workers AI doesn't support complex schemas in guided_json, so we
        fall back to json_object: constrain to valid JSON and inject the schema via
        a system message prompt.
        """
        llm = _make_llm("@cf/mistralai/mistral-small-3.1-24b-instruct")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        # json_object path: starts with injection lambda (same as llama/gemma)
        assert isinstance(chain.first, RunnableLambda)
        bound_llm = chain.steps[1]
        assert bound_llm.kwargs.get("response_format") == {"type": "json_object"}

    def test_mistral_function_calling_auto_routes_to_json_schema(self):
        """Mistral with method='function_calling' should auto-route to json_schema.

        Mistral doesn't support tool_choice (see MODEL_BEHAVIORS), so a bound
        tool can never be forced -- the model is always free to answer in
        prose instead of calling it, making tool-calling-based structured
        output unreliable the same way it is for gemma. Confirmed by a real
        integration failure: mistral-small-3.1-24b-instruct returned prose
        with embedded JSON instead of a tool call for a plain extraction
        prompt, so with_structured_output(method="function_calling") (the
        default) returned None.
        """
        llm = _make_llm("@cf/mistralai/mistral-small-3.1-24b-instruct")
        chain = llm.with_structured_output(_Announcement)

        # Same pipeline shape as json_schema: starts with injection lambda
        assert isinstance(chain.first, RunnableLambda)
        bound_llm = chain.steps[1]
        assert bound_llm.kwargs.get("response_format") == {"type": "json_object"}

    # MARK: - gpt-oss json_schema_rf mode

    def test_gpt_oss_json_schema_uses_json_schema_rf(self):
        """method='json_schema' on gpt-oss should bind response_format=json_schema."""
        llm = _make_llm("@cf/openai/gpt-oss-120b")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        assert not isinstance(chain.first, RunnableLambda)
        rf = chain.first.kwargs.get("response_format", {})
        assert rf.get("type") == "json_schema"
        assert "title" in rf.get("json_schema", {})

    def test_gpt_oss_function_calling_auto_routes_to_json_schema(self):
        """gpt-oss with method='function_calling' should auto-route to json_schema.

        Even with tool_choice forced, gpt-oss sometimes answers with
        reasoning + text content blocks (raw JSON embedded in a text block)
        instead of an actual tool call -- same unreliable-tool-calling
        category as gemma/mistral. Confirmed by a real integration failure:
        gpt-oss-20b returned no tool_calls for a plain extraction prompt, so
        with_structured_output(method="function_calling") (the default)
        returned None.
        """
        llm = _make_llm("@cf/openai/gpt-oss-120b")
        chain = llm.with_structured_output(_Announcement)

        assert not isinstance(chain.first, RunnableLambda)
        rf = chain.first.kwargs.get("response_format", {})
        assert rf.get("type") == "json_schema"

    # MARK: - reasoning-model structured output max_tokens floor

    def test_json_schema_rf_sets_max_tokens_floor_for_reasoning_model(self):
        """json_schema_rf structured output should set a max_tokens floor
        for reasoning models when the caller hasn't set one.

        Reasoning models "think" before answering, and an injected schema
        gives them more to reason about -- with no explicit max_tokens,
        Cloudflare's platform default can be exhausted by the reasoning
        phase alone, leaving no budget for the actual answer. Confirmed via
        a real integration failure: gpt-oss-20b intermittently returned only
        a "thinking" content block (no "text" block), so the JSON parser got
        an empty string and structured output failed.
        """
        llm = _make_llm("@cf/openai/gpt-oss-120b")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        assert chain.first.kwargs.get("max_tokens") == 4096

    def test_json_schema_rf_respects_explicit_max_tokens(self):
        """An explicit max_tokens should not be overridden by the floor."""
        llm = ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/openai/gpt-oss-120b",
            max_tokens=256,
        )
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        assert "max_tokens" not in chain.first.kwargs

    def test_json_schema_rf_no_max_tokens_floor_for_non_reasoning_model(self):
        """The floor is reasoning-model-specific; non-reasoning models
        (e.g. llama, via its json_object path) shouldn't get it."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        chain = llm.with_structured_output(_Announcement, method="json_schema")

        # llama uses the default json_object mode, whose pipeline starts
        # with the schema-injection lambda -- the bound llm is steps[1].
        bound_llm = chain.steps[1]
        assert "max_tokens" not in bound_llm.kwargs

    # MARK: - invalid method

    def test_invalid_method_raises(self):
        """Unknown method value should raise ValueError."""
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        with pytest.raises(ValueError, match="Unrecognized method argument"):
            llm.with_structured_output(_Announcement, method="bad_method")  # type: ignore[arg-type]


class TestStreamingSafeUsage:
    """Regression test: streamed chunk usage must survive chunk merging.

    Workers AI reports `usage` on every streamed chunk, not just the last
    one, and includes a `neurons` field that's a float. langchain_core's
    merge_dicts auto-sums matching int keys (prompt_tokens/completion_tokens/
    total_tokens merge fine) but has no rule for combining two floats under
    the same key, so ChatGenerationChunk.__add__ raised TypeError the moment
    a stream produced more than one chunk with usage data -- breaking
    _stream/_astream for any response longer than a single token.
    """

    def test_streaming_safe_usage_drops_neurons(self):
        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        usage = {
            "prompt_tokens": 39,
            "completion_tokens": 1,
            "total_tokens": 40,
            "neurons": 0.204805,
        }

        safe = llm._streaming_safe_usage(usage)

        assert "neurons" not in safe
        assert safe == {
            "prompt_tokens": 39,
            "completion_tokens": 1,
            "total_tokens": 40,
        }

    def test_merging_two_streamed_usage_chunks_does_not_raise(self):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        llm = _make_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        raw_usage_chunk_1 = {
            "prompt_tokens": 39,
            "completion_tokens": 0,
            "total_tokens": 39,
            "neurons": 1.04,
        }
        raw_usage_chunk_2 = {
            "prompt_tokens": 0,
            "completion_tokens": 1,
            "total_tokens": 1,
            "neurons": 0.2048,
        }

        chunk_1 = ChatGenerationChunk(
            message=AIMessageChunk(content="hello"),
            generation_info={"usage": llm._streaming_safe_usage(raw_usage_chunk_1)},
        )
        chunk_2 = ChatGenerationChunk(
            message=AIMessageChunk(content=" world"),
            generation_info={"usage": llm._streaming_safe_usage(raw_usage_chunk_2)},
        )

        merged = chunk_1 + chunk_2

        assert merged.text == "hello world"
        assert merged.generation_info["usage"] == {
            "prompt_tokens": 39,
            "completion_tokens": 1,
            "total_tokens": 40,
        }


# MARK: - Reject If Busy Tests


class TestRejectIfBusy:
    """Test the reject_if_busy field reaches the REST body and binding options."""

    @staticmethod
    def _create_llm(**kwargs):
        return ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model="@cf/qwen/qwen3-30b-a3b-fp8",
            **kwargs,
        )

    def _payload(self, **kwargs):
        """Build the body a REST send site would actually post."""
        llm = self._create_llm(**kwargs)
        return llm._rest_body(
            llm._create_request_payload(
                [{"role": "user", "content": "hi"}],
                llm._translate_params_for_model(dict(llm._default_params)),
            )
        )

    @pytest.mark.parametrize("endpoint_format", ["workers_ai", "openai_compatible"])
    def test_option_emitted_when_set(self, endpoint_format):
        """Both endpoint formats take a top-level options object."""
        payload = self._payload(reject_if_busy=True, endpoint_format=endpoint_format)
        assert payload["options"] == {"rejectIfBusy": True}

    @pytest.mark.parametrize("endpoint_format", ["workers_ai", "openai_compatible"])
    @pytest.mark.parametrize("value", [None, False])
    def test_no_options_key_when_unset(self, endpoint_format, value):
        """Leaving it off must not add an options key at all."""
        payload = self._payload(reject_if_busy=value, endpoint_format=endpoint_format)
        assert "options" not in payload

    def test_default_is_off(self):
        """The field defaults to off, so existing callers are unaffected."""
        assert self._create_llm().reject_if_busy is None
        assert "options" not in self._payload()

    def test_merges_with_model_kwargs_options(self):
        """A caller's own options dict survives; only rejectIfBusy is set.

        model_kwargs is the documented passthrough that already worked for
        REST, so setting both must not double-emit or drop the other keys.
        """
        payload = self._payload(
            reject_if_busy=True,
            model_kwargs={"options": {"someOtherOption": "keep-me"}},
        )
        assert payload["options"] == {
            "someOtherOption": "keep-me",
            "rejectIfBusy": True,
        }

    def test_model_kwargs_passthrough_still_works_alone(self):
        """Without the field, the old passthrough is untouched."""
        payload = self._payload(
            model_kwargs={"options": {"rejectIfBusy": True}},
        )
        assert payload["options"] == {"rejectIfBusy": True}

    def test_binding_options_receive_the_flag(self):
        """The binding must get it in the run options, never in the input."""
        from langchain_cloudflare.bindings import create_binding_run_options

        llm = self._create_llm(reject_if_busy=True, ai_gateway="gw")
        run_options = create_binding_run_options(
            gateway_id=llm.ai_gateway,
            session_id=llm.session_id,
            reject_if_busy=llm.reject_if_busy,
        )

        assert run_options == {
            "gateway": {"id": "gw"},
            "rejectIfBusy": True,
        }

    def test_model_input_object_never_carries_options(self):
        """The binding ignores rejectIfBusy in the input object, so keep it out.

        _create_request_payload builds the object handed to binding.run() as
        its second argument, so the option must only be added by the REST
        senders via _rest_body().
        """
        llm = self._create_llm(reject_if_busy=True)

        payload = llm._create_request_payload([{"role": "user", "content": "hi"}], {})
        assert "options" not in payload

        assert llm._rest_body(payload)["options"] == {"rejectIfBusy": True}

    def test_streaming_body_carries_the_option(self):
        """The streaming send sites must use _rest_body too.

        _stream/_astream build the same input object as _generate, so moving
        the option to the send sites has to cover all four, not just the two
        non-streaming ones.
        """
        captured = {}

        class _FakeStream:
            def __enter__(self):
                raise RuntimeError("stop after capturing the request body")

            def __exit__(self, *exc):
                return False

        llm = self._create_llm(reject_if_busy=True)

        def fake_stream(method, url, json=None, **kwargs):
            captured["json"] = json
            return _FakeStream()

        llm.client.stream = fake_stream

        with pytest.raises(RuntimeError, match="stop after capturing"):
            list(llm.stream("hi"))

        assert captured["json"]["options"] == {"rejectIfBusy": True}
        assert captured["json"]["stream"] is True


# MARK: - Dynamic Route Endpoint Format Validation Tests


class TestDynamicRouteEndpointFormatValidation:
    """A dynamic route over REST needs the OpenAI-compatible endpoint.

    The native endpoint builds the model into the URL, so /ai/run/dynamic/<r>
    is a 400 code 7000. The guard turns that into a construction-time error
    naming the fix, and must fire only when all three conditions hold.
    """

    MESSAGE = "openai_compatible"

    @staticmethod
    def _create_llm(**kwargs):
        defaults = {
            "account_id": "test_account",
            "api_token": "test_token",
            "model": "dynamic/rt-qwen",
        }
        return ChatCloudflareWorkersAI(**{**defaults, **kwargs})

    @pytest.mark.parametrize(
        "model",
        ["dynamic/rt-qwen", "dynamic/rt-fallback", "DYNAMIC/RT-Qwen", "Dynamic/mixed"],
        ids=["lower", "no-family-substring", "upper", "mixed-case"],
    )
    def test_raises_for_dynamic_route_on_default_format(self, model):
        """All three conditions met: dynamic route, workers_ai, no binding."""
        with pytest.raises(ValueError, match=self.MESSAGE):
            self._create_llm(model=model)

    def test_message_is_actionable(self):
        """The message must name the model, the fix, and why it fails."""
        with pytest.raises(ValueError) as excinfo:
            self._create_llm(model="dynamic/rt-qwen")

        message = str(excinfo.value)
        assert "dynamic/rt-qwen" in message
        assert "endpoint_format='openai_compatible'" in message
        assert "7000" in message

    def test_no_raise_with_openai_compatible(self):
        """The documented fix must construct cleanly."""
        llm = self._create_llm(endpoint_format="openai_compatible")
        assert llm.model == "dynamic/rt-qwen"

    def test_no_raise_with_binding(self):
        """Bindings involve no URL, so the route is fine there.

        Bindings also reject openai_compatible outright, so raising here
        would leave a dynamic route with no valid configuration at all.
        """
        llm = ChatCloudflareWorkersAI(model="dynamic/rt-qwen", binding=object())
        assert llm.endpoint_format == "workers_ai"

    @pytest.mark.parametrize(
        "model",
        [
            "@cf/qwen/qwen3-30b-a3b-fp8",
            "@cf/zai-org/glm-5.2",
            "@cf/my-org/dynamic-sounding-model",
        ],
        ids=["qwen", "glm", "dynamic-substring-not-prefix"],
    )
    def test_no_raise_for_normal_models(self, model):
        """Only the dynamic/ prefix counts, not the substring anywhere."""
        llm = self._create_llm(model=model)
        assert llm.endpoint_format == "workers_ai"
