"""Test CloudflareWorkersAI Chat API wrapper."""

from typing import Any, Dict, List, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_cloudflare.chat_models import (
    ChatCloudflareWorkersAI,
    _convert_message_to_dict,
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
        """Create a ChatCloudflareWorkersAI instance for testing."""
        return ChatCloudflareWorkersAI(
            account_id="test_account",
            api_token="test_token",
            model=model,
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

    def test_no_reasoning_content_for_llama(self):
        """Llama model should not extract reasoning_content even if present."""
        llm = self._create_llm("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        response = {
            "result": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                            "reasoning_content": "Some text",
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
        """Qwen response with both reasoning_content and tool_calls should preserve both."""
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
        """GLM response with both reasoning_content and tool_calls should preserve both."""
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
        """GPT-OSS response with both reasoning_content and tool_calls should preserve both."""
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

    def test_tool_calls_without_reasoning_content_unchanged(self):
        """Tool calls without reasoning_content should still produce empty string content."""
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

    def test_glm_unsupported_params_removed(self):
        """GLM unsupported params should be removed."""
        llm = self._create_llm("@cf/zai-org/glm-4.7-flash")
        params = {
            "max_tokens": 100,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "tool_choice": "required",
            "temperature": 0.7,
        }

        translated = llm._translate_params_for_model(params)

        assert "max_tokens" not in translated
        assert "top_k" not in translated
        assert "repetition_penalty" not in translated
        assert "tool_choice" not in translated
        assert translated["temperature"] == 0.7


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
