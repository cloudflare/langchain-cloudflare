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
        """Qwen response with reasoning_content should surface in response_metadata."""
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

        assert msg.content == "925"
        assert "reasoning_content" in msg.response_metadata
        assert msg.response_metadata["reasoning_content"] == (
            "Let me calculate 25 * 37..."
        )

    def test_no_reasoning_content_when_absent(self):
        """Qwen response without reasoning_content should not have it in metadata."""
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

        assert msg.content == "Hello!"
        assert "reasoning_content" not in msg.response_metadata

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

        assert msg.content == "Hello!"
        assert "reasoning_content" not in msg.response_metadata

    def test_reasoning_content_empty_string_not_added(self):
        """Empty reasoning_content should not be added to response_metadata."""
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

        assert "reasoning_content" not in msg.response_metadata

    def test_reasoning_content_extracted_for_glm(self):
        """GLM response with reasoning_content should surface in response_metadata."""
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

        assert msg.content == "925"
        assert "reasoning_content" in msg.response_metadata
        assert msg.response_metadata["reasoning_content"] == "25 * 37 = 925"

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
