"""Test chat model integration using standard integration tests."""

from typing import Type
import pytest

from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests.chat_models import ChatModelIntegrationTests

from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI


class TestChatCloudflareWorkersAI(ChatModelIntegrationTests):
    """Test CloudflareWorkersAI chat model."""

    @property
    def chat_model_class(self) -> Type[ChatCloudflareWorkersAI]:
        """Get the class of the chat model under test."""
        return ChatCloudflareWorkersAI

    @property
    def chat_model_params(self) -> dict:
        """Get the parameters to initialize the chat model."""
        return {
            "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "temperature": 0.7,
        }

    @property
    def supports_json_mode(self) -> bool:
        """Whether the model supports JSON mode."""
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return False

    @property
    def has_tool_choice(self) -> bool:
        """Whether the model supports tool choice."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return False

    @pytest.mark.xfail(reason=("Does not support tool_choice."))
    def test_tool_calling(self, model: BaseChatModel) -> None:
        super().test_tool_calling(model)

    @pytest.mark.xfail(reason=("Does not support tool_choice."))
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        await super().test_tool_calling_async(model)

    @pytest.mark.xfail(reason=("Does not support tool_choice."))
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)
