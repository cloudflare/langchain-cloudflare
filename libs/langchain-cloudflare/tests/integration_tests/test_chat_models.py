"""Test chat model integration using standard integration tests."""

from typing import Literal, Type

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

    # llama-3.3-70b-instruct-fp8-fast intermittently returns zero chunks when
    # streaming a structured-output response ("Stream returned no chunks --
    # possible API issue" from langchain_tests itself), while the equivalent
    # non-streaming invoke() call succeeds -- a live API/model reliability
    # issue, not a code bug. Confirmed pre-existing (reproduces identically
    # against a clean checkout with none of this session's changes) and
    # intermittent (different subset of these fails each run, never all).
    # Same flaky-model category as gemma-4-26b-a4b-it/gpt-oss-120b elsewhere
    # in this repo, just in a file with no existing FLAKY_MODELS/_model_param
    # infrastructure to hook into, since it's a fixed single-model conformance
    # suite rather than a parametrized model list.
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    def test_structured_output(
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> None:
        super().test_structured_output(model, schema_type)

    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    async def test_structured_output_async(
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> None:
        await super().test_structured_output_async(model, schema_type)

    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)
