"""Native extraction transport contracts; live tests cover model semantics."""

# MARK: - IMPORTS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from langchain_cloudflare import CloudflareWorkersAIExtractionModel
from tests.extraction_helpers import (
    ExtractionQualityError,
    assert_extraction_result,
    extraction_case,
)

# MARK: - FIXTURES
INPUT = {
    "state": "Invoice INV-2291 totals $1080.",
    "fields": {"total": {"type": "number", "decimals": 2, "unit": "USD"}},
    "images": [{"content_type": "image/png", "base64": "encoded-image"}],
    "reference_date": "2026-10-19",
    "options": {"include_quantiles": True, "return_state_text": True},
}
RESULT = {
    "model": "clairic",
    "answers": {"total": {"type": "number", "value": 1080, "confidence": 0.99}},
    "truncated": False,
    "state_text": INPUT["state"],
    "usage": {"input_tokens": 100, "output_tokens": 0, "passes": 1},
}


# MARK: - REST CONTRACT
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reject_if_busy", [None, True])
async def test_rest_transport(asynchronous, reject_if_busy):
    model = CloudflareWorkersAIExtractionModel(
        account_id="account",
        api_token="token",
        ai_gateway="gateway",
        reject_if_busy=reject_if_busy,
        timeout=45,
    )
    response = MagicMock()
    response.json.return_value = {"result": RESULT, "success": True}
    if asynchronous:
        with patch("httpx.AsyncClient") as client_type:
            client = client_type.return_value.__aenter__.return_value
            client.post = AsyncMock(return_value=response)
            result = await model.aextract(**INPUT)
            kwargs = client.post.call_args.kwargs
            assert client_type.call_args.kwargs["timeout"] == 45
    else:
        with patch(
            "langchain_cloudflare._native_models.requests.post", return_value=response
        ) as post:
            result = model.extract(**INPUT)
            kwargs = post.call_args.kwargs
            assert kwargs["timeout"] == 45
    assert result == RESULT
    response.raise_for_status.assert_called_once()
    assert kwargs["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/account/ai/run/"
        "@cf/cloudflare/clairic"
    )
    assert kwargs["headers"] == {
        "Authorization": "Bearer token",
        "cf-aig-gateway-id": "gateway",
    }
    options = {**INPUT["options"]}
    if reject_if_busy:
        options["rejectIfBusy"] = True
    assert kwargs["json"] == {**INPUT, "model": "clairic", "options": options}
    assert INPUT["options"] == {
        "include_quantiles": True,
        "return_state_text": True,
    }


# MARK: - BINDING CONTRACT
@pytest.mark.parametrize("with_options", [False, True])
async def test_binding_transport(with_options):
    js_response = MagicMock()
    js_response.to_py.return_value = RESULT
    binding = MagicMock()
    binding.run = AsyncMock(return_value=js_response)
    model = CloudflareWorkersAIExtractionModel(
        binding=binding,
        account_id="",
        api_token="",
        ai_gateway="gateway" if with_options else None,
        reject_if_busy=with_options,
    )
    assert await model.aextract(**INPUT) == RESULT
    args = (model.model_name, {**INPUT, "model": "clairic"})
    if with_options:
        binding.run.assert_awaited_once_with(
            *args, {"gateway": {"id": "gateway"}, "rejectIfBusy": True}
        )
    else:
        binding.run.assert_awaited_once_with(*args)


async def test_image_only_input_and_optional_omission():
    binding = MagicMock()
    binding.run = AsyncMock(return_value=RESULT)
    model = CloudflareWorkersAIExtractionModel(binding=binding, ai_gateway=None)
    await model.aextract(fields=INPUT["fields"], images=INPUT["images"])
    binding.run.assert_awaited_once_with(
        model.model_name,
        {
            "model": "clairic",
            "state": "",
            "fields": INPUT["fields"],
            "images": INPUT["images"],
        },
    )


# MARK: - ERRORS
def test_chat_model_rejected():
    with pytest.raises(ValidationError):
        CloudflareWorkersAIExtractionModel(model_name="@cf/cloudflare/clef")


def test_sync_binding_requires_async():
    model = CloudflareWorkersAIExtractionModel(binding=MagicMock())
    with pytest.raises(ValueError, match="aextract"):
        model.extract(**INPUT)


# MARK: - MODEL QUALITY CLASSIFICATION
@pytest.mark.parametrize("case", ["sentence", "company_list"])
@pytest.mark.parametrize(
    "invalid_part",
    [None, "type", "confidence", "span"],
    ids=["wrong_value", "wrong_type", "invalid_confidence", "invalid_span"],
)
def test_quality_does_not_mask_malformed_answers(case, invalid_part):
    """Only valid source spans with wrong meanings may be expected-failure."""
    fixture = (
        "press_release_partnership_sentence"
        if case == "sentence"
        else "press_release_company_list"
    )
    payload, expected = extraction_case(fixture)
    key = next(iter(payload["fields"]))
    answer = (
        {"type": "text", "value": "Acme Corp", "span": [0, 9], "p_null": 0.001}
        if case == "sentence"
        else {
            "type": "list",
            "value": ["Acme Corp", "NYSE"],
            "spans": [[0, 9], [11, 15]],
        }
    )
    answer["confidence"] = 0.4
    if invalid_part == "type":
        answer["type"] = "number"
    elif invalid_part == "confidence":
        answer["confidence"] = 1.1
    elif invalid_part == "span":
        answer["span" if case == "sentence" else "spans"] = (
            [0, 4] if case == "sentence" else [[0, 4], [11, 15]]
        )
    result = {
        "model": "clairic",
        "truncated": False,
        "answers": {key: answer},
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    with pytest.raises(AssertionError) as error:
        assert_extraction_result(result, payload, expected)
    assert isinstance(error.value, ExtractionQualityError) == (invalid_part is None)


@pytest.mark.parametrize("value", [None, "Acme Corp", 123])
def test_malformed_list_is_not_model_quality_failure(value):
    """A list must remain a list even when the model has no matching values."""
    payload, expected = extraction_case("press_release_company_list")
    result = {
        "model": "clairic",
        "truncated": False,
        "answers": {
            "companies": {
                "type": "list",
                "value": value,
                "spans": [],
                "confidence": 0.4,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    with pytest.raises(AssertionError) as error:
        assert_extraction_result(result, payload, expected)
    assert not isinstance(error.value, ExtractionQualityError)


@pytest.mark.parametrize(
    "value,p_null",
    [(None, 0.001), ("Acme Corp", 0.1)],
    ids=["null_below_threshold", "value_above_threshold"],
)
def test_null_policy_violation_is_not_model_quality_failure(value, p_null):
    """Provider threshold violations must fail even in a known-quality case."""
    payload, expected = extraction_case("press_release_partnership_sentence")
    result = {
        "model": "clairic",
        "truncated": False,
        "answers": {
            "announcement_sentence": {
                "type": "text",
                "value": value,
                "span": [0, 9],
                "confidence": 0.4,
                "p_null": p_null,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    with pytest.raises(AssertionError) as error:
        assert_extraction_result(result, payload, expected)
    assert not isinstance(error.value, ExtractionQualityError)


# MARK: - IMAGE-SOURCE ANSWERS
@pytest.mark.parametrize("invalid_part", [None, "images", "span", "spell_confidence"])
def test_image_text_has_explicit_source_and_no_state_span(invalid_part):
    """Image text is valid without a character span only for image inputs."""
    payload = {
        "state": "",
        "images": ["data:image/png;base64,fixture"],
        "fields": {"ticker": {"type": "text"}},
    }
    answer = {
        "type": "text",
        "value": "ACME",
        "source": "image",
        "span": None,
        "p_null": 0.03,
        "confidence": 0.99,
        "spell_confidence": 0.99,
    }
    result = {
        "model": "clairic",
        "truncated": False,
        "answers": {"ticker": answer},
        "usage": {
            "input_tokens": 100,
            "output_tokens": 0,
            "image_tokens": 50,
            "state_tokens": 0,
        },
    }
    if invalid_part == "images":
        payload.pop("images")
    elif invalid_part == "span":
        answer["span"] = [0, 4]
    elif invalid_part == "spell_confidence":
        answer["spell_confidence"] = 1.1
    if invalid_part is None:
        assert_extraction_result(result, payload, {"ticker": "ACME"})
    else:
        with pytest.raises(AssertionError) as error:
            assert_extraction_result(result, payload, {"ticker": "ACME"})
        assert not isinstance(error.value, ExtractionQualityError)
