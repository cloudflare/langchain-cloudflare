"""Pydantic translation and execution through existing native adapters."""

# MARK: - IMPORTS
from datetime import date, datetime, time
from enum import Enum
from typing import Dict, List, Literal, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator

from langchain_cloudflare import (
    CloudflareWorkersAIDecisionModel,
    CloudflareWorkersAIExtractionModel,
)


# MARK: - SCHEMAS AND NATIVE FIXTURES
class Category(str, Enum):
    billing = "billing"
    technical = "technical"


class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice identifier printed in the source")
    total: float = Field(gt=0)
    due_date: date
    contacts: List[str]
    po_number: Optional[str]


class Decision(BaseModel):
    urgent: bool = Field(description="Is this an urgent support request?")
    team: Category


class Mixed(BaseModel):
    invoice: Invoice
    paid: bool = Field(
        description="Does the source explicitly say payment is completed?"
    )
    department: Literal["billing", "technical"]


EXTRACTION = {
    "model": "clairic",
    "truncated": False,
    "answers": {
        "f0": {"type": "text", "value": "INV-2291", "span": [8, 16]},
        "f1": {"type": "number", "value": 1080},
        "f2": {"type": "date", "value": "2026-10-02"},
        "f3": {"type": "list", "value": ["Dana Ruiz", "Priya Shah"]},
        "f4": {"type": "text", "value": None},
    },
    "usage": {"input_tokens": 100, "output_tokens": 0},
}
EXPECTED = {
    "invoice_number": "INV-2291",
    "total": 1080,
    "due_date": "2026-10-02",
    "contacts": ["Dana Ruiz", "Priya Shah"],
    "po_number": None,
}


def extraction(binding=None):
    return CloudflareWorkersAIExtractionModel(
        account_id="account", api_token="token", ai_gateway=None, binding=binding
    )


def decision(binding=None):
    return CloudflareWorkersAIDecisionModel(
        account_id="account", api_token="token", ai_gateway=None, binding=binding
    )


def response(result):
    result_response = MagicMock()
    result_response.json.return_value = {"result": result}
    return result_response


# MARK: - REST AND RUNNABLE CONTRACT
def test_invoice_uses_existing_rest_transport_and_pydantic():
    wrapper = extraction().with_structured_output(Invoice, include_raw=True)
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(EXTRACTION),
    ) as post:
        result = wrapper.invoke({"state": "invoice", "reference_date": "2026-10-19"})
    assert isinstance(result["parsed"], Invoice)
    assert result["parsed"].model_dump(mode="json") == EXPECTED
    assert result["raw"] == {"extraction": EXTRACTION}
    assert result["parsing_error"] is None
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "clairic"
    assert payload["reference_date"] == "2026-10-19"
    assert "invoice_number" in payload["fields"]["f0"]["instructions"]
    assert "Invoice identifier" in payload["fields"]["f0"]["instructions"]
    assert payload["fields"]["f3"]["items"] == "text"


def test_decision_enum_and_boolean_threshold():
    native = {
        "answers": {
            "f0": {"type": "noul", "noul": 0.7},
            "f1": {"type": "choice", "choice": "c1"},
        }
    }
    wrapper = decision().with_structured_output(Decision, boolean_threshold=0.7)
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(native),
    ) as post:
        result = wrapper.invoke("Support request")
    assert result.urgent is False
    assert result.team is Category.technical
    questions = post.call_args.kwargs["json"]["questions"]
    assert questions["f1"]["criteria"] == {"c0": '"billing"', "c1": '"technical"'}


@pytest.mark.parametrize("primary", ["extraction", "decision"])
def test_mixed_fixed_nested_schema_from_either_model(primary):
    native = {
        "answers": {
            "f5": {"type": "noul", "noul": 0.01},
            "f6": {"type": "choice", "choice": "c0"},
        }
    }
    wrapper = (
        extraction().with_structured_output(Mixed, decision_model=decision())
        if primary == "extraction"
        else decision().with_structured_output(Mixed, extraction_model=extraction())
    )
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        side_effect=[response(EXTRACTION), response(native)],
    ) as post:
        result = wrapper.invoke("Invoice")
    assert result.invoice.model_dump(mode="json") == EXPECTED
    assert result.paid is False
    assert result.department == "billing"
    assert post.call_count == 2


async def test_async_binding_and_batch_share_public_adapter():
    binding = MagicMock()
    binding.run = AsyncMock(return_value=EXTRACTION)
    wrapper = extraction(binding).with_structured_output(Invoice)
    results = await wrapper.abatch(
        ["invoice one", "invoice two"], config={"max_concurrency": 1}
    )
    assert [item.model_dump(mode="json") for item in results] == [EXPECTED, EXPECTED]
    assert binding.run.await_count == 2
    assert all(
        call.args[0] == "@cf/cloudflare/clairic" for call in binding.run.await_args_list
    )
    assert wrapper.get_output_schema() is Invoice


def test_sync_batch():
    wrapper = extraction().with_structured_output(Invoice)
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(EXTRACTION),
    ) as post:
        results = wrapper.batch(["one", "two"], config={"max_concurrency": 1})
    assert all(isinstance(item, Invoice) for item in results)
    assert post.call_count == 2


# MARK: - OPTIONALS, ALIASES, CONSTRAINTS
def test_nullable_decisions_and_numeric_enum():
    class OptionalDecision(BaseModel):
        approved: Optional[bool]
        rating: Optional[Literal[1, 2]]

    native = {
        "answers": {
            "f0": {"type": "choice", "choice": "c2"},
            "f1": {"type": "choice", "choice": "c1"},
        }
    }
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(native),
    ):
        result = decision().with_structured_output(OptionalDecision).invoke("state")
    assert result.approved is None
    assert result.rating == 2


def test_alias_dates_integers_and_custom_validator():
    class Values(BaseModel):
        count: int = Field(alias="item_count", ge=1)
        date_value: date
        time_value: time
        timestamp: datetime

        @field_validator("count")
        @classmethod
        def even_count(cls, value):
            if value % 2:
                raise ValueError("count must be even")
            return value

    native = {
        "answers": {
            "f0": {"type": "number", "value": 2},
            "f1": {"type": "date", "value": "2026-10-19"},
            "f2": {"type": "time", "value": "10:30:00"},
            "f3": {"type": "datetime", "value": "2026-10-19T10:30:00Z"},
        }
    }
    wrapper = extraction().with_structured_output(Values)
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(native),
    ) as post:
        result = wrapper.invoke("state")
        assert result.count == 2
        assert result.timestamp.year == 2026
        assert post.call_args.kwargs["json"]["fields"]["f0"]["decimals"] == 0
        assert (
            "item_count"
            in post.call_args.kwargs["json"]["fields"]["f0"]["instructions"]
        )
        native["answers"]["f0"]["value"] = 3
        with pytest.raises(ValidationError, match="even"):
            wrapper.invoke("state")


@pytest.mark.parametrize("include_raw", [False, True])
def test_missing_required_value_does_not_use_default(include_raw):
    class RequiredValue(BaseModel):
        name: str = "default"

    native = {"answers": {"f0": {"type": "text", "value": None}}}
    wrapper = extraction().with_structured_output(
        RequiredValue, include_raw=include_raw
    )
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(native),
    ):
        if include_raw:
            result = wrapper.invoke("state")
            assert result["parsed"] is None
            assert isinstance(result["parsing_error"], ValidationError)
            assert result["raw"]["extraction"] is native
        else:
            with pytest.raises(ValidationError):
                wrapper.invoke("state")


# MARK: - FAIL EARLY AND PRESERVE ERRORS
@pytest.mark.parametrize(
    "annotation",
    [List[Invoice], Dict[str, str], Union[str, int], Optional[Invoice], List[int]],
)
def test_unsupported_fields_rejected_before_transport(annotation):
    from pydantic import create_model

    schema = create_model("Unsupported", value=(annotation, ...))
    with patch("langchain_cloudflare._native_models.requests.post") as post:
        with pytest.raises(ValueError, match="Unsupported structured-output field"):
            extraction().with_structured_output(schema, decision_model=decision())
    post.assert_not_called()


def test_recursion_and_root_model_rejected():
    class Recursive(BaseModel):
        child: "Recursive"

    Recursive.model_rebuild()
    for schema in (Recursive, RootModel[List[str]]):
        with pytest.raises(ValueError, match="Unsupported"):
            extraction().with_structured_output(schema)


def test_plain_schema_and_missing_companions():
    with pytest.raises(TypeError, match="BaseModel"):
        extraction().with_structured_output({"type": "object"})
    with pytest.raises(ValueError, match="decision_model"):
        extraction().with_structured_output(Decision)
    with pytest.raises(ValueError, match="extraction_model"):
        decision().with_structured_output(Invoice)


def test_wrong_companion_type_rejected_before_calls():
    with pytest.raises(TypeError, match="decision_model"):
        extraction().with_structured_output(Mixed, decision_model=extraction())


def test_optional_enum_and_sibling_nested_models():
    class Child(BaseModel):
        value: Optional[Category]

    class Parent(BaseModel):
        first: Child
        second: Child

    native = {
        "answers": {
            "f0": {"type": "choice", "choice": "c2"},
            "f1": {"type": "choice", "choice": "c0"},
        }
    }
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(native),
    ):
        result = decision().with_structured_output(Parent).invoke("state")
    assert result.first.value is None
    assert result.second.value is Category.billing


def test_decision_field_limit_rejected_before_calls():
    from pydantic import create_model

    schema = create_model(
        "TooMany", **{f"question_{i}": (bool, ...) for i in range(65)}
    )
    with pytest.raises(ValueError, match="64"):
        decision().with_structured_output(schema)


@pytest.mark.parametrize("threshold", [-1, 2, float("nan"), float("inf")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="boolean_threshold"):
        decision().with_structured_output(Decision, boolean_threshold=threshold)


def test_http_error_propagates_even_with_include_raw():
    failed = response(EXTRACTION)
    failed.raise_for_status.side_effect = requests.HTTPError("provider failed")
    with patch(
        "langchain_cloudflare._native_models.requests.post", return_value=failed
    ):
        with pytest.raises(requests.HTTPError, match="provider failed"):
            extraction().with_structured_output(Invoice, include_raw=True).invoke(
                "state"
            )


def test_sync_mixed_binding_rejected_before_rest_call():
    wrapper = extraction().with_structured_output(
        Mixed, decision_model=decision(MagicMock())
    )
    with patch("langchain_cloudflare._native_models.requests.post") as post:
        with pytest.raises(ValueError, match="ainvoke"):
            wrapper.invoke("state")
    post.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [
        {"state": "s", "unknown": True},
        {"state": {}, "images": []},
        {"state": "s", "audio": []},
    ],
)
def test_invalid_input_rejected_before_calls(value):
    with patch("langchain_cloudflare._native_models.requests.post") as post:
        with pytest.raises((ValueError, TypeError)):
            extraction().with_structured_output(Invoice).invoke(value)
    post.assert_not_called()


@pytest.mark.parametrize("mutation", ["truncated", "missing", "wrong_type"])
def test_malformed_answer_is_parsing_error(mutation):
    from copy import deepcopy

    native = deepcopy(EXTRACTION)
    if mutation == "truncated":
        native["truncated"] = True
    elif mutation == "missing":
        native["answers"].pop("f0")
    else:
        native["answers"]["f0"]["type"] = "number"
    with patch(
        "langchain_cloudflare._native_models.requests.post",
        return_value=response(native),
    ):
        result = (
            extraction()
            .with_structured_output(Invoice, include_raw=True)
            .invoke("state")
        )
    assert result["parsed"] is None
    assert result["parsing_error"] is not None
