"""Source evidence and dependent candidate qualification in the shared example."""

# MARK: - IMPORTS
import re
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from examples.workers.src.system_one import extract_announcements, group_list_spans
from langchain_cloudflare import (
    CloudflareWorkersAIDecisionModel,
    CloudflareWorkersAIExtractionModel,
)
from tests.extraction_helpers import (
    ExtractionQualityError,
    assert_extraction_result,
    assert_system_one_result,
)
from tests.press_release_examples import PRESS_RELEASE_EXAMPLES


# MARK: - REPEATED MENTIONS
@pytest.fixture
def repeated_mentions():
    example = deepcopy(PRESS_RELEASE_EXAMPLES["press_release_partnership"])
    example["text"] += (
        "\nAcme Corp and TechGiant Inc will collaborate."
        "\nAcme Corp is listed on NYSE; TechGiant Inc is a partner."
    )
    # Three unique values have eight occurrences; neither order nor count
    # corresponds positionally to the value array.
    values = ["Acme Corp", "NYSE", "TechGiant Inc"]
    spans = sorted(
        [
            [match.start(), match.end()]
            for value in values
            for match in re.finditer(re.escape(value), example["text"])
        ],
        reverse=True,
    )
    result = {
        "model": "clairic",
        "answers": {
            "companies": {
                "type": "list",
                "value": values,
                "spans": spans,
                "confidence": 0.9,
            }
        },
        "truncated": False,
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    return example, result


async def test_repeated_mentions_are_grouped_and_qualified(repeated_mentions):
    example, candidates = repeated_mentions
    answer = candidates["answers"]["companies"]
    original = deepcopy(candidates)
    grouped = group_list_spans(
        text=example["text"], values=answer["value"], spans=answer["spans"]
    )
    assert {value: len(spans) for value, spans in grouped.items()} == {
        "Acme Corp": 3,
        "NYSE": 2,
        "TechGiant Inc": 3,
    }
    assert sum(grouped.values(), []) != answer["spans"]
    assert_extraction_result(
        candidates,
        {"state": example["text"], "fields": {"companies": {"type": "list"}}},
        {"companies": answer["value"]},
    )

    extraction_binding = MagicMock()
    extraction_binding.run = AsyncMock(
        side_effect=[
            candidates,
            {
                "model": "clairic",
                "answers": {
                    "ticker_0": {"value": "ACME"},
                    "ticker_1": {"value": None},
                },
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        ]
    )
    decision_binding = MagicMock()
    decision_binding.run = AsyncMock(
        return_value={
            "model": "clef",
            "answers": {
                "is_announcement": {"noul": 0.99},
                "announcement_type": {"choice": "partnership"},
                "context": {"choice": "S0"},
                "role_0": {"choice": "Partner"},
                "role_1": {"choice": "None"},
                "role_2": {"choice": "Partner"},
            },
            "usage": {"input_tokens": 100, "output_tokens": 0},
        }
    )
    result = await extract_announcements(
        text=example["text"],
        extraction_model=CloudflareWorkersAIExtractionModel(binding=extraction_binding),
        decision_model=CloudflareWorkersAIDecisionModel(binding=decision_binding),
    )
    assert_system_one_result(result, example)
    assert result["stages"]["candidates"] == original
    assert candidates == original
    decision_binding.run.assert_awaited_once()
    decision_payload = decision_binding.run.call_args.args[1]
    assert decision_payload["state"] == example["text"]
    roles = {
        key: question
        for key, question in decision_payload["questions"].items()
        if key.startswith("role_")
    }
    assert list(roles) == ["role_0", "role_1", "role_2"]
    for question, value in zip(roles.values(), answer["value"]):
        assert repr(value) in question["instructions"]
    assert extraction_binding.run.await_count == 2


# MARK: - INVALID EVIDENCE
@pytest.mark.parametrize(
    "values,spans",
    [
        (["Acme Corp"], [[0, 4]]),  # Substring not in declared values.
        (["Acme Corp", "TechGiant Inc"], [[0, 9]]),  # Unanchored value.
        (["Acme Corp"], [[-1, 9]]),
        (["Acme Corp"], [[0, 9999]]),
        (["Acme Corp"], [[9, 0]]),
        (["Acme Corp"], [[0, 0]]),
        (["Acme Corp"], [[False, 9]]),
        (["Acme Corp"], [[0.0, 9]]),
        (["Acme Corp"], [[0]]),
        (["Acme Corp"], [None]),
        (["Acme Corp"], None),
        ([123], [[0, 9]]),
    ],
)
async def test_invalid_evidence_fails_before_qualification(
    repeated_mentions, values, spans
):
    example, candidates = repeated_mentions
    candidates["answers"]["companies"].update(value=values, spans=spans)
    extraction_binding = MagicMock()
    extraction_binding.run = AsyncMock(return_value=candidates)
    decision_binding = MagicMock()
    decision_binding.run = AsyncMock()
    with pytest.raises(ValueError):
        await extract_announcements(
            text=example["text"],
            extraction_model=CloudflareWorkersAIExtractionModel(
                binding=extraction_binding
            ),
            decision_model=CloudflareWorkersAIDecisionModel(binding=decision_binding),
        )
    decision_binding.run.assert_not_awaited()
    with pytest.raises(AssertionError) as error:
        assert_extraction_result(
            candidates,
            {"state": example["text"], "fields": {"companies": {"type": "list"}}},
            {"companies": ["different expected answer"]},
        )
    assert not isinstance(error.value, ExtractionQualityError)


def test_empty_list_and_duplicate_values():
    assert group_list_spans(text="", values=[], spans=[]) == {}
    assert group_list_spans(
        text="Acme Corp", values=["Acme Corp", "Acme Corp"], spans=[[0, 9]]
    ) == {"Acme Corp": [[0, 9]]}
