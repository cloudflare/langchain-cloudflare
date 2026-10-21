"""Shared native extraction inputs and assertions for REST and Worker tests."""

# MARK: - IMPORTS
from examples.workers.src.system_one import group_list_spans
from tests.press_release_examples import PRESS_RELEASE_EXAMPLES
from tests.vision_helpers import create_labeled_test_image_base64

# MARK: - FIXTURES
INVOICE_REQUEST = {
    "state": (
        "Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30. "
        "4 pallets at $250.00 each, shipping $80.00. Contact: Dana Ruiz, Priya Shah."
    ),
    "fields": {
        "invoice_number": {"type": "text"},
        "due_date": {
            "type": "date",
            "instructions": "Issue date plus the net payment days",
        },
        "total": {
            "type": "number",
            "decimals": 2,
            "unit": "USD",
            "instructions": "Amount due including shipping",
        },
        "contacts": {"type": "list", "items": "text"},
        "po_number": {"type": "text", "instructions": "Purchase order number"},
    },
    "options": {"return_state_text": True},
}
INVOICE_EXPECTED = {
    "invoice_number": "INV-2291",
    "due_date": "2026-10-02",
    "total": 1080,
    "contacts": ["Dana Ruiz", "Priya Shah"],
    "po_number": None,
}


def extraction_case(case: str) -> tuple[dict, dict]:
    """Reuse public source text; never supply expected values in instructions."""
    if case == "invoice":
        return INVOICE_REQUEST, INVOICE_EXPECTED
    if case in ("image_data_url", "image_base64"):
        encoded = create_labeled_test_image_base64()
        image = (
            f"data:image/png;base64,{encoded}"
            if case == "image_data_url"
            else {"content_type": "image/png", "base64": encoded}
        )
        return (
            {
                "state": "",
                "images": [image],
                "fields": {
                    "ticker": {
                        "type": "text",
                        "instructions": "Ticker printed under TICKER",
                    },
                    "timeframe": {
                        "type": "text",
                        "instructions": "Timeframe printed under TIMEFRAME",
                    },
                    "date": {"type": "date", "instructions": "Date printed under DATE"},
                },
                "options": {"return_state_text": True},
            },
            {"ticker": "ACME", "timeframe": "Q4", "date": "2026-06-21"},
        )
    if case == "press_release_company_list":
        example = PRESS_RELEASE_EXAMPLES["press_release_partnership"]
        return (
            {
                "state": example["text"],
                "fields": {
                    "companies": {
                        "type": "list",
                        "items": "text",
                        "instructions": (
                            "Extract each distinct company's full name exactly "
                            "once, in first-mention order. Include legal suffixes "
                            "such as Corp or Inc. Exclude stock exchanges, ticker "
                            "symbols, shortened names and possessive references."
                        ),
                    }
                },
            },
            {"companies": list(example["roles"])},
        )
    if case.endswith("_sentence"):
        example = PRESS_RELEASE_EXAMPLES[case.removesuffix("_sentence")]
        return (
            {
                "state": example["text"],
                "fields": {
                    "announcement_sentence": {
                        "type": "text",
                        # This cutoff is specific to the two fixtures.
                        "null_threshold": 0.01,
                        "instructions": (
                            "Copy the entire announcement sentence verbatim, "
                            "from its first character through its ending "
                            "punctuation, not just a company name or phrase. "
                            "Eligible announcements are new partnerships, "
                            "investments, regulatory approvals, milestones, "
                            "events or mergers/acquisitions. Return null if "
                            "there is no eligible announcement; routine "
                            "earnings alone do not qualify."
                        ),
                    }
                },
            },
            {
                "announcement_sentence": (
                    example["sentences"][0] if example["announcement"] else None
                )
            },
        )
    example = PRESS_RELEASE_EXAMPLES[case]
    fields = {
        "announcing_company": {
            "type": "text",
            "instructions": (
                "Full company name of the company making the announcement, "
                "as written at its first mention. Include the company suffix."
            ),
        },
        "partner_company": {
            "type": "text",
            "instructions": (
                "Full company name of the other participant in the partnership, "
                "including the company suffix. Return null if no partnership "
                "counterparty is stated."
            ),
        },
    }
    companies = list(example["roles"])
    expected = {
        "announcing_company": companies[0],
        "partner_company": companies[1] if len(companies) > 1 else None,
    }
    for i, (name, ticker) in enumerate(example["tickers"].items()):
        key = f"ticker_{i}"
        fields[key] = {
            "type": "text",
            "instructions": (
                f"Stock ticker explicitly written for {name}; "
                "return null if absent. Do not infer it from prior knowledge."
            ),
        }
        expected[key] = None if ticker == "none" else ticker
    return {"state": example["text"], "fields": fields}, expected


# MARK: - ASSERTIONS
class ExtractionQualityError(AssertionError):
    """A well-formed native answer disagrees with the fixed expected value."""


def assert_extraction_result(result: dict, payload: dict, expected: dict) -> None:
    """Require correct values, source evidence, uncertainty and native usage."""
    assert result["model"] == "clairic"
    assert result["truncated"] is False
    assert set(result["answers"]) == set(payload["fields"])
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] == 0
    if payload.get("images"):
        assert result["usage"]["image_tokens"] > 0
        if not payload["state"]:
            assert result["usage"]["state_tokens"] == 0
    if payload.get("options", {}).get("return_state_text"):
        assert result["state_text"] == payload["state"]
    mismatches = []
    for key, field in payload["fields"].items():
        answer = result["answers"][key]
        assert answer["type"] == field["type"]
        if answer["value"] != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, got {answer}")
        assert 0 <= answer["confidence"] <= 1
        if field["type"] != "list":
            assert 0 <= answer["p_null"] <= 1
            threshold = field.get("null_threshold", 0.5)
            assert (answer["value"] is None) == (answer["p_null"] > threshold)
        if field["type"] == "list":
            try:
                group_list_spans(
                    text=payload["state"],
                    values=answer["value"],
                    spans=answer["spans"],
                )
            except ValueError as error:
                raise AssertionError(str(error)) from error
        elif answer["value"] is not None and field["type"] == "text":
            assert isinstance(answer["value"], str)
            if answer.get("source") == "image":
                assert payload.get("images"), "Image answers require image input"
                assert answer["span"] is None
                if "spell_confidence" in answer:
                    assert 0 <= answer["spell_confidence"] <= 1
            else:
                start, end = answer["span"]
                assert 0 <= start < end <= len(payload["state"])
                assert payload["state"][start:end] == answer["value"]
    if mismatches:
        raise ExtractionQualityError("; ".join(mismatches))


# MARK: - COMBINED RESULT ASSERTIONS
def assert_system_one_result(result: dict, example: dict) -> None:
    """Require the original expected nested result and native stage evidence."""
    expected = {"announcements": []}
    if example["announcement"]:
        expected["announcements"] = [
            {
                "type": example["type"],
                "context": example["sentences"][int(example["context"][1:])],
                "entities": [
                    {
                        "name": name,
                        "role": role,
                        "ticker": (
                            None
                            if example["tickers"][name] == "none"
                            else example["tickers"][name]
                        ),
                    }
                    for name, role in example["roles"].items()
                ],
            }
        ]
    assert result["extracted"] == expected, result
    stages = result["stages"]
    assert stages["candidates"]["model"] == "clairic"
    assert stages["decisions"]["model"] == "clef"
    assert (stages["decisions"]["answers"]["is_announcement"]["noul"] > 0.5) == (
        example["announcement"]
    )
    for stage in stages.values():
        assert stage["usage"]["input_tokens"] > 0
        assert stage["usage"]["output_tokens"] == 0
