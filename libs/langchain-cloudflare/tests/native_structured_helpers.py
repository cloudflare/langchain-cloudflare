"""Public inputs and assertions shared by native structured REST/Worker tests."""

# MARK: - IMPORTS
from examples.workers.src.models import (
    NativeImageFacts,
    NativeInvoice,
    NativeInvoiceReview,
    NativeSupportDecision,
)
from tests.extraction_helpers import INVOICE_EXPECTED, INVOICE_REQUEST, extraction_case

# MARK: - CASES
NATIVE_STRUCTURED_CASES = {
    "invoice": {
        "schema": NativeInvoice,
        "state": INVOICE_REQUEST["state"],
        "expected": INVOICE_EXPECTED,
        "stages": {"extraction"},
    },
    "decision": {
        "schema": NativeSupportDecision,
        "state": "Checkout has been failing for every customer for the last hour.",
        "expected": {"urgent": True, "team": "technical"},
        "stages": {"decision"},
    },
    "mixed": {
        "schema": NativeInvoiceReview,
        "state": INVOICE_REQUEST["state"],
        "expected": {
            "invoice": INVOICE_EXPECTED,
            "paid": False,
            "department": "billing",
        },
        "stages": {"extraction", "decision"},
    },
}

NATIVE_AGENT_CASES = {
    name: {**case, "messages": [{"role": "user", "content": case["state"]}]}
    for name, case in NATIVE_STRUCTURED_CASES.items()
}
for encoding in ("image_data_url", "image_base64"):
    image_payload, expected = extraction_case(encoding)
    image = image_payload["images"][0]
    block = (
        {"type": "image_url", "image_url": {"url": image}}
        if isinstance(image, str)
        else {
            "type": "image",
            "source_type": "base64",
            "mime_type": image["content_type"],
            "data": image["base64"],
        }
    )
    NATIVE_AGENT_CASES[encoding] = {
        "schema": NativeImageFacts,
        "messages": [{"role": "user", "content": [block]}],
        "expected": expected,
        "stages": {"extraction"},
    }


# MARK: - ASSERTIONS
def assert_native_structured_result(result: dict, case: dict) -> None:
    assert result["parsing_error"] is None, result
    assert result["parsed"] == case["expected"], result
    assert set(result["raw"]) == case["stages"]
    for stage in result["raw"].values():
        assert stage["usage"]["input_tokens"] > 0
        assert stage["usage"]["output_tokens"] == 0
        assert stage.get("truncated", False) is False


def assert_native_agent_result(result: dict, case: dict) -> None:
    assert result["schema_type"] == case["schema"].__name__
    assert_native_structured_result(
        {
            "parsed": result["structured_response"],
            "raw": result["raw"],
            "parsing_error": None,
        },
        case,
    )
