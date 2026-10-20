"""Public inputs and assertions shared by native structured REST/Worker tests."""

# MARK: - IMPORTS
from examples.workers.src.models import (
    NativeInvoice,
    NativeInvoiceReview,
    NativeSupportDecision,
)
from tests.extraction_helpers import INVOICE_EXPECTED, INVOICE_REQUEST

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


# MARK: - ASSERTIONS
def assert_native_structured_result(result: dict, case: dict) -> None:
    assert result["parsing_error"] is None, result
    assert result["parsed"] == case["expected"], result
    assert set(result["raw"]) == case["stages"]
    for stage in result["raw"].values():
        assert stage["usage"]["input_tokens"] > 0
        assert stage["usage"]["output_tokens"] == 0
        assert stage.get("truncated", False) is False
