"""Shared assertions for real REST and Worker decision-model results."""

# MARK: - IMPORTS
import pytest


# MARK: - ASSERTIONS
def assert_decision_result(
    result: dict, model: str, payload: dict, expected: dict
) -> None:
    """Check native answers, probability distributions, and usage on both paths."""
    assert result["model"] == model.rsplit("/", 1)[-1]
    assert set(result["answers"]) == set(payload["questions"])
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] >= 0
    for key, question in payload["questions"].items():
        answer = result["answers"][key]
        assert answer["type"] == question["type"]
        if question["type"] == "noul":
            assert 0 <= answer["noul"] <= 1
            assert (answer["noul"] > 0.5) == expected[key]
            continue
        probabilities = answer["probabilities"]
        assert all(0 <= p <= 1 for p in probabilities.values())
        assert sum(probabilities.values()) == pytest.approx(1, abs=5e-4)
        assert 0 <= answer["confidence"] <= 1
        if question["type"] == "choice":
            assert set(probabilities) == set(question["criteria"])
            assert answer["choice"] == max(probabilities, key=probabilities.get)
            assert answer["choice"] == expected[key]
            if key == "context" and answer["choice"] != "none":
                context = question["criteria"][answer["choice"]]
                assert context in payload["state"]["press_release"]
        else:
            legend = {str(i): level for i, level in enumerate(question["criteria"])}
            assert answer["legend"] == legend
            assert set(probabilities) == set(legend)
            assert 0 <= answer["score"] <= len(legend) - 1
            expected = sum(int(level) * p for level, p in probabilities.items())
            assert answer["score"] == pytest.approx(expected, abs=2e-3)
