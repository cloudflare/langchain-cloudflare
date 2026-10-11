"""Shared fixtures and assertions for REST and Worker decision-model results."""

# MARK: - IMPORTS
import base64
from pathlib import Path

import pytest

# MARK: - MEDIA FIXTURES
RED_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP4z8BA"
    "EiJN9aiGUQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII="
)
OMNI_MEDIA_CASES = [
    f"omni_{case}-{encoding}"
    for case in (
        "audio_open",
        "audio_closed",
        "video_square",
        "video_circle",
        "video_sound_square_open",
        "video_sound_circle_closed",
        "mixed",
    )
    for encoding in ("data_url", "base64")
]


def decision_media_case(case_id: str) -> tuple[dict, dict]:
    """Attach fixed media without putting its expected answers in the state."""
    case, encoding = case_id.removeprefix("omni_").split("-")
    media_dir = Path(__file__).parent / "fixtures" / "clef_omni"

    def embedded(content_type: str, encoded: str):
        if encoding == "data_url":
            return f"data:{content_type};base64,{encoded}"
        return {"content_type": content_type, "base64": encoded}

    def media(filename: str, content_type: str):
        encoded = base64.b64encode((media_dir / filename).read_bytes()).decode("ascii")
        return embedded(content_type, encoded)

    speech = {
        "type": "choice",
        "instructions": "Which statement is spoken in the audio recording?",
        "criteria": {"open": "The door is open.", "closed": "The door is closed."},
    }
    shape = {
        "type": "choice",
        "instructions": "Which geometric shape is shown in the video?",
        "criteria": {
            "square": "A square",
            "circle": "A circle",
            "triangle": "A triangle",
        },
    }
    payload = {"state": "Evaluate the attached media.", "questions": {}}
    expected = {}
    if case.startswith("audio_"):
        spoken = case.removeprefix("audio_")
        payload["audio"] = [media(f"door_{spoken}.wav", "audio/wav")]
        payload["questions"]["speech"] = speech
        expected["speech"] = spoken
    elif case.startswith("video_"):
        clip = case.removeprefix("video_").removeprefix("sound_")
        payload["videos"] = [media(f"{clip}.mp4", "video/mp4")]
        payload["questions"]["shape"] = shape
        expected["shape"] = clip.split("_")[0]
        if case.startswith("video_sound_"):
            speech["instructions"] = (
                "Which statement is spoken in the video's soundtrack?"
            )
            payload["questions"]["speech"] = speech
            expected["speech"] = clip.split("_")[1]
    else:
        payload["images"] = [embedded("image/png", RED_IMAGE)]
        payload["audio"] = [media("door_open.wav", "audio/wav")]
        payload["videos"] = [media("circle.mp4", "video/mp4")]
        payload["questions"] = {
            "speech": speech,
            "shape": shape,
            "color": {
                "type": "choice",
                "instructions": "What is the dominant color of the attached image?",
                "criteria": {"red": "Red", "green": "Green", "blue": "Blue"},
            },
        }
        expected = {"speech": "open", "shape": "circle", "color": "red"}
    return payload, expected


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
            message = (
                f"{key}: expected {expected[key]!r}, got {answer['choice']!r}; "
                f"probabilities={probabilities}"
            )
            assert answer["choice"] == expected[key], message
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
