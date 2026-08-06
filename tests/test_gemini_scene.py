"""Tests for Gemini verdict parsing and frame preparation.

No API key and no network — the live check at the bottom skips itself without
credentials. Everything worth testing here is the code that stands between an
LLM and the pipeline, so the cases are deliberately hostile: the model is an
external system that can return anything, and this module's whole job is that
nothing it returns can take a sweep down.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src import config
from src.analyze.gemini_scene import (
    FALLBACK_FLOW_STATE,
    FLOW_STATES,
    SceneVerdict,
    clamp_confidence,
    normalize_flow_state,
    upscale_for_scene,
    verdict_from_payload,
)


def jpeg(width: int, height: int) -> bytes:
    """A real JPEG of the given size — Pillow must actually be able to open it."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 90, 95)).save(buffer, format="JPEG")
    return buffer.getvalue()


def size_of(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.size


# --- the happy path --------------------------------------------------------

def test_well_formed_response_becomes_a_verdict():
    payload = {
        "flow_state": "jammed",
        "confidence": 0.82,
        "reason": "Eight cars queued nose-to-tail across both lanes.",
        "notable": "construction blocking right lane",
    }
    verdict = verdict_from_payload(payload)

    assert isinstance(verdict, SceneVerdict)
    assert verdict.flow_state == "jammed"
    assert verdict.confidence == pytest.approx(0.82)
    assert verdict.reason == "Eight cars queued nose-to-tail across both lanes."
    assert verdict.notable == "construction blocking right lane"
    assert verdict.raw == payload


@pytest.mark.parametrize("state", FLOW_STATES)
def test_every_allowed_flow_state_survives_untouched(state):
    assert verdict_from_payload({"flow_state": state, "confidence": 0.7}).flow_state == state


def test_verdict_is_frozen():
    """Observations are built from these; a mutated verdict would be untraceable."""
    verdict = verdict_from_payload({"flow_state": "clear", "confidence": 0.9})
    with pytest.raises(Exception):
        verdict.flow_state = "jammed"  # type: ignore[misc]


# --- flow states the model was not supposed to produce ---------------------
#
# The schema constrains generation to three values, but a schema is not a
# guarantee: the SDK abandons validation silently, and truncated or
# safety-filtered responses arrive as whatever fragment made it out. Every one
# of these must produce a usable verdict rather than an exception, because this
# runs unattended across a 250-camera sweep.

@pytest.mark.parametrize("bad", ["", "   ", "HEAVY TRAFFIC AND ALSO", "unknown", "3", "-"])
def test_unrecognized_flow_state_does_not_raise(bad):
    verdict = verdict_from_payload({"flow_state": bad, "confidence": 0.9})
    assert verdict.flow_state in FLOW_STATES


def test_unrecognized_flow_state_falls_back_and_loses_its_confidence():
    """Zero confidence is what routes the frame back to the geometric verdict."""
    verdict = verdict_from_payload({"flow_state": "somewhat spicy", "confidence": 0.95})
    assert verdict.flow_state == FALLBACK_FLOW_STATE
    assert verdict.confidence == 0.0
    assert verdict.confidence < config.GEMINI_MIN_CONFIDENCE


@pytest.mark.parametrize("value", [None, 42, ["jammed"], {"state": "jammed"}])
def test_non_string_flow_state_does_not_raise(value):
    verdict = verdict_from_payload({"flow_state": value, "confidence": 0.5})
    assert verdict.flow_state == FALLBACK_FLOW_STATE
    assert verdict.confidence == 0.0


@pytest.mark.parametrize("given,expected", [
    ("heavy", "jammed"),
    ("congested", "jammed"),
    ("gridlock", "jammed"),
    ("free flowing", "clear"),
    ("free_flowing", "clear"),
    ("light", "clear"),
    ("busy", "moderate"),
])
def test_unambiguous_paraphrases_are_recovered_not_discarded(given, expected):
    """"heavy" plainly means jammed; throwing that away loses a good verdict."""
    verdict = verdict_from_payload({"flow_state": given, "confidence": 0.8})
    assert verdict.flow_state == expected
    assert verdict.confidence == pytest.approx(0.8)  # recovered, so still trusted


@pytest.mark.parametrize("given", ["Clear", "  JAMMED  ", "Moderate"])
def test_case_and_whitespace_are_normalized(given):
    verdict = verdict_from_payload({"flow_state": given, "confidence": 0.6})
    assert verdict.flow_state == given.strip().lower()
    assert verdict.confidence == pytest.approx(0.6)


# --- confidence clamping ---------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    (1.7, 1.0),
    (-0.2, 0.0),
    (100, 1.0),       # answered on a 0-100 scale
    (0.0, 0.0),
    (1.0, 1.0),
    (0.55, 0.55),
])
def test_confidence_is_clamped_into_range(given, expected):
    assert clamp_confidence(given) == pytest.approx(expected)


def test_out_of_range_confidence_is_clamped_on_the_verdict_too():
    assert verdict_from_payload({"flow_state": "clear", "confidence": 1.7}).confidence == 1.0
    assert verdict_from_payload({"flow_state": "clear", "confidence": -0.2}).confidence == 0.0


@pytest.mark.parametrize("junk", [None, "high", "", [], {}, float("nan")])
def test_unusable_confidence_becomes_zero(junk):
    assert clamp_confidence(junk) == 0.0


def test_numeric_string_confidence_is_accepted():
    """JSON is typed, but a model emitting "0.7" should not cost us the verdict."""
    assert clamp_confidence("0.7") == pytest.approx(0.7)


# --- missing and malformed fields ------------------------------------------

def test_missing_notable_defaults_to_empty_string():
    verdict = verdict_from_payload({"flow_state": "clear", "confidence": 0.9, "reason": "Empty road."})
    assert verdict.notable == ""


def test_missing_reason_defaults_to_empty_string():
    assert verdict_from_payload({"flow_state": "clear", "confidence": 0.9}).reason == ""


@pytest.mark.parametrize("junk", [None, 12, ["a", "b"], {"text": "x"}])
def test_non_string_text_fields_become_empty(junk):
    verdict = verdict_from_payload({"flow_state": "clear", "confidence": 0.9, "reason": junk, "notable": junk})
    assert verdict.reason == "" and verdict.notable == ""


def test_completely_empty_payload_yields_a_zero_confidence_verdict():
    verdict = verdict_from_payload({})
    assert verdict.flow_state in FLOW_STATES
    assert verdict.confidence == 0.0
    assert verdict.reason == "" and verdict.notable == ""


@pytest.mark.parametrize("payload", [None, "jammed", 7, ["jammed"]])
def test_non_dict_payload_is_absorbed(payload):
    """json.loads can legally return a scalar or a list; neither may raise here."""
    verdict = verdict_from_payload(payload)
    assert verdict.confidence == 0.0
    assert verdict.raw == {"response": payload}


def test_multiline_reason_is_collapsed_to_one_line():
    verdict = verdict_from_payload({
        "flow_state": "moderate", "confidence": 0.7,
        "reason": "Cars are\n  queued\tat the light.",
    })
    assert verdict.reason == "Cars are queued at the light."


def test_runaway_reason_is_truncated():
    verdict = verdict_from_payload({
        "flow_state": "moderate", "confidence": 0.7, "reason": "word " * 500,
    })
    assert len(verdict.reason) <= 300


def test_raw_payload_is_preserved_verbatim_for_eval():
    """Phase 2 debugs disagreements from raw; normalization must not erase it."""
    payload = {"flow_state": "heavy", "confidence": 7, "extra": "model chatter"}
    assert verdict_from_payload(payload).raw == payload


# --- normalize_flow_state's second return value ----------------------------

def test_normalize_reports_whether_it_understood_the_value():
    assert normalize_flow_state("jammed") == ("jammed", True)
    assert normalize_flow_state("heavy") == ("jammed", True)
    assert normalize_flow_state("banana") == (FALLBACK_FLOW_STATE, False)


def test_fallback_state_is_itself_a_valid_state():
    """Downstream code indexes on flow_state; the fallback cannot be a new value."""
    assert FALLBACK_FLOW_STATE in FLOW_STATES


# --- upscaling -------------------------------------------------------------

def test_native_nycdot_frame_is_upscaled_to_the_target_long_edge():
    """Every NYC DOT frame is exactly 352x240."""
    width, height = size_of(upscale_for_scene(jpeg(352, 240)))
    assert max(width, height) == config.SCENE_MIN_DIM_PX


def test_aspect_ratio_is_preserved():
    original = 352 / 240
    width, height = size_of(upscale_for_scene(jpeg(352, 240)))
    assert width / height == pytest.approx(original, rel=0.01)


def test_portrait_frame_scales_on_its_long_edge_too():
    """The long edge is the target, so the short edge never overshoots it."""
    width, height = size_of(upscale_for_scene(jpeg(240, 352)))
    assert max(width, height) == config.SCENE_MIN_DIM_PX
    assert min(width, height) < config.SCENE_MIN_DIM_PX


def test_already_large_image_is_not_downscaled():
    """Downscaling would destroy the very detail this function exists to keep."""
    source = jpeg(1920, 1080)
    result = upscale_for_scene(source)
    assert size_of(result) == (1920, 1080)
    assert result == source  # returned untouched, not re-encoded


def test_image_exactly_at_the_target_is_left_alone():
    source = jpeg(config.SCENE_MIN_DIM_PX, 480)
    assert upscale_for_scene(source) == source


def test_upscale_honours_an_explicit_target():
    width, height = size_of(upscale_for_scene(jpeg(352, 240), min_dim_px=1408))
    assert max(width, height) == 1408


def test_upscaled_output_is_a_decodable_jpeg():
    """It goes straight to the API; a broken encode would only fail over the wire."""
    result = upscale_for_scene(jpeg(352, 240))
    assert result.startswith(b"\xff\xd8")
    with Image.open(io.BytesIO(result)) as img:
        assert img.format == "JPEG" and img.mode == "RGB"


def test_greyscale_frame_is_converted_rather_than_crashing():
    """A camera in night mode can return a single-channel JPEG."""
    buffer = io.BytesIO()
    Image.new("L", (352, 240), 40).save(buffer, format="JPEG")
    with Image.open(io.BytesIO(upscale_for_scene(buffer.getvalue()))) as img:
        assert img.mode == "RGB"


# --- live smoke test -------------------------------------------------------
#
# Skipped without credentials so a fresh clone runs the suite clean. This is a
# reachability and shape check, not an accuracy check — flow state on a live
# camera has no fixed ground truth to assert against, so it can only pin down
# that a real round trip still produces a schema-valid verdict.

@pytest.mark.skipif(not config.GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_live_classification_returns_a_valid_verdict():
    from src.analyze.gemini_scene import SceneClassifier
    from src.sources import nycdot

    cameras = nycdot.demo_cameras(limit=1)
    if not cameras:
        pytest.skip("no NYC DOT cameras available")
    frame = nycdot.fetch_snapshot(cameras[0])
    if frame is None:
        pytest.skip("camera offline")

    # The free tier allows 20 requests/day/model, and Gemini 3 models return 503
    # under load. Neither says anything about this code, and a red suite that
    # means "someone else used the quota" trains people to ignore the suite.
    try:
        verdict = SceneClassifier().classify(frame)
    except Exception as exc:  # noqa: BLE001 — narrow on message, not type
        if any(code in str(exc) for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
            pytest.skip(f"Gemini quota or availability: {exc}")
        raise

    assert verdict.flow_state in FLOW_STATES
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.reason
