"""Tests for Roboflow response normalization.

No API key and no network. Response-shape handling is the part most likely to
break — workflow outputs nest differently depending on how the workflow was
authored — so it is worth pinning down with synthetic payloads.
"""

from __future__ import annotations

import pytest

from src import config
from src.detect.roboflow_client import (
    _find_predictions,
    _scan,
    _to_boxes,
    is_vehicle_label,
)


def pred(x, y, w, h, cls="car", conf=0.9) -> dict:
    return {"x": x, "y": y, "width": w, "height": h, "class": cls, "confidence": conf}


# --- locating predictions in varied response shapes ------------------------

def test_finds_predictions_in_direct_infer_response():
    """Plain infer(): {"predictions": [...], "image": {...}}"""
    response = {"predictions": [pred(100, 50, 40, 20)], "image": {"width": 352, "height": 240}}
    found = _find_predictions(response)
    assert found is not None and len(found) == 1


def test_finds_predictions_in_workflow_list_response():
    """run_workflow() returns a list, one entry per input image."""
    response = [{"vehicle_boxes": {"predictions": [pred(10, 10, 20, 20)]}}]
    found = _find_predictions(response)
    assert found is not None and len(found) == 1


def test_finds_predictions_under_unexpected_output_key():
    """The workflow author may name the output anything; we still find it."""
    response = [{"some_other_name": {"predictions": [pred(1, 2, 3, 4), pred(5, 6, 7, 8)]}}]
    found = _find_predictions(response)
    assert found is not None and len(found) == 2


def test_finds_bare_prediction_list():
    response = {"vehicle_boxes": [pred(10, 10, 20, 20)]}
    found = _find_predictions(response)
    assert found is not None and len(found) == 1


def test_empty_predictions_list_is_found_not_missed():
    """An empty result is a valid answer and must not read as 'not found'."""
    assert _find_predictions({"predictions": []}) == []


def test_returns_none_when_no_predictions_present():
    assert _find_predictions({"status": "ok", "message": "nothing here"}) is None


def test_does_not_recurse_forever_on_deep_junk():
    node: dict = {}
    cursor = node
    for _ in range(50):
        cursor["nested"] = {}
        cursor = cursor["nested"]
    assert _find_predictions(node) is None


def test_scan_pulls_nested_inference_id():
    response = [{"vehicle_boxes": {"predictions": []}, "inference_id": "abc-123"}]
    assert _scan(response, "inference_id") == "abc-123"


def test_scan_returns_none_when_absent():
    assert _scan({"predictions": []}, "inference_id") is None


# --- converting predictions to boxes ---------------------------------------

def test_center_form_converted_to_corner_form():
    boxes = _to_boxes([pred(100, 50, 40, 20)])
    assert len(boxes) == 1
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (80, 40, 120, 60)


def test_low_confidence_predictions_dropped():
    below = config.MIN_BOX_CONFIDENCE - 0.05
    assert _to_boxes([pred(10, 10, 20, 20, conf=below)]) == []


def test_confidence_at_threshold_is_kept():
    boxes = _to_boxes([pred(10, 10, 20, 20, conf=config.MIN_BOX_CONFIDENCE)])
    assert len(boxes) == 1


def test_non_vehicle_classes_dropped():
    """Universe models often emit people, traffic lights and signs."""
    predictions = [
        pred(10, 10, 20, 20, cls="car"),
        pred(30, 30, 10, 10, cls="person"),
        pred(50, 50, 10, 10, cls="traffic light"),
        pred(70, 70, 30, 20, cls="truck"),
    ]
    labels = {b.label for b in _to_boxes(predictions)}
    assert labels == {"car", "truck"}


def test_class_matching_is_case_insensitive():
    boxes = _to_boxes([pred(10, 10, 20, 20, cls="CAR"), pred(40, 10, 20, 20, cls="Truck")])
    assert len(boxes) == 2


def test_malformed_prediction_skipped_without_killing_the_batch():
    """One bad record must not lose the good ones in the same frame."""
    predictions = [
        pred(10, 10, 20, 20),
        {"x": "not-a-number", "y": 1, "width": 2, "height": 3, "class": "car", "confidence": 0.9},
        {"class": "car", "confidence": 0.9},  # missing geometry entirely
        pred(80, 80, 20, 20),
    ]
    assert len(_to_boxes(predictions)) == 2


def test_missing_class_defaults_to_vehicle():
    boxes = _to_boxes([{"x": 10, "y": 10, "width": 20, "height": 20, "confidence": 0.9}])
    assert len(boxes) == 1 and boxes[0].label == "vehicle"


def test_empty_input_yields_empty_output():
    assert _to_boxes([]) == []


# --- vehicle-class matching ------------------------------------------------
#
# Regression coverage for a silent failure: the original exact-match set held
# "truck", "pickup" and "vehicle", while the deployed model emits
# "pickup-truck", "semi-trailer" and "vehicles". Every one was dropped, and a
# dropped box leaves no trace — it just never reaches the tracker. On NYC local
# streets "semi-trailer" is the label for box and delivery trucks, i.e. the
# most common double-parker of all.

@pytest.mark.parametrize("label", [
    "car", "truck", "bus", "motorcycle",          # COCO vocabulary
    "pickup-truck", "semi-trailer", "vehicles",   # vehicle-detection-bz0yu
    "Pickup_Truck", "  CAR  ", "delivery van", "mini-bus",
])
def test_vehicle_labels_are_kept(label):
    assert is_vehicle_label(label)


@pytest.mark.parametrize("label", [
    "person", "traffic light", "traffic sign", "fire hydrant",
    "bicycle", "dog", "", "   ",
])
def test_non_vehicle_labels_are_dropped(label):
    assert not is_vehicle_label(label)


@pytest.mark.parametrize("label", ["bus stop", "truck stop sign", "bus lane"])
def test_veto_words_beat_a_vehicle_word_in_the_same_label(label):
    """"bus stop" contains "bus" but is street furniture, not a vehicle."""
    assert not is_vehicle_label(label)


def test_semi_trailer_survives_the_full_normalization_path():
    """End-to-end through _to_boxes, not just the predicate."""
    boxes = _to_boxes([pred(50, 50, 40, 30, cls="semi-trailer")])
    assert [b.label for b in boxes] == ["semi-trailer"]
