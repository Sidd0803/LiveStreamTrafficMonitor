"""Tracker and heuristic tests using synthetic boxes.

These run with no API keys and no network, so they stay useful as the fast
inner loop while tuning thresholds in Phase 2.
"""

from __future__ import annotations

import pytest

from src import config
from src.detect.boxes import Box, iou
from src.detect.heuristic import find_candidates, point_in_polygon
from src.detect.tracker import CameraTracker

FRAME_W, FRAME_H = 352, 240


def box(x1, y1, x2, y2, label="car", conf=0.9) -> Box:
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, label=label, confidence=conf)


# --- geometry --------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    b = box(10, 10, 50, 50)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou(box(0, 0, 10, 10), box(100, 100, 110, 110)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150.
    assert iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_from_center_matches_corner_form():
    b = Box.from_center(x=100, y=50, width=40, height=20)
    assert (b.x1, b.y1, b.x2, b.y2) == (80, 40, 120, 60)


def test_padded_clips_to_frame():
    b = box(5, 5, 20, 20).padded(30, FRAME_W, FRAME_H)
    assert (b.x1, b.y1) == (0, 0)
    assert b.x2 <= FRAME_W and b.y2 <= FRAME_H


# --- tracker ---------------------------------------------------------------

def test_fixed_box_becomes_stationary():
    """A vehicle that does not move must flag stationary."""
    tracker = CameraTracker("cam-fixed")
    parked = box(100, 100, 140, 130)

    for _ in range(config.STATIONARY_POLLS):
        tracker.update([parked])

    stationary = tracker.stationary_tracks()
    assert len(stationary) == 1
    assert stationary[0].stationary_polls >= config.STATIONARY_POLLS


def test_moving_box_never_becomes_stationary():
    """A vehicle translating each poll must NOT flag stationary."""
    tracker = CameraTracker("cam-moving")

    for i in range(config.STATIONARY_POLLS * 3):
        # 45px per poll on a 40px-wide box: no overlap, so no match.
        tracker.update([box(10 + i * 45, 100, 50 + i * 45, 130)])

    assert tracker.stationary_tracks() == []


def test_slow_drift_below_threshold_breaks_the_track():
    """Movement large enough to drop IoU under threshold resets the count."""
    tracker = CameraTracker("cam-drift")
    for i in range(config.STATIONARY_POLLS * 2):
        # 25px/poll on a 40px box -> IoU ~0.23, below the 0.6 threshold.
        tracker.update([box(100 + i * 25, 100, 140 + i * 25, 130)])
    assert tracker.stationary_tracks() == []


def test_single_missed_detection_is_tolerated():
    """One dropped frame must not destroy a long-lived track."""
    tracker = CameraTracker("cam-flicker")
    parked = box(100, 100, 140, 130)

    for _ in range(config.STATIONARY_POLLS):
        tracker.update([parked])
    assert len(tracker.stationary_tracks()) == 1

    tracker.update([])                      # detector drops it for one poll
    assert tracker.stationary_tracks() == []  # not reported while missing
    assert len(tracker.tracks) == 1           # but not forgotten either

    tracker.update([parked])                # reacquired
    assert len(tracker.stationary_tracks()) == 1


def test_track_forgotten_after_max_misses():
    tracker = CameraTracker("cam-gone")
    tracker.update([box(100, 100, 140, 130)])
    for _ in range(config.TRACK_MAX_MISSES + 1):
        tracker.update([])
    assert tracker.tracks == []


def test_two_vehicles_get_distinct_tracks():
    tracker = CameraTracker("cam-two")
    a, b = box(10, 100, 50, 130), box(200, 100, 240, 130)
    tracker.update([a, b])
    tracker.update([a, b])
    assert len({t.id for t in tracker.tracks}) == 2


# --- heuristic -------------------------------------------------------------

def test_point_in_polygon_basic():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 50), square)
    assert not point_in_polygon((150, 50), square)


def _stationary_tracker(boxes: list[Box], camera_id: str) -> CameraTracker:
    tracker = CameraTracker(camera_id)
    for _ in range(config.STATIONARY_POLLS):
        tracker.update(boxes)
    return tracker


def test_congestion_suppresses_all_candidates():
    """A red-light queue: everything stationary, so nothing is double parked."""
    queue = [box(20 + i * 45, 90, 60 + i * 45, 120) for i in range(6)]
    tracker = _stationary_tracker(queue, "cam-redlight")

    candidates = find_candidates("cam-redlight", tracker.tracks, FRAME_W, FRAME_H)
    assert candidates == [], "congested frame must not produce candidates"


def test_curbside_vehicles_are_not_flagged():
    """Stationary vehicles in the near-field curb band are legally parked."""
    # Bottom band starts at 240 * (1 - 0.18) = ~197px.
    curb = [box(20, 205, 70, 235), box(90, 205, 140, 235)]
    tracker = CameraTracker("cam-curb")
    # Add moving traffic so the congestion guard does not trip.
    for i in range(config.STATIONARY_POLLS):
        tracker.update(curb + [box(150 + i * 40, 80, 190 + i * 40, 110),
                               box(10 + i * 50, 60, 45 + i * 50, 85)])

    candidates = find_candidates("cam-curb", tracker.tracks, FRAME_W, FRAME_H)
    assert candidates == []


def test_lone_stopped_vehicle_needs_a_neighbour():
    """One stopped car in an empty street is not the double-parking pattern."""
    lone = box(150, 120, 200, 150)
    tracker = CameraTracker("cam-lone")
    for i in range(config.STATIONARY_POLLS):
        tracker.update([lone,
                        box(10 + i * 50, 60, 45 + i * 50, 85),
                        box(300 - i * 40, 70, 340 - i * 40, 95),
                        box(20 + i * 45, 170, 60 + i * 45, 200)])

    assert find_candidates("cam-lone", tracker.tracks, FRAME_W, FRAME_H) == []


def test_zone_polygon_flags_vehicle_in_travel_lane(tmp_path, monkeypatch):
    """With a hand-drawn lane polygon, a stationary vehicle inside it flags."""
    monkeypatch.setattr(config, "ZONES_DIR", tmp_path)
    (tmp_path / "cam-zoned.json").write_text(
        '{"travel_lane": [[100, 100], [250, 100], [250, 200], [100, 200]]}',
        encoding="utf-8",
    )

    # Ground point (bottom-center) at (175, 160) -> inside the polygon.
    offender = box(150, 130, 200, 160)
    tracker = CameraTracker("cam-zoned")
    for i in range(config.STATIONARY_POLLS):
        tracker.update([offender,
                        box(10 + i * 50, 60, 45 + i * 50, 85),
                        box(300 - i * 40, 40, 340 - i * 40, 65),
                        box(20 + i * 45, 210, 60 + i * 45, 238)])

    candidates = find_candidates("cam-zoned", tracker.tracks, FRAME_W, FRAME_H)
    assert len(candidates) == 1
    assert "travel lane" in candidates[0].reason


def test_zone_polygon_ignores_vehicle_outside_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ZONES_DIR", tmp_path)
    (tmp_path / "cam-zoned2.json").write_text(
        '{"travel_lane": [[100, 100], [250, 100], [250, 200], [100, 200]]}',
        encoding="utf-8",
    )

    # Ground point at (35, 235): well outside the polygon.
    parked = box(10, 205, 60, 235)
    tracker = CameraTracker("cam-zoned2")
    for i in range(config.STATIONARY_POLLS):
        tracker.update([parked,
                        box(110 + i * 40, 60, 150 + i * 40, 85),
                        box(300 - i * 40, 40, 340 - i * 40, 65),
                        box(20 + i * 45, 150, 60 + i * 45, 180)])

    assert find_candidates("cam-zoned2", tracker.tracks, FRAME_W, FRAME_H) == []


def test_fallback_flags_genuine_double_parking_pattern():
    """Positive case for the no-zone fallback.

    Without this, the negative tests above would all pass trivially if the
    heuristic simply rejected everything.

    Scene: two vehicles stopped side by side above the curb band (the classic
    double-parked-alongside-parked pattern), with traffic still moving.
    """
    stopped_pair = [box(120, 130, 170, 160), box(175, 132, 225, 162)]
    tracker = CameraTracker("cam-positive")
    for i in range(config.STATIONARY_POLLS):
        moving = [
            box(10 + i * 55, 60, 45 + i * 55, 85),
            box(300 - i * 50, 45, 340 - i * 50, 70),
            box(20 + i * 60, 95, 55 + i * 60, 120),
        ]
        tracker.update(stopped_pair + moving)

    candidates = find_candidates("cam-positive", tracker.tracks, FRAME_W, FRAME_H)
    assert candidates, "genuine double-parking pattern must produce candidates"
    assert all(0.0 < c.score <= 1.0 for c in candidates)
    assert "curb band" in candidates[0].reason


def test_tiny_boxes_are_ignored():
    """Distant specks are below the area floor and must never be candidates."""
    speck = box(150, 100, 155, 103)
    tracker = _stationary_tracker([speck], "cam-speck")
    assert find_candidates("cam-speck", tracker.tracks, FRAME_W, FRAME_H) == []
