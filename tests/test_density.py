"""Tests for the geometric congestion metrics.

No network, no API key — every layout here is synthetic, which is the point:
crowding makes two strong claims (scale invariance and perspective robustness)
and synthetic geometry is the only way to test a claim like that exactly rather
than approximately.

Layouts are built at 352x240, the real NYC DOT frame size, so the
`MIN_BOX_AREA_FRACTION` filter behaves here the way it will in production.

A note on reading the assertions: crowding is *space per vehicle*, so
**low crowding = congested**. The tests deliberately assert flow states as well
as numbers, in both directions, so that a trivial implementation returning a
constant fails at least one of them.
"""

from __future__ import annotations

import math

import pytest

from src import config
from src.analyze.density import (
    FrameMetrics,
    classify,
    crowding_ratio,
    frame_metrics,
)
from src.detect.boxes import Box

# Real NYC DOT snapshot dimensions.
FRAME_W, FRAME_H = 352, 240
FRAME_AREA = FRAME_W * FRAME_H
MIN_KEPT_AREA = config.MIN_BOX_AREA_FRACTION * FRAME_AREA  # ~42 px^2 at 352x240

# Side of a square box guaranteed to fall under the area floor. Derived rather
# than hardcoded: MIN_BOX_AREA_FRACTION is expected to move during Phase 2
# tuning, and a literal here would turn that into a spurious test failure.
SUB_FLOOR_SIDE = math.sqrt(MIN_KEPT_AREA) / 2

# Representative crowding values inside each band, derived from the thresholds
# for the same reason. These moved once already when live measurement showed
# the originals classified 23 of 30 real cameras as jammed, and they will move
# again when eval/labels.json exists — tests must survive that.
MID_MODERATE = (config.CROWDING_JAMMED_MAX + config.CROWDING_MODERATE_MAX) / 2
CLEARLY_CLEAR = config.CROWDING_MODERATE_MAX * 3
CLEARLY_JAMMED = config.CROWDING_JAMMED_MAX / 2


def car(cx: float, cy: float, w: float = 40, h: float = 24, label: str = "car") -> Box:
    """A vehicle box centred at (cx, cy). Defaults are car-shaped at this scale."""
    return Box.from_center(x=cx, y=cy, width=w, height=h, label=label)


def grid(
    cols: int, rows: int, spacing: float, w: float, h: float, x0: float, y0: float
) -> list[Box]:
    """A regular lattice of identical boxes — every vehicle's nearest neighbour
    is exactly `spacing` away, so the expected crowding is exact, not fitted."""
    return [
        car(x0 + c * spacing, y0 + r * spacing, w, h)
        for r in range(rows)
        for c in range(cols)
    ]


def spacing_for(crowding: float, w: float = 40, h: float = 24) -> float:
    """Lattice spacing that yields exactly `crowding` for boxes of this size.

    In a regular grid every nearest-neighbour distance is the spacing, so
    crowding reduces to spacing / diagonal. Inverting that lets a test say
    "put this layout in the jammed band" without hardcoding a pixel distance
    that silently stops meaning what it meant when a threshold moves.
    """
    return crowding * math.hypot(w, h)


# --- the two ends of the scale ---------------------------------------------

def test_packed_grid_scores_low_crowding_and_reads_jammed():
    """Vehicles packed within the jammed band read as jammed."""
    spacing = spacing_for(CLEARLY_JAMMED)
    boxes = grid(cols=4, rows=3, spacing=spacing, w=40, h=24, x0=40, y0=60)
    assert len(boxes) == 12

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)

    assert metrics.crowding == pytest.approx(CLEARLY_JAMMED, rel=1e-6)
    assert metrics.crowding < config.CROWDING_JAMMED_MAX
    assert metrics.flow_state == "jammed"
    assert metrics.vehicle_count == 12


def test_sparse_scatter_scores_high_crowding_and_reads_clear():
    """The same frame size with four vehicles spread across it is free-flowing."""
    boxes = [car(40, 40, 30, 18), car(300, 50, 30, 18),
             car(60, 200, 30, 18), car(310, 200, 30, 18)]

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)

    assert metrics.crowding is not None
    assert metrics.crowding > config.CROWDING_MODERATE_MAX
    assert metrics.flow_state == "clear"
    assert metrics.vehicle_count == 4


def test_middle_band_reads_moderate():
    """The moderate bucket must be reachable, or the classifier is really binary."""
    boxes = grid(
        cols=3, rows=2, spacing=spacing_for(MID_MODERATE), w=40, h=24, x0=60, y0=70
    )

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)

    assert config.CROWDING_JAMMED_MAX < metrics.crowding <= config.CROWDING_MODERATE_MAX
    assert metrics.flow_state == "moderate"


def test_packed_and_sparse_are_not_the_same_verdict():
    """Guards against a constant-returning implementation passing both tests."""
    packed = frame_metrics(
        grid(4, 3, 48, 40, 24, 40, 60), FRAME_W, FRAME_H
    )
    sparse = frame_metrics(
        [car(40, 40, 30, 18), car(300, 50, 30, 18),
         car(60, 200, 30, 18), car(310, 200, 30, 18)],
        FRAME_W, FRAME_H,
    )
    assert packed.flow_state != sparse.flow_state
    assert packed.crowding < sparse.crowding


# --- the load-bearing property: scale invariance ----------------------------

def test_crowding_is_invariant_to_uniform_scale():
    """The whole justification for the metric.

    The same layout photographed from twice as far away produces boxes and gaps
    that are all divided by the same factor. If crowding moved under that, it
    would be measuring camera placement rather than congestion, and it could not
    be compared across ~950 differently-mounted cameras.
    """
    layout = [(60, 70, 40, 24), (110, 90, 36, 22), (75, 150, 44, 26),
              (170, 120, 38, 20), (140, 180, 42, 25)]
    scale = 3.0

    small = [car(x, y, w, h) for x, y, w, h in layout]
    large = [car(x * scale, y * scale, w * scale, h * scale) for x, y, w, h in layout]

    small_crowding = crowding_ratio(small)
    large_crowding = crowding_ratio(large)

    assert small_crowding is not None
    assert large_crowding == pytest.approx(small_crowding, rel=1e-9)


def test_scale_invariance_survives_the_full_frame_metrics_path():
    """Scaling the frame with the layout must also leave flow_state alone.

    `MIN_BOX_AREA_FRACTION` is a *fraction*, so it scales with the frame; this
    checks the filter does not quietly break the invariance the ratio provides.
    """
    layout = [(60, 70, 40, 24), (110, 90, 36, 22), (75, 150, 44, 26),
              (170, 120, 38, 20), (140, 180, 42, 25)]
    scale = 3

    small = frame_metrics(
        [car(x, y, w, h) for x, y, w, h in layout], FRAME_W, FRAME_H
    )
    large = frame_metrics(
        [car(x * scale, y * scale, w * scale, h * scale) for x, y, w, h in layout],
        FRAME_W * scale,
        FRAME_H * scale,
    )

    assert large.crowding == pytest.approx(small.crowding, rel=1e-9)
    assert large.vehicle_count == small.vehicle_count
    assert large.occupancy == pytest.approx(small.occupancy, rel=1e-9)
    assert large.flow_state == small.flow_state


# --- the other load-bearing property: perspective robustness ----------------

# Two depth bands in one frame, each *genuinely equally congested*: the near
# field's boxes and gaps are both exactly 2x the far field's, which is what
# perspective does to a road of uniform occupancy. Counts are deliberately
# unbalanced (5 far, 3 near) — with a balanced split a global-average normalizer
# would coincidentally land on the right median and the test would prove nothing.
_FAR_W, _FAR_H = 25, 15      # area 375 px^2, above the filter
_NEAR_W, _NEAR_H = 50, 30    # exactly twice the far field, in both dimensions

# Both bands are laid out to the *same* crowding ratio, chosen inside the
# jammed band so the mixed frame has a definite verdict to assert on. Spacings
# are derived from it rather than fixed, so the fixture keeps testing
# perspective robustness rather than accidentally testing a threshold.
_EXPECTED_RATIO = CLEARLY_JAMMED
_FAR_SPACING = _EXPECTED_RATIO * math.hypot(_FAR_W, _FAR_H)
_NEAR_SPACING = _EXPECTED_RATIO * math.hypot(_NEAR_W, _NEAR_H)


def _far_field() -> list[Box]:
    return [car(20 + i * _FAR_SPACING, 40, _FAR_W, _FAR_H) for i in range(5)]


def _near_field() -> list[Box]:
    return [car(40 + i * _NEAR_SPACING, 190, _NEAR_W, _NEAR_H) for i in range(3)]


def test_perspective_bands_agree_in_isolation():
    """Sanity floor for the mixed test: each band alone scores the same value."""
    far, near = crowding_ratio(_far_field()), crowding_ratio(_near_field())

    assert far == pytest.approx(_EXPECTED_RATIO, rel=1e-9)
    assert near == pytest.approx(_EXPECTED_RATIO, rel=1e-9)
    # Every box in the far field really is smaller than every box in the near
    # field — otherwise "perspective" is not what is being tested.
    assert max(b.area for b in _far_field()) < min(b.area for b in _near_field())
    # Each vehicle's nearest neighbour is inside its own band, not across the
    # frame; otherwise the bands would not be independently measurable.
    assert min(
        math.dist(f.center, n.center) for f in _far_field() for n in _near_field()
    ) > _NEAR_SPACING


def test_mixed_depth_frame_is_not_skewed_by_the_mix():
    """Far and near bands together score exactly what each scores alone."""
    mixed = _far_field() + _near_field()

    metrics = frame_metrics(mixed, FRAME_W, FRAME_H)

    assert metrics.vehicle_count == 8  # nothing lost to the area filter
    assert metrics.crowding == pytest.approx(_EXPECTED_RATIO, rel=1e-9)
    assert metrics.flow_state == "jammed"


def test_mixed_depth_frame_beats_a_global_normalizer():
    """Explicitly pins the per-vehicle divisor against the tempting shortcut.

    Dividing every gap by the *mean* diagonal of the frame gives a materially
    different — and wrong — answer here, because the two depth bands hold
    different numbers of vehicles. This test fails the moment someone
    "simplifies" `crowding_ratio` to a global normalizer.
    """
    mixed = _far_field() + _near_field()

    mean_diagonal = sum(math.hypot(b.width, b.height) for b in mixed) / len(mixed)
    global_style = sorted(
        min(math.dist(b.center, o.center) for o in mixed if o is not b) / mean_diagonal
        for b in mixed
    )
    global_median = (global_style[3] + global_style[4]) / 2

    actual = crowding_ratio(mixed)
    assert actual == pytest.approx(_EXPECTED_RATIO, rel=1e-9)
    assert global_median != pytest.approx(actual, rel=1e-3)


# --- guards: too few vehicles ----------------------------------------------

def test_empty_box_list():
    metrics = frame_metrics([], FRAME_W, FRAME_H)

    assert metrics == FrameMetrics(
        vehicle_count=0, occupancy=0.0, crowding=None, flow_state="clear", by_class={}
    )


def test_single_box_has_no_neighbour():
    metrics = frame_metrics([car(100, 100)], FRAME_W, FRAME_H)

    assert metrics.vehicle_count == 1
    assert metrics.crowding is None
    assert metrics.flow_state == "clear"


def test_one_below_the_crowding_floor_returns_none():
    """MIN_VEHICLES_FOR_CROWDING - 1 vehicles, packed tight enough that a naive
    implementation would happily call it jammed."""
    count = config.MIN_VEHICLES_FOR_CROWDING - 1
    boxes = [car(60 + i * 44, 100) for i in range(count)]

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)

    assert metrics.vehicle_count == count
    assert metrics.crowding is None
    assert metrics.flow_state == "clear"


def test_exactly_at_the_floor_does_produce_a_ratio():
    """The complement of the test above — the floor must not be off by one."""
    boxes = [car(60 + i * 44, 100) for i in range(config.MIN_VEHICLES_FOR_CROWDING)]

    assert crowding_ratio(boxes) is not None


# --- guards: degenerate geometry -------------------------------------------

def test_zero_area_box_does_not_raise():
    """A collapsed box has no length scale, so it cannot have a ratio — but it
    must not take the frame down with it."""
    boxes = grid(2, 2, 48, 40, 24, 60, 80) + [Box(x1=200, y1=200, x2=200, y2=200)]

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)  # must not raise
    direct = crowding_ratio(boxes)  # unfiltered path, where the box survives

    assert metrics.crowding == pytest.approx(48 / math.hypot(40, 24), rel=1e-6)
    assert direct is not None


def test_zero_width_box_does_not_raise():
    """Zero width with non-zero height still has a non-zero diagonal, so it is a
    legitimate subject — the guard must key off the diagonal, not the area."""
    boxes = grid(2, 2, 48, 40, 24, 60, 80) + [Box(x1=200, y1=180, x2=200, y2=220)]

    assert crowding_ratio(boxes) is not None


def test_all_boxes_degenerate_returns_none():
    boxes = [Box(x1=x, y1=100, x2=x, y2=100) for x in (50, 100, 150, 200)]

    assert crowding_ratio(boxes) is None
    assert frame_metrics(boxes, FRAME_W, FRAME_H).flow_state == "clear"


def test_coincident_centres_are_maximally_crowded_not_an_error():
    """Overlapping detections of the same vehicle give distance 0. That is a real
    (if extreme) reading of 'packed', and must not be confused with None."""
    boxes = [car(100, 100), car(100, 100), car(100, 100)]

    crowding = crowding_ratio(boxes)

    assert crowding == 0.0
    assert classify(crowding, len(boxes)) == "jammed"


def test_zero_area_frame_degrades_instead_of_raising():
    metrics = frame_metrics([car(100, 100)], 0, 0)

    assert metrics.vehicle_count == 0
    assert metrics.occupancy == 0.0
    assert metrics.flow_state == "clear"


# --- supporting metrics -----------------------------------------------------

def test_by_class_counts_labels():
    boxes = (
        [car(40 + i * 60, 60, label="car") for i in range(3)]
        + [car(40 + i * 60, 160, w=60, h=40, label="truck") for i in range(2)]
        + [car(300, 210, w=70, h=30, label="bus")]
    )

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)

    assert metrics.by_class == {"car": 3, "truck": 2, "bus": 1}
    assert sum(metrics.by_class.values()) == metrics.vehicle_count


def test_occupancy_is_the_area_fraction():
    boxes = [car(60, 60, 40, 24), car(200, 60, 40, 24), car(120, 180, 40, 24)]

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)

    assert metrics.occupancy == pytest.approx(3 * 40 * 24 / FRAME_AREA)
    assert 0.0 < metrics.occupancy < 1.0


def test_occupancy_is_clamped_when_boxes_overlap():
    """Summing overlapping boxes can exceed the frame area; the contract says 0..1."""
    boxes = [car(176, 120, 340, 230) for _ in range(4)]

    metrics = frame_metrics(boxes, FRAME_W, FRAME_H)

    assert metrics.occupancy == 1.0


def test_by_class_and_occupancy_reflect_the_filtered_set():
    """Every field must describe the same population of vehicles."""
    kept = [car(60, 60, 40, 24, label="car"), car(200, 60, 40, 24, label="car")]
    dropped = [car(120, 180, SUB_FLOOR_SIDE, SUB_FLOOR_SIDE, label="truck")]
    assert dropped[0].area < MIN_KEPT_AREA

    metrics = frame_metrics(kept + dropped, FRAME_W, FRAME_H)

    assert metrics.vehicle_count == 2
    assert "truck" not in metrics.by_class
    assert metrics.occupancy == pytest.approx(2 * 40 * 24 / FRAME_AREA)


# --- the area filter --------------------------------------------------------

def test_boxes_below_the_area_fraction_are_excluded_from_count():
    real = grid(2, 2, 48, 40, 24, 60, 80)
    specks = [car(250 + i * 8, 30, 6, 4) for i in range(20)]
    assert all(b.area < MIN_KEPT_AREA for b in specks)

    metrics = frame_metrics(real + specks, FRAME_W, FRAME_H)

    assert metrics.vehicle_count == len(real)


def test_specks_cannot_drag_the_verdict_away_from_jammed():
    """Unfiltered, twenty far-field specks packed at their own scale would still
    have huge ratios (tiny diagonals) and would pull the median toward clear."""
    real = grid(4, 3, spacing_for(CLEARLY_JAMMED), 40, 24, 40, 60)
    specks = [car(300 + (i % 4) * 10, 20 + (i // 4) * 10, 6, 4) for i in range(8)]

    assert frame_metrics(real + specks, FRAME_W, FRAME_H).flow_state == "jammed"


def test_area_threshold_boundary_is_inclusive():
    """'Smaller than' the fraction is dropped, so equal must be kept.

    Built in corner form with height 1 so the area lands *exactly* on the
    threshold in floating point — going through `from_center` reconstructs the
    width via two divisions and lands a few ULPs off, which would make this test
    about float rounding rather than about the comparison operator.
    """
    at_threshold = Box(x1=0.0, y1=0.0, x2=MIN_KEPT_AREA, y2=1.0)
    just_below = Box(x1=0.0, y1=0.0, x2=math.nextafter(MIN_KEPT_AREA, 0.0), y2=1.0)
    assert at_threshold.area == MIN_KEPT_AREA
    assert just_below.area < MIN_KEPT_AREA

    assert frame_metrics([at_threshold], FRAME_W, FRAME_H).vehicle_count == 1
    assert frame_metrics([just_below], FRAME_W, FRAME_H).vehicle_count == 0


# --- classify() in isolation ------------------------------------------------

@pytest.mark.parametrize(
    "crowding, expected",
    [
        (0.0, "jammed"),
        (config.CROWDING_JAMMED_MAX / 2, "jammed"),
        (config.CROWDING_JAMMED_MAX, "jammed"),           # boundary is inclusive
        (config.CROWDING_JAMMED_MAX + 1e-9, "moderate"),
        (MID_MODERATE, "moderate"),
        (config.CROWDING_MODERATE_MAX, "moderate"),       # boundary is inclusive
        (config.CROWDING_MODERATE_MAX + 1e-9, "clear"),
        (config.CROWDING_MODERATE_MAX * 4, "clear"),
    ],
)
def test_classify_thresholds(crowding: float, expected: str):
    assert classify(crowding, vehicle_count=10) == expected


def test_classify_none_is_clear():
    assert classify(None, vehicle_count=0) == "clear"
    assert classify(None, vehicle_count=100) == "clear"


def test_classify_ignores_a_ratio_below_the_vehicle_floor():
    """Defensive: a caller passing a ratio for a near-empty frame still gets
    'clear'. An almost-empty road is clear whatever the geometry says."""
    assert classify(0.5, vehicle_count=config.MIN_VEHICLES_FOR_CROWDING - 1) == "clear"


def test_flow_states_are_exactly_the_three_contract_values():
    values = {
        classify(c, 10)
        for c in (
            None,
            config.CROWDING_JAMMED_MAX / 2,
            MID_MODERATE,
            config.CROWDING_MODERATE_MAX * 10,
        )
    }
    assert values == {"clear", "moderate", "jammed"}
