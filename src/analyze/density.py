"""Geometric congestion metrics for a single frame.

Raw vehicle count does not travel between cameras. Fifteen vehicles on a wide
six-lane arterial is free-flowing; fifteen on a narrow one-way is gridlock.
Field of view, road width and camera angle all differ across the ~950 NYC DOT
cameras, and per-camera calibration at that scale is not an option.

**Crowding** is the answer to that, and it is the reason this module exists::

    for each vehicle i:
        r_i = (distance to nearest other vehicle's centre) / (i's own box diagonal)

    crowding = median(r_i)

The division is by *each vehicle's own* diagonal, never by a global average.
That detail is the whole metric. A vehicle 200m down the street projects to a
small box **and** its neighbours project close to it; both the numerator and the
denominator shrink together with depth, so their ratio is invariant to scale and
therefore to perspective. A global normalizer would instead be dominated by
whichever depth band happened to hold more vehicles, and would report a far-field
jam as clear whenever the near field was empty. `tests/test_density.py` pins both
properties down directly.

Reading the number is physical: `crowding ~= 1.0` means vehicles sit roughly one
car-length apart, i.e. bumper to bumper. **Larger means more space per vehicle,
so LOW crowding is MORE congested** — the comparison direction is the easiest
thing in this file to get backwards.

Supporting metrics are `vehicle_count` (the base signal) and `occupancy` (summed
box area over frame area — a crude "how much of the view is vehicle").

Pure functions: no network, no I/O, no state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

from src import config
from src.detect.boxes import Box

#: Occupancy is a sum over boxes that may overlap (and Roboflow does emit
#: overlapping boxes for occluded vehicles), so the raw sum can exceed the frame
#: area. The contract says 0..1, so clamp. Not a config constant because it is a
#: property of the definition rather than a tunable.
_MAX_OCCUPANCY = 1.0

FLOW_CLEAR = "clear"
FLOW_MODERATE = "moderate"
FLOW_JAMMED = "jammed"


@dataclass(frozen=True)
class FrameMetrics:
    """Everything geometry alone can say about one frame."""

    vehicle_count: int
    #: sum(box area) / frame area, 0..1
    occupancy: float
    #: None when there are too few usable vehicles for a median to mean anything.
    crowding: float | None
    #: "clear" | "moderate" | "jammed"
    flow_state: str
    #: {"car": 12, "truck": 2}
    by_class: dict[str, int] = field(default_factory=dict)


def _diagonal(box: Box) -> float:
    """Length of the box's diagonal — our per-vehicle length scale.

    The diagonal rather than the width or the height because a camera sees
    vehicles at every orientation: a car crossing the frame is wide and short, the
    same car heading away is narrow and tall. The diagonal is the one linear
    measure that stays roughly proportional to vehicle size under both.
    """
    return math.hypot(box.width, box.height)


def crowding_ratio(boxes: list[Box]) -> float | None:
    """Median per-vehicle nearest-neighbour distance, in units of car length.

    Returns None when the frame cannot support the statistic — see the guards
    below. Callers must treat None as "no measurement", not as zero; zero would
    mean perfectly stacked vehicles, which is the opposite end of the scale.

    Boxes are *not* area-filtered here; `frame_metrics` does that, since the
    threshold is a fraction of frame area and this function has no frame.
    """
    # One vehicle has no neighbour and two give a single sample — a "median" over
    # that is just noise wearing a statistic's clothes.
    if len(boxes) < config.MIN_VEHICLES_FOR_CROWDING:
        return None

    centers = [b.center for b in boxes]
    ratios: list[float] = []

    # O(n^2), deliberately. n is vehicles visible in one 352x240 frame — tens at
    # the very worst — so a spatial index would cost more in complexity than it
    # saves in time.
    for i, box in enumerate(boxes):
        diagonal = _diagonal(box)
        if diagonal <= 0.0:
            # A zero-width or zero-height detection has no length scale, so it has
            # no meaningful ratio. Skip it as a *subject* but keep it above as a
            # possible *neighbour*: the detection still marks a real vehicle
            # position even when its extent came back malformed.
            continue

        nearest = min(
            math.dist(centers[i], other) for j, other in enumerate(centers) if j != i
        )
        ratios.append(nearest / diagonal)

    # Degenerate boxes can drop the usable sample below the floor even when the
    # raw count cleared it. Apply the same floor to the samples that actually
    # survived, for the same reason it exists in the first place.
    if len(ratios) < config.MIN_VEHICLES_FOR_CROWDING:
        return None

    return median(ratios)


def classify(crowding: float | None, vehicle_count: int) -> str:
    """Map a crowding ratio to a flow state.

    Remember the direction: crowding is *space per vehicle*, so smaller is worse.

    Thresholds come from config and are an initial guess, not a measurement —
    Phase 2 sets them from the labeled set. Do not hardcode them anywhere else.
    """
    # No measurement, or too sparse to have earned one. An almost-empty road is
    # the definition of clear, so this is a real answer rather than a fallback.
    if crowding is None or vehicle_count < config.MIN_VEHICLES_FOR_CROWDING:
        return FLOW_CLEAR

    if crowding <= config.CROWDING_JAMMED_MAX:
        return FLOW_JAMMED
    if crowding <= config.CROWDING_MODERATE_MAX:
        return FLOW_MODERATE
    return FLOW_CLEAR


def frame_metrics(boxes: list[Box], frame_w: int, frame_h: int) -> FrameMetrics:
    """Compute every geometric metric for one frame.

    Boxes below `config.MIN_BOX_AREA_FRACTION` of the frame are dropped first.
    At 352x240 a box that small is a few pixels of far-field noise, and it would
    distort crowding badly: a speck has a tiny diagonal, so its ratio blows up and
    drags the median toward "clear".

    `vehicle_count`, `occupancy` and `by_class` all describe the **filtered** set,
    so every field of the result refers to the same population of vehicles.
    """
    frame_area = float(frame_w) * float(frame_h)
    if frame_area <= 0:
        # A frame with no area has no meaningful density. Degrade rather than
        # raise: the pipeline should never lose an observation over frame
        # metadata it can survive without.
        return FrameMetrics(
            vehicle_count=0,
            occupancy=0.0,
            crowding=None,
            flow_state=FLOW_CLEAR,
            by_class={},
        )

    min_area = config.MIN_BOX_AREA_FRACTION * frame_area
    kept = [b for b in boxes if b.area >= min_area]

    by_class: dict[str, int] = {}
    for box in kept:
        by_class[box.label] = by_class.get(box.label, 0) + 1

    occupancy = min(_MAX_OCCUPANCY, sum(b.area for b in kept) / frame_area)

    crowding = crowding_ratio(kept)

    return FrameMetrics(
        vehicle_count=len(kept),
        occupancy=occupancy,
        crowding=crowding,
        flow_state=classify(crowding, len(kept)),
        by_class=by_class,
    )
