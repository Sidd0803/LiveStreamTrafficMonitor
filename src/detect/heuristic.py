"""Turn stationary tracks into double-parking *candidates*.

This is the spatial half of the decision. The tracker answers "did it move?";
this module answers "is it somewhere it shouldn't be?". Anything surviving both
goes to Gemini for the final call.

Two failure modes drive the design:

1. **Red lights and gridlock.** At a red light every vehicle is stationary and
   in a travel lane. Naively that is a frame full of double-parking. So if most
   vehicles in the frame are stationary we treat it as congestion and suppress
   everything — double parking is defined by a vehicle stopped *while traffic
   moves around it*.

2. **Curbside parking is legal.** A parked car is stationary and roadside. The
   distinguishing feature of double parking is lateral offset from the parked
   row, toward the travel lane.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from src import config
from src.detect.boxes import Box
from src.detect.tracker import Track

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A stationary track that also looks spatially wrong."""

    track: Track
    reason: str
    #: 0-1 geometric suspicion. Not a probability; used only for ranking.
    score: float


def _load_zone(camera_id: str) -> list[tuple[float, float]] | None:
    """Load a hand-drawn travel-lane polygon for this camera, if one exists.

    Format: {"travel_lane": [[x, y], [x, y], ...]} in pixel coordinates of the
    352x240 frame. Optional — the fallback heuristic runs when absent.
    """
    path = config.ZONES_DIR / f"{camera_id}.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        poly = [(float(x), float(y)) for x, y in data["travel_lane"]]
        return poly if len(poly) >= 3 else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        log.warning("bad zone file %s: %s", path, exc)
        return None


def point_in_polygon(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Standard ray-casting test."""
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


def _ground_point(box: Box) -> tuple[float, float]:
    """Where the vehicle meets the road: bottom-center of the box.

    Using the box center instead would place tall vehicles (box trucks) well
    above the roadway and misjudge which lane they occupy.
    """
    cx, _ = box.center
    return (cx, box.y2)


def find_candidates(
    camera_id: str,
    all_tracks: list[Track],
    frame_w: int,
    frame_h: int,
) -> list[Candidate]:
    """Select stationary tracks that are plausibly double parked."""
    stationary = [t for t in all_tracks if t.is_stationary and t.misses == 0]
    if not stationary:
        return []

    # --- Guard 1: congestion / red light -----------------------------------
    # If nearly everything is stopped, nothing is "stopped while traffic moves".
    if len(all_tracks) >= config.CONGESTION_MIN_TRACKS:
        stationary_fraction = len(stationary) / len(all_tracks)
        if stationary_fraction >= config.CONGESTION_STATIONARY_FRACTION:
            log.info(
                "%s: suppressing %d candidates — %.0f%% of %d tracks stationary "
                "(congestion/red light, not double parking)",
                camera_id, len(stationary), stationary_fraction * 100, len(all_tracks),
            )
            return []

    zone = _load_zone(camera_id)
    frame_area = float(frame_w * frame_h)
    candidates: list[Candidate] = []

    for track in stationary:
        box = track.box

        # --- Guard 2: too small to judge ----------------------------------
        if box.area / frame_area < config.MIN_BOX_AREA_FRACTION:
            continue

        gx, gy = _ground_point(box)

        if zone is not None:
            # Hand-drawn travel lane: authoritative when available.
            if not point_in_polygon((gx, gy), zone):
                continue
            candidates.append(
                Candidate(
                    track=track,
                    reason="stationary inside hand-annotated travel lane",
                    score=0.85,
                )
            )
            continue

        # --- Fallback: lateral offset from the curb ------------------------
        # Without a zone polygon, approximate "not at the curb" as "not in the
        # bottom band of the frame". Cameras look down the roadway, so the
        # near-field bottom edge is where curbside parking appears largest.
        # This is crude and is exactly why zone polygons exist; it is a
        # starting point for Phase 2 tuning, not a finished rule.
        curb_band_start = frame_h * (1.0 - config.CURB_BAND_FRACTION)
        if gy >= curb_band_start:
            continue

        # Require a neighbour roughly beside it: double parking means stopping
        # *alongside* a parked vehicle, and this cheaply rejects a lone stopped
        # car in an empty street (more likely a camera artifact or a driveway).
        has_neighbour = any(
            other is not track
            and abs(_ground_point(other.box)[1] - gy) < box.height * 1.5
            and abs(_ground_point(other.box)[0] - gx) < box.width * 4.0
            for other in stationary
        )
        if not has_neighbour:
            continue

        # Score by how far from the curb band it sits — further in is more
        # suspicious. Purely for ranking which candidates to send first.
        depth = (curb_band_start - gy) / max(1.0, curb_band_start)
        candidates.append(
            Candidate(
                track=track,
                reason="stationary outside the curb band, alongside another stopped vehicle",
                score=round(min(0.8, 0.4 + depth * 0.4), 3),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    if candidates:
        log.info("%s: %d candidate(s) from %d stationary tracks",
                 camera_id, len(candidates), len(stationary))
    return candidates
