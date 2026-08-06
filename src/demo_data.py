"""Synthetic observations, so the dashboard is useful with an empty store.

Two reasons this exists:

1. **Development.** `app.py` is being built while the sweep pipeline is still
   in pieces. A dashboard that needs a live sweep before it renders anything
   cannot be iterated on.
2. **The demo itself.** A network-wide sweep is ~250 Roboflow calls plus ~250
   Gemini calls; that is not something to run live in front of an audience, and
   a venue with no usable wifi would otherwise leave nothing on screen. This
   module has no network access, no API keys, and no dependency on the storage
   backend existing — `python -c "from src.demo_data import demo_observations"`
   works on a fresh clone.

The generator is **seeded**, so the same seed produces the same map every time.
A demo that reshuffles itself between run-throughs is a demo you cannot
rehearse.

Everything here is fabricated. `app.py` is responsible for saying so loudly on
screen; camera ids are prefixed `demo-` so a synthetic observation is
identifiable even if it escapes into a log or a CSV export.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src import config

# Import the real Observation, deliberately without a fallback.
#
# An earlier version defined a structurally identical stand-in if this import
# failed, so demo mode could work before storage/base.py existed. That is now a
# hazard rather than a convenience: if the real type ever drifts or its import
# breaks, a silent fallback would render demo data against a divergent schema
# and look perfectly healthy. Failing loudly here is the whole point — demo
# mode is only useful if it exercises the same type the live path does.
from src.storage.base import Observation


#: Real NYC intersections with real coordinates, spread across all five
#: boroughs so the map reads as NYC rather than as a cloud of random dots. The
#: first six are the hand-curated demo cameras from data/demo_cameras.json.
#:
#: `bias` is the camera's temperament: how congested this location tends to be.
#: Times Square-adjacent and bridge-approach corridors sit jammed; Staten
#: Island arterials sit clear. Without it every camera averages to "moderate"
#: and the map is a uniform amber smear that demonstrates nothing.
DEMO_LOCATIONS: list[tuple[str, str, float, float, str]] = [
    # (name, borough, lat, lon, bias)
    ("Amsterdam Ave @ 60 St", "Manhattan", 40.771057, -73.987139, "moderate"),
    ("Columbus Ave @ 65 St", "Manhattan", 40.772970, -73.981993, "moderate"),
    ("Northern Blvd @ 114 St", "Queens", 40.758271, -73.855567, "clear"),
    ("Grand Concourse @ Bedford Park Blvd", "Bronx", 40.872207, -73.887658, "moderate"),
    ("Atlantic Ave @ 111 St", "Brooklyn", 40.692157, -73.834927, "clear"),
    ("Flatbush Ave @ Eastern Pkwy", "Brooklyn", 40.672796, -73.969278, "jammed"),
    ("Broadway @ 34 St", "Manhattan", 40.749700, -73.987800, "jammed"),
    ("Canal St @ Bowery", "Manhattan", 40.716800, -73.997000, "jammed"),
    ("1 Ave @ 42 St", "Manhattan", 40.749000, -73.968000, "moderate"),
    ("W 14 St @ 8 Ave", "Manhattan", 40.740000, -74.002500, "jammed"),
    ("3 Ave @ 86 St", "Manhattan", 40.778500, -73.954000, "moderate"),
    ("125 St @ Lenox Ave", "Manhattan", 40.809000, -73.945000, "moderate"),
    ("FDR Dr @ 96 St", "Manhattan", 40.782500, -73.943000, "clear"),
    ("Bedford Ave @ N 7 St", "Brooklyn", 40.717500, -73.956500, "moderate"),
    ("Ocean Pkwy @ Church Ave", "Brooklyn", 40.643000, -73.976000, "clear"),
    ("Kings Hwy @ Coney Island Ave", "Brooklyn", 40.609000, -73.964000, "moderate"),
    ("Queens Blvd @ 71 Ave", "Queens", 40.720500, -73.848000, "moderate"),
    ("Astoria Blvd @ 31 St", "Queens", 40.770000, -73.917000, "clear"),
    ("Roosevelt Ave @ 82 St", "Queens", 40.747500, -73.884000, "jammed"),
    ("Jamaica Ave @ 168 St", "Queens", 40.704500, -73.797000, "moderate"),
    ("E Fordham Rd @ Webster Ave", "Bronx", 40.862000, -73.890000, "jammed"),
    ("Bruckner Blvd @ Hunts Point Ave", "Bronx", 40.813000, -73.883000, "moderate"),
    ("Webster Ave @ Gun Hill Rd", "Bronx", 40.878000, -73.865000, "clear"),
    ("Hylan Blvd @ New Dorp Ln", "Staten Island", 40.573000, -74.117000, "clear"),
    ("Richmond Ave @ Forest Hill Rd", "Staten Island", 40.596000, -74.165000, "clear"),
    ("Bay St @ Vanderbilt Ave", "Staten Island", 40.618000, -74.074000, "moderate"),
]

#: Plausible one-sentence rationales, in the register Gemini actually answers
#: in. Grouped by the verdict they justify so the reason never contradicts the
#: flow state shown next to it.
_REASONS: dict[str, list[str]] = {
    "clear": [
        "Vehicles are well spaced across all lanes with no queuing at the signal.",
        "Traffic is light and moving freely; the intersection is clear.",
        "Only a handful of vehicles are visible and none are stopped behind another.",
        "Lanes are open with large gaps between vehicles.",
    ],
    "moderate": [
        "Traffic is steady with a short queue forming in the near lane.",
        "Vehicles are closely spaced but still rolling; no full stop visible.",
        "Moderate volume with a queue of about five cars at the light.",
        "The curb lane is slow but the through lane is still moving.",
    ],
    "jammed": [
        "Vehicles are bumper to bumper across every lane with no visible gaps.",
        "A stationary queue extends past the far end of the frame.",
        "Traffic is fully stopped and blocking the box at the intersection.",
        "Dense standing queue in all lanes; no forward movement apparent.",
    ],
}

#: Occasional scene notes. Mostly empty, because most frames have nothing
#: remarkable in them and a demo where every camera reports an incident is not
#: a believable demo.
_NOTABLE = [
    "",
    "",
    "",
    "",
    "",
    "double-parked box truck blocking the right lane",
    "construction plates narrowing the roadway",
    "bus stopped at the near-side stop",
    "emergency vehicle with lights on in the median",
]

#: Plausible ranges per flow state: (count_lo, count_hi, crowding_lo, crowding_hi).
#: Crowding bands straddle the thresholds in config.py so the geometric verdict
#: derived from them lands where intended.
_PROFILE: dict[str, tuple[int, int, float, float]] = {
    "clear": (1, 7, 3.1, 6.5),
    "moderate": (6, 14, 1.7, 3.0),
    "jammed": (12, 26, 0.9, 1.55),
}

_FLOW_STATES = ("clear", "moderate", "jammed")


def _slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in name]
    return "".join(keep).strip("-").replace("--", "-")


def _classify(crowding: float | None, vehicle_count: int) -> str:
    """Local mirror of `density.classify`, using the same config thresholds.

    Deliberately not importing `src.analyze.density`: demo mode must work on a
    fresh clone before that module exists. The thresholds come from config, so
    the two cannot silently drift apart.
    """
    if vehicle_count < config.MIN_VEHICLES_FOR_CROWDING or crowding is None:
        return "clear"
    if crowding <= config.CROWDING_JAMMED_MAX:
        return "jammed"
    if crowding <= config.CROWDING_MODERATE_MAX:
        return "moderate"
    return "clear"


def _drift(rng: random.Random, bias: str) -> str:
    """Nudge a camera's temperament one step for a given observation.

    Real corridors do not sit at one flow state all day, and a per-camera
    history where every row is identical makes the recent-observations table
    look fake.
    """
    i = _FLOW_STATES.index(bias)
    roll = rng.random()
    if roll < 0.60:
        return bias
    step = -1 if roll < 0.80 else 1
    return _FLOW_STATES[max(0, min(len(_FLOW_STATES) - 1, i + step))]


def _by_class(rng: random.Random, count: int) -> dict[str, int]:
    """Split a vehicle count into a plausible class mix — mostly cars."""
    if count <= 0:
        return {}
    trucks = rng.randint(0, max(0, count // 6))
    buses = 1 if count >= 10 and rng.random() < 0.35 else 0
    cars = count - trucks - buses
    mix = {"car": cars}
    if trucks:
        mix["truck"] = trucks
    if buses:
        mix["bus"] = buses
    return mix


def _one_observation(
    rng: random.Random,
    location: tuple[str, str, float, float, str],
    captured_at: datetime,
) -> Observation:
    name, area, lat, lon, bias = location
    state = _drift(rng, bias)

    lo, hi, c_lo, c_hi = _PROFILE[state]
    count = rng.randint(lo, hi)
    crowding = (
        round(rng.uniform(c_lo, c_hi), 2)
        if count >= config.MIN_VEHICLES_FOR_CROWDING
        else None
    )
    # Occupancy tracks count loosely; more vehicles fill more of the frame.
    occupancy = round(min(0.85, count * rng.uniform(0.012, 0.028)), 3)

    geometric_flow = _classify(crowding, count)

    # Gemini usually agrees with geometry. It is made to disagree ~25% of the
    # time on purpose: the side-by-side comparison is the point of the project,
    # and a demo where the two columns always match shows nothing.
    if rng.random() < 0.25:
        alternatives = [s for s in _FLOW_STATES if s != geometric_flow]
        gemini_flow = rng.choice(alternatives)
    else:
        gemini_flow = geometric_flow

    # Confidence straddles GEMINI_MIN_CONFIDENCE so both branches of the fusion
    # rule are visible on screen.
    gemini_confidence = round(rng.uniform(0.38, 0.97), 2)

    # A small share of cameras have no Gemini verdict at all — quota exhausted,
    # API down, or simply not called. The UI has to survive that.
    if rng.random() < 0.08:
        gemini_flow = None
        gemini_confidence = None
        gemini_reason = None
        gemini_notable = None
    else:
        gemini_reason = rng.choice(_REASONS[gemini_flow])
        gemini_notable = rng.choice(_NOTABLE) or None

    if gemini_flow is not None and (gemini_confidence or 0.0) >= config.GEMINI_MIN_CONFIDENCE:
        final_flow = gemini_flow
    else:
        final_flow = geometric_flow

    return Observation(
        camera_id=f"demo-{_slug(name)}",
        camera_name=name,
        latitude=lat,
        longitude=lon,
        area=area,
        captured_at=captured_at,
        vehicle_count=count,
        occupancy=occupancy,
        crowding=crowding,
        by_class=_by_class(rng, count),
        geometric_flow=geometric_flow,
        gemini_flow=gemini_flow,
        gemini_confidence=gemini_confidence,
        gemini_reason=gemini_reason,
        gemini_notable=gemini_notable,
        final_flow=final_flow,
        # No cached frames exist in demo mode. The app renders a placeholder
        # rather than a broken image.
        frame_path=None,
    )


def demo_observations(
    cameras: int = len(DEMO_LOCATIONS),
    history: int = 4,
    seed: int = 7,
    now: datetime | None = None,
) -> list[Observation]:
    """Generate synthetic observations, newest first.

    `history` observations per camera, spaced roughly a sweep apart, so the
    recent-observations table has something to show and the map has a defensible
    "latest" for each camera.
    """
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)

    out: list[Observation] = []
    for location in DEMO_LOCATIONS[:cameras]:
        for step in range(history):
            # Sweeps land a few minutes apart, with jitter — a perfectly regular
            # grid of timestamps is another tell that the data is fake.
            captured_at = now - timedelta(
                minutes=step * 5, seconds=rng.randint(0, 90)
            )
            out.append(_one_observation(rng, location, captured_at))

    out.sort(key=lambda o: o.captured_at, reverse=True)
    return out


def latest_per_camera(observations: list[Observation]) -> list[Observation]:
    """Newest observation for each camera — the demo-mode stand-in for the store.

    Mirrors `LocalStore.latest_per_camera()` so `app.py` can treat the two
    sources identically.
    """
    newest: dict[str, Observation] = {}
    for obs in observations:
        seen = newest.get(obs.camera_id)
        if seen is None or obs.captured_at > seen.captured_at:
            newest[obs.camera_id] = obs
    return sorted(newest.values(), key=lambda o: o.camera_name)
