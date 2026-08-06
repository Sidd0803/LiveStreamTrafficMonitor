"""The Observation record and the storage interface it is written through.

Why an interface at all: Phase 1 runs entirely on local disk and Phase 3 moves
to Firestore + GCS. Both implementations satisfy `Store`, so the migration is a
config flip rather than a rewrite of every call site that reads or writes
observations.

Why one record carries both verdicts: the geometric classifier and Gemini judge
the same frame independently, and storing both next to the fused answer makes
the Phase 2 ablation -- geometry alone, Gemini alone, the fusion -- free at eval
time, scored from data already on disk. If only `final_flow` were kept, that
comparison would be unrecoverable after the fact, so the schema is deliberately
shaped to make losing it impossible.

An Observation describes one camera at one instant and never references another
frame. Time series are built by aggregating these, never by comparing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

# One canonical timezone for everything stored. See _as_utc() for why naive
# datetimes are not simply rejected.
UTC = timezone.utc


def _as_utc(value: datetime) -> datetime:
    """Coerce a datetime to timezone-aware UTC.

    **A naive datetime is interpreted as already being UTC**, not as local time.

    This is the classic bug in this kind of code and it is worth being explicit
    about. The tempting alternative -- letting a naive datetime through
    untouched -- means it serializes to an ISO string with no offset, reads back
    naive, and then compares and sorts against genuinely-UTC values as though
    the machine's local offset did not exist. On a US-Eastern laptop that is a
    silent four-or-five hour error in `recent()` ordering and in
    `latest_per_camera()`, with nothing anywhere to indicate it happened.

    Interpreting as UTC is the safe reading because every producer in this
    project already works in UTC (`datetime.now(UTC)` in the poller, ISO strings
    with an offset off the wire). A naive value therefore means "someone dropped
    the tzinfo", not "this is wall-clock local time". The cost of being wrong is
    bounded and visible; the cost of the alternative is invisible.

    Aware values in any other zone are converted, so everything downstream can
    assume UTC without checking.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(raw: str | datetime) -> datetime:
    """Parse a stored timestamp back into an aware UTC datetime.

    Accepts a trailing "Z" as well as a numeric offset. We always *write*
    "+00:00", but Firestore and hand-edited JSON both emit "Z", and this is the
    boundary where that has to stop mattering.
    """
    if isinstance(raw, datetime):
        return _as_utc(raw)
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return _as_utc(datetime.fromisoformat(text))


@dataclass(frozen=True)
class Observation:
    """One camera, one instant: the metrics, both verdicts, and the fusion."""

    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    area: str                      # borough, straight off the NYC DOT catalog
    captured_at: datetime          # timezone-aware UTC; see _as_utc()
    vehicle_count: int
    occupancy: float
    crowding: float | None         # None below MIN_VEHICLES_FOR_CROWDING
    by_class: dict[str, int]
    geometric_flow: str            # density.py's verdict
    # No defaults on any of the fields below, deliberately. The Gemini fields
    # being None is meaningful ("not called, or unavailable") and the caller
    # should have to say so. A default of None would let a wiring bug drop
    # Gemini's verdict silently -- exactly the loss the schema exists to prevent.
    gemini_flow: str | None        # None when not called or unavailable
    gemini_confidence: float | None
    gemini_reason: str | None
    gemini_notable: str | None
    final_flow: str                # after the fusion rule
    frame_path: str | None

    def __post_init__(self) -> None:
        # Normalize at construction rather than only at serialization, so the
        # "always aware UTC" invariant holds for in-memory objects too. Sorting
        # in recent() and the max() in latest_per_camera() both compare these
        # directly, and mixing naive and aware datetimes raises TypeError --
        # a crashed dashboard, from one caller that forgot a tzinfo.
        object.__setattr__(self, "captured_at", _as_utc(self.captured_at))

    def to_dict(self) -> dict:
        """JSON-safe mapping. The only non-trivial field is the timestamp."""
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "area": self.area,
            # ISO 8601 with an explicit offset. Sorts lexicographically in the
            # same order it sorts chronologically, which makes a raw JSONL file
            # greppable and sortable without parsing.
            "captured_at": self.captured_at.isoformat(),
            "vehicle_count": self.vehicle_count,
            "occupancy": self.occupancy,
            "crowding": self.crowding,
            # Copy: the caller must not be able to mutate a stored record's
            # dict through a reference it still holds.
            "by_class": dict(self.by_class),
            "geometric_flow": self.geometric_flow,
            "gemini_flow": self.gemini_flow,
            "gemini_confidence": self.gemini_confidence,
            "gemini_reason": self.gemini_reason,
            "gemini_notable": self.gemini_notable,
            "final_flow": self.final_flow,
            "frame_path": self.frame_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        """Rebuild from a stored mapping.

        Core identity and metric fields are looked up strictly -- a record
        missing them is corrupt, and raising here is what lets the reader in
        local.py recognise a bad line and skip it. The nullable Gemini fields
        and frame_path use .get() so records written before a field existed
        still load; a rolling schema is normal for an append-only log.
        """
        return cls(
            camera_id=str(d["camera_id"]),
            camera_name=str(d.get("camera_name", "")),
            latitude=float(d.get("latitude", 0.0)),
            longitude=float(d.get("longitude", 0.0)),
            area=str(d.get("area", "")),
            captured_at=_parse_timestamp(d["captured_at"]),
            vehicle_count=int(d["vehicle_count"]),
            occupancy=float(d.get("occupancy", 0.0)),
            crowding=None if d.get("crowding") is None else float(d["crowding"]),
            by_class={str(k): int(v) for k, v in (d.get("by_class") or {}).items()},
            geometric_flow=str(d["geometric_flow"]),
            gemini_flow=_opt_str(d.get("gemini_flow")),
            gemini_confidence=_opt_float(d.get("gemini_confidence")),
            gemini_reason=_opt_str(d.get("gemini_reason")),
            gemini_notable=_opt_str(d.get("gemini_notable")),
            final_flow=str(d["final_flow"]),
            frame_path=_opt_str(d.get("frame_path")),
        )


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _opt_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


@runtime_checkable
class Store(Protocol):
    """What the pipeline and the dashboard are allowed to assume about storage.

    Kept deliberately small. Every method here has a cheap Firestore equivalent
    (append, batch write, ordered query, per-camera latest), which is what keeps
    Phase 3 a swap rather than a redesign. Anything richer -- aggregation,
    joins, time bucketing -- belongs above this line, computed from the returned
    observations, not pushed into the backend.
    """

    def save(self, obs: Observation) -> None:
        """Persist one observation."""
        ...

    def save_many(self, observations: list[Observation]) -> None:
        """Persist a batch. A sweep produces hundreds at once."""
        ...

    def recent(
        self, camera_id: str | None = None, limit: int = 500
    ) -> list[Observation]:
        """Newest-first, optionally for one camera, at most `limit` rows."""
        ...

    def latest_per_camera(self) -> list[Observation]:
        """Exactly one row per camera -- the newest. This is the map."""
        ...
