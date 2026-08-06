"""Tests for the Observation record and the local JSONL store.

No network, no API keys, and nothing written outside pytest's tmp_path.

The two things most worth pinning down here are timezone handling and
robustness to a half-written file. Both fail *silently* in the wild: a naive
datetime sorts wrongly without raising, and a poller killed mid-write leaves a
truncated line that would otherwise take the whole dashboard down on every
subsequent refresh.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.storage.base import Observation
from src.storage.local import DEFAULT_FILENAME, LocalStore, get_store


def make_obs(
    camera_id: str = "cam-1",
    captured_at: datetime | None = None,
    **overrides,
) -> Observation:
    """A fully-populated Observation, with the interesting fields overridable."""
    fields = dict(
        camera_id=camera_id,
        camera_name="Amsterdam Ave @ 96 St",
        latitude=40.7935,
        longitude=-73.9712,
        area="Manhattan",
        captured_at=captured_at or datetime(2026, 8, 6, 14, 30, tzinfo=timezone.utc),
        vehicle_count=12,
        occupancy=0.184,
        crowding=1.42,
        by_class={"car": 9, "truck": 2, "bus": 1},
        geometric_flow="jammed",
        gemini_flow="moderate",
        gemini_confidence=0.72,
        gemini_reason="Vehicles are closely spaced but the lane is still moving.",
        gemini_notable="construction blocking right lane",
        final_flow="moderate",
        frame_path="runs/frames/cam-1_20260806T143000Z.jpg",
    )
    fields.update(overrides)
    return Observation(**fields)


@pytest.fixture()
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path / "observations.jsonl")


# --- round-trip ------------------------------------------------------------

def test_round_trip_preserves_every_field(store):
    """A saved observation reads back byte-for-byte equal, nulls included."""
    obs = make_obs(
        crowding=None,          # below MIN_VEHICLES_FOR_CROWDING
        gemini_flow=None,       # Gemini not called / unavailable
        gemini_confidence=None,
        gemini_reason=None,
        gemini_notable=None,
        frame_path=None,
        by_class={"car": 4, "truck": 1, "motorcycle": 2},
    )
    store.save(obs)

    (loaded,) = store.recent()
    assert loaded == obs
    # Spelled out as well as compared, so a change to __eq__ cannot make this
    # pass vacuously.
    assert loaded.crowding is None
    assert loaded.gemini_flow is None
    assert loaded.gemini_confidence is None
    assert loaded.gemini_reason is None
    assert loaded.gemini_notable is None
    assert loaded.frame_path is None
    assert loaded.by_class == {"car": 4, "truck": 1, "motorcycle": 2}


def test_to_dict_is_json_safe(store):
    """to_dict() must contain nothing json.dumps would choke on."""
    import json

    payload = make_obs().to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["captured_at"], str)


def test_captured_at_round_trips_as_aware_utc(store):
    when = datetime(2026, 8, 6, 14, 30, 15, tzinfo=timezone.utc)
    store.save(make_obs(captured_at=when))

    (loaded,) = store.recent()
    assert loaded.captured_at == when
    assert loaded.captured_at.tzinfo is not None
    assert loaded.captured_at.utcoffset() == timedelta(0)


def test_non_utc_aware_datetime_is_converted_to_utc():
    """An aware datetime in another zone is stored as the same instant in UTC."""
    eastern = timezone(timedelta(hours=-4))
    obs = make_obs(captured_at=datetime(2026, 8, 6, 10, 30, tzinfo=eastern))

    assert obs.captured_at == datetime(2026, 8, 6, 14, 30, tzinfo=timezone.utc)
    assert obs.captured_at.utcoffset() == timedelta(0)


# --- the naive-datetime rule -----------------------------------------------

def test_naive_datetime_is_interpreted_as_utc_not_local(store):
    """Documented rule: a naive datetime means UTC with the tzinfo dropped.

    The alternative -- storing it naive -- reads back naive and then compares
    against aware values, which raises TypeError inside sort(). This pins the
    rule so nobody "fixes" it into local-time interpretation later.
    """
    naive = datetime(2026, 8, 6, 14, 30)
    obs = make_obs(captured_at=naive)

    # Normalized at construction, not merely on the way to disk.
    assert obs.captured_at == datetime(2026, 8, 6, 14, 30, tzinfo=timezone.utc)
    assert obs.captured_at.tzinfo is not None

    store.save(obs)
    (loaded,) = store.recent()
    assert loaded.captured_at == datetime(2026, 8, 6, 14, 30, tzinfo=timezone.utc)
    assert loaded == obs


def test_naive_and_aware_observations_sort_together(store):
    """The point of the rule: mixed inputs must still be orderable."""
    store.save_many([
        make_obs(camera_id="a", captured_at=datetime(2026, 8, 6, 12, 0)),
        make_obs(camera_id="b", captured_at=datetime(2026, 8, 6, 13, 0, tzinfo=timezone.utc)),
    ])
    assert [o.camera_id for o in store.recent()] == ["b", "a"]


# --- recent() --------------------------------------------------------------

def test_recent_returns_newest_first_and_respects_limit(store):
    base = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    # Written oldest-first so a store that just echoed file order would fail.
    store.save_many([
        make_obs(camera_id=f"cam-{i}", captured_at=base + timedelta(minutes=i))
        for i in range(5)
    ])

    everything = store.recent()
    assert [o.camera_id for o in everything] == [
        "cam-4", "cam-3", "cam-2", "cam-1", "cam-0"
    ]

    limited = store.recent(limit=2)
    assert [o.camera_id for o in limited] == ["cam-4", "cam-3"]


def test_recent_filters_by_camera(store):
    base = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    store.save_many([
        make_obs(camera_id="cam-a", captured_at=base),
        make_obs(camera_id="cam-b", captured_at=base + timedelta(minutes=1)),
        make_obs(camera_id="cam-a", captured_at=base + timedelta(minutes=2)),
        make_obs(camera_id="cam-c", captured_at=base + timedelta(minutes=3)),
    ])

    rows = store.recent(camera_id="cam-a")
    assert len(rows) == 2
    assert {o.camera_id for o in rows} == {"cam-a"}
    # limit applies after filtering, not before
    assert len(store.recent(camera_id="cam-a", limit=1)) == 1
    assert store.recent(camera_id="cam-a", limit=1)[0].captured_at == base + timedelta(minutes=2)


def test_recent_on_unknown_camera_is_empty(store):
    store.save(make_obs(camera_id="cam-a"))
    assert store.recent(camera_id="nope") == []


# --- latest_per_camera() ---------------------------------------------------

def test_latest_per_camera_picks_the_newest_row_per_camera(store):
    """Seeded deliberately out of order, so file position cannot carry this."""
    base = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    store.save_many([
        # cam-a: newest is written first
        make_obs(camera_id="cam-a", captured_at=base + timedelta(minutes=30), vehicle_count=30),
        make_obs(camera_id="cam-b", captured_at=base + timedelta(minutes=5), vehicle_count=5),
        make_obs(camera_id="cam-a", captured_at=base, vehicle_count=1),
        # cam-b: newest is written last
        make_obs(camera_id="cam-b", captured_at=base + timedelta(minutes=40), vehicle_count=40),
        # cam-a: newest sits in the middle of that camera's own rows
        make_obs(camera_id="cam-a", captured_at=base + timedelta(minutes=10), vehicle_count=10),
        make_obs(camera_id="cam-c", captured_at=base + timedelta(minutes=2), vehicle_count=2),
    ])

    latest = store.latest_per_camera()
    assert len(latest) == 3
    by_id = {o.camera_id: o for o in latest}
    assert by_id["cam-a"].vehicle_count == 30
    assert by_id["cam-b"].vehicle_count == 40
    assert by_id["cam-c"].vehicle_count == 2


def test_latest_per_camera_is_empty_before_anything_is_written(store):
    assert store.latest_per_camera() == []


# --- robustness ------------------------------------------------------------

def test_corrupt_line_is_skipped_and_neighbours_survive(store, caplog):
    """A poller killed mid-write must cost one observation, not the dashboard."""
    base = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    store.save(make_obs(camera_id="cam-a", captured_at=base))
    store.save(make_obs(camera_id="cam-b", captured_at=base + timedelta(minutes=1)))

    # Splice a truncated line between the two good ones.
    lines = store.path.read_text(encoding="utf-8").splitlines()
    truncated = lines[1][: len(lines[1]) // 2]
    store.path.write_text(
        "\n".join([lines[0], truncated, lines[1]]) + "\n", encoding="utf-8"
    )

    with caplog.at_level("WARNING"):
        rows = store.recent()

    assert {o.camera_id for o in rows} == {"cam-a", "cam-b"}
    assert len(rows) == 2
    assert any("unreadable" in r.message for r in caplog.records)


def test_valid_json_of_the_wrong_shape_is_skipped(store):
    """Not every bad line is truncated -- wrong shape must not raise either."""
    store.save(make_obs(camera_id="cam-a"))
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"hello": "world"}\n')   # valid JSON, not an Observation
        fh.write("[1, 2, 3]\n")            # valid JSON, not even a mapping
        fh.write("\n")                     # blank line, ignored silently

    rows = store.recent()
    assert len(rows) == 1
    assert rows[0].camera_id == "cam-a"


def test_reading_a_nonexistent_file_returns_empty(tmp_path):
    empty = LocalStore(tmp_path / "nested" / "never-written.jsonl")
    assert empty.recent() == []
    assert empty.latest_per_camera() == []
    # Reading must not have created anything on the way past.
    assert not empty.path.exists()


def test_save_creates_parent_directories(tmp_path):
    nested = LocalStore(tmp_path / "deep" / "deeper" / "observations.jsonl")
    nested.save(make_obs())
    assert nested.path.exists()
    assert len(nested.recent()) == 1


def test_saves_append_rather_than_overwrite(tmp_path):
    """The whole reason for JSONL: a later write cannot destroy an earlier one."""
    path = tmp_path / "observations.jsonl"
    base = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)

    LocalStore(path).save(make_obs(camera_id="cam-a", captured_at=base))
    # A fresh instance, as a restarted poller would be.
    LocalStore(path).save(make_obs(camera_id="cam-b", captured_at=base + timedelta(minutes=1)))

    assert len(LocalStore(path).recent()) == 2


def test_save_many_with_nothing_to_save_is_a_no_op(tmp_path):
    empty = LocalStore(tmp_path / "observations.jsonl")
    empty.save_many([])
    assert not empty.path.exists()


def test_directory_path_gets_the_default_filename(tmp_path):
    """tmp_path is a directory; treat that as 'the default file, in here'."""
    store = LocalStore(tmp_path)
    assert store.path == tmp_path / DEFAULT_FILENAME
    store.save(make_obs())
    assert len(store.recent()) == 1


# --- the Phase 3 seam ------------------------------------------------------

def test_get_store_returns_local_store_for_local():
    assert isinstance(get_store("local"), LocalStore)
    assert isinstance(get_store("LOCAL "), LocalStore)  # normalized


def test_get_store_reads_the_configured_backend(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "STORAGE_BACKEND", "local")
    assert isinstance(get_store(), LocalStore)


def test_get_store_raises_clearly_for_gcp():
    """Phase 3 must fail loudly, not fall back to local and lose data."""
    with pytest.raises(NotImplementedError, match="Phase 3"):
        get_store("gcp")


def test_get_store_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown STORAGE_BACKEND"):
        get_store("postgres")
