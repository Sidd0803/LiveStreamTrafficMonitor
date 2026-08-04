"""NYC DOT traffic camera source.

This is the detection source for the project. NYC DOT operates ~950 cameras on
local streets, which is where double parking actually happens. (511NY's NYC
coverage is limited-access highways, where it does not.)

Both endpoints are public and require no API key:
    GET /api/cameras                  -> JSON array of camera objects
    GET /api/cameras/{id}/image       -> JPEG snapshot

Polling is rate-limited per camera. These are public civic feeds; hammering
them is both rude and a good way to get blocked.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import requests

from src import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Camera:
    """One NYC DOT camera."""

    id: str
    name: str
    latitude: float
    longitude: float
    area: str
    image_url: str
    online: bool

    @property
    def short_name(self) -> str:
        return f"{self.name} ({self.area})"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


_SESSION = _session()

# Per-camera timestamp of the last snapshot fetch, guarding MIN_POLL_INTERVAL_S.
_last_fetch: dict[str, float] = {}
_fetch_lock = threading.Lock()


def _as_bool(value: object) -> bool:
    """The API returns isOnline as the *string* "true"/"false", not a bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def list_cameras(online_only: bool = True) -> list[Camera]:
    """Fetch the full NYC DOT camera catalog."""
    resp = _SESSION.get(config.NYCDOT_CAMERA_LIST_URL, timeout=config.REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()

    cameras: list[Camera] = []
    for item in payload:
        try:
            cam = Camera(
                id=str(item["id"]),
                name=str(item.get("name", "")).strip(),
                latitude=float(item.get("latitude", 0.0)),
                longitude=float(item.get("longitude", 0.0)),
                area=str(item.get("area", "")).strip(),
                image_url=str(
                    item.get("imageUrl")
                    or config.NYCDOT_IMAGE_URL.format(camera_id=item["id"])
                ),
                online=_as_bool(item.get("isOnline", False)),
            )
        except (KeyError, TypeError, ValueError):
            log.warning("skipping malformed camera record: %r", item)
            continue
        if online_only and not cam.online:
            continue
        cameras.append(cam)

    log.info("fetched %d cameras (online_only=%s)", len(cameras), online_only)
    return cameras


def demo_cameras(limit: int = config.MAX_DEMO_CAMERAS) -> list[Camera]:
    """Load the hand-curated demo camera list, verified against the live catalog.

    Curation matters more than it sounds: matching camera names against street
    keywords returns bridge decks, highway ramps and skyline-facing cameras,
    none of which can show double parking. The curated file was built by
    looking at actual frames. See data/demo_cameras.json.

    Falls back to name hints if the file is missing or every curated camera is
    offline, so the demo degrades rather than dies.
    """
    if not config.DEMO_CAMERAS_FILE.exists():
        log.warning("curated camera file missing; falling back to name hints")
        return find_cameras(limit=limit)

    with config.DEMO_CAMERAS_FILE.open(encoding="utf-8") as fh:
        curated = json.load(fh)

    # Cross-check against the live catalog: cameras get repositioned and drop
    # offline, and a stale hardcoded ID is a silent demo failure.
    live = {c.id: c for c in list_cameras(online_only=True)}
    picked: list[Camera] = []
    for entry in curated.get("cameras", []):
        cam = live.get(entry["id"])
        if cam is None:
            log.warning(
                "curated camera %r (%s) is offline or gone from the catalog",
                entry.get("name"), entry["id"],
            )
            continue
        picked.append(cam)
        if len(picked) >= limit:
            break

    if not picked:
        log.warning("no curated cameras available; falling back to name hints")
        return find_cameras(limit=limit)

    log.info("using %d curated demo cameras", len(picked))
    return picked


def find_cameras(
    hints: list[str] | None = None, limit: int = config.MAX_DEMO_CAMERAS
) -> list[Camera]:
    """Pick cameras whose name matches any of `hints` (case-insensitive).

    Fallback path only — prefer demo_cameras(). See its docstring for why.
    """
    hints = hints if hints is not None else config.DEFAULT_CAMERA_NAME_HINTS
    cameras = list_cameras(online_only=True)

    lowered = [h.lower() for h in hints]
    matched = [c for c in cameras if any(h in c.name.lower() for h in lowered)]

    if not matched:
        log.warning("no cameras matched hints %r; falling back to first online", hints)
        matched = cameras

    # Spread across distinct streets rather than 6 views of one intersection.
    seen_prefix: set[str] = set()
    picked: list[Camera] = []
    for cam in matched:
        prefix = cam.name.split("@")[0].strip().lower()
        if prefix in seen_prefix:
            continue
        seen_prefix.add(prefix)
        picked.append(cam)
        if len(picked) >= limit:
            break
    return picked


def _throttle(camera_id: str) -> None:
    """Block until this camera is eligible for another fetch."""
    with _fetch_lock:
        last = _last_fetch.get(camera_id, 0.0)
        wait = config.MIN_POLL_INTERVAL_S - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _last_fetch[camera_id] = time.monotonic()


def fetch_snapshot(camera: Camera, respect_throttle: bool = True) -> bytes | None:
    """Fetch one JPEG snapshot. Returns None if the camera is unavailable.

    Returning None rather than raising is deliberate: cameras drop offline
    routinely, and a live dashboard must skip them, not crash.
    """
    if respect_throttle:
        _throttle(camera.id)

    url = camera.image_url or config.NYCDOT_IMAGE_URL.format(camera_id=camera.id)

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(url, timeout=config.REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            content = resp.content
            if not content:
                log.warning("empty snapshot from %s", camera.short_name)
                return None
            # JPEG magic bytes. An HTML error page would otherwise sail through
            # and only fail later inside Pillow with a confusing message.
            if not content.startswith(b"\xff\xd8"):
                log.warning(
                    "non-JPEG response from %s (%d bytes, content-type=%s)",
                    camera.short_name,
                    len(content),
                    resp.headers.get("Content-Type"),
                )
                return None
            return content
        except requests.RequestException as exc:
            log.warning(
                "snapshot attempt %d/%d failed for %s: %s",
                attempt,
                config.MAX_RETRIES,
                camera.short_name,
                exc,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(2**attempt)

    return None
