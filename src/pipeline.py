"""One camera, one instant, one Observation — and the sweep that does it widely.

This is the only module that knows the whole story, so it is also the only one
that decides what happens when a piece of it fails. The rule throughout is that
a degraded observation beats a missing one:

* Camera offline -> return None. Cameras drop out routinely; that is not an error.
* Detection fails -> no observation. Counting is the base signal; without it
  there is nothing honest to record.
* Gemini fails -> keep the observation with geometry only. The geometric
  verdict stands on its own, and losing the whole row because the adjudicator
  was unavailable would be a worse outcome than losing its opinion.

Both verdicts are always written, never just the fused one. That is what makes
the geometry-vs-Gemini ablation free at eval time — see BUILD_PLAN.md.

**Gemini is rationed, not run on every frame.** The free tier allows 20
requests per day per model against sweeps that span hundreds of cameras, so a
call-everything design fails after the twentieth camera. The sweep therefore
measures every camera geometrically, then spends its Gemini budget on the
frames nearest a classification boundary — where a second opinion actually
changes something. A frame with no vehicles does not need adjudicating.
"""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from PIL import Image

from src import config
from src.analyze.density import FrameMetrics, frame_metrics
from src.analyze.gemini_scene import SceneClassifier, SceneVerdict
from src.detect.roboflow_client import VehicleDetector
from src.sources import nycdot
from src.sources.nycdot import Camera
from src.storage.base import Observation, Store

log = logging.getLogger(__name__)

FRAMES_DIR = config.RUNS_DIR / "frames"


@dataclass
class _Measured:
    """A camera measured geometrically, with its frame kept for adjudication."""

    camera: Camera
    frame: bytes
    captured: datetime
    metrics: FrameMetrics


def ambiguity(metrics: FrameMetrics) -> float:
    """How much a second opinion would be worth on this frame, 0..1.

    Ranks by proximity to a classification boundary. A frame sitting exactly on
    the jammed/moderate line is where geometry is least trustworthy and where
    Gemini's read of road capacity has the most to add; a frame with three
    vehicles a dozen car-lengths apart is unambiguous and would waste a call.

    Used only to spend a limited budget well. It has no effect on the verdict.
    """
    if metrics.crowding is None:
        # Too few vehicles to compute a ratio — trivially clear, nothing to argue.
        return 0.0
    distance = min(
        abs(metrics.crowding - config.CROWDING_JAMMED_MAX),
        abs(metrics.crowding - config.CROWDING_MODERATE_MAX),
    )
    return 1.0 / (1.0 + distance)


def fuse(geometric: str, verdict: SceneVerdict | None) -> str:
    """Pick the flow state to report.

    Gemini wins when it is confident, because it can weigh road width and lane
    count against vehicle density — the judgement the geometry cannot make.
    Below the confidence floor we fall back rather than average: these are
    ordinal categories, and a "mean" of clear and jammed is not moderate, it is
    a number nobody can act on.
    """
    if verdict is None or verdict.confidence < config.GEMINI_MIN_CONFIDENCE:
        return geometric
    return verdict.flow_state


def classify_with_retry(
    classifier: SceneClassifier, frame: bytes, label: str
) -> SceneVerdict | None:
    """Call Gemini, retrying transport failures. None when it stays unavailable.

    Gemini 3 models return 503 under load often enough that a single attempt
    leaves holes in the map for reasons that have nothing to do with traffic.
    Content problems are already absorbed inside the classifier, so anything
    reaching here is transport, quota, or a genuine outage — all worth one more
    try, none worth losing the observation over.
    """
    for attempt in range(config.GEMINI_MAX_RETRIES):
        try:
            return classifier.classify(frame)
        except Exception as exc:  # noqa: BLE001 — degrade to geometry only
            last = attempt == config.GEMINI_MAX_RETRIES - 1
            if last:
                log.warning("Gemini gave up on %s, keeping geometry: %s", label, exc)
                return None
            time.sleep(config.GEMINI_RETRY_BASE_S * (2**attempt))
    return None


def _save_frame(frame: bytes, camera_id: str, captured: datetime) -> str | None:
    """Cache the frame so the dashboard can show what a verdict was based on."""
    try:
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        path = FRAMES_DIR / f"{camera_id}_{captured:%Y%m%dT%H%M%SZ}.jpg"
        path.write_bytes(frame)
        return str(path.relative_to(config.REPO_ROOT))
    except OSError as exc:
        # A full disk should cost us the picture, not the measurement.
        log.warning("could not cache frame for %s: %s", camera_id, exc)
        return None


def _measure(camera: Camera, detector: VehicleDetector) -> _Measured | None:
    """Fetch and measure one camera. None when there is nothing to record."""
    frame = nycdot.fetch_snapshot(camera)
    if frame is None:
        log.info("no frame from %s — offline or throttled", camera.short_name)
        return None

    captured = datetime.now(timezone.utc)

    try:
        detection = detector.detect(frame)
    except Exception as exc:  # noqa: BLE001 — one camera must not end a sweep
        log.warning("detection failed for %s: %s", camera.short_name, exc)
        return None

    try:
        width, height = Image.open(io.BytesIO(frame)).size
    except Exception as exc:  # noqa: BLE001
        log.warning("unreadable frame from %s: %s", camera.short_name, exc)
        return None

    return _Measured(
        camera=camera,
        frame=frame,
        captured=captured,
        metrics=frame_metrics(detection.boxes, width, height),
    )


def _to_observation(
    measured: _Measured, verdict: SceneVerdict | None, save_frame: bool
) -> Observation:
    metrics = measured.metrics
    camera = measured.camera
    return Observation(
        camera_id=camera.id,
        camera_name=camera.name,
        latitude=camera.latitude,
        longitude=camera.longitude,
        area=camera.area,
        captured_at=measured.captured,
        vehicle_count=metrics.vehicle_count,
        occupancy=metrics.occupancy,
        crowding=metrics.crowding,
        by_class=metrics.by_class,
        geometric_flow=metrics.flow_state,
        gemini_flow=verdict.flow_state if verdict else None,
        gemini_confidence=verdict.confidence if verdict else None,
        gemini_reason=verdict.reason if verdict else None,
        gemini_notable=verdict.notable if verdict else None,
        final_flow=fuse(metrics.flow_state, verdict),
        frame_path=(
            _save_frame(measured.frame, camera.id, measured.captured)
            if save_frame
            else None
        ),
    )


def observe(
    camera: Camera,
    detector: VehicleDetector,
    classifier: SceneClassifier | None = None,
    store: Store | None = None,
    save_frame: bool = True,
) -> Observation | None:
    """Fetch, measure and judge one camera.

    Unlike `sweep`, this always calls Gemini when a classifier is supplied —
    a single deliberate observation is not the thing that exhausts a quota.
    """
    measured = _measure(camera, detector)
    if measured is None:
        return None

    verdict = (
        classify_with_retry(classifier, measured.frame, camera.short_name)
        if classifier is not None
        else None
    )
    observation = _to_observation(measured, verdict, save_frame)

    if store is not None:
        store.save(observation)
    return observation


def sweep(
    cameras: list[Camera],
    detector: VehicleDetector,
    classifier: SceneClassifier | None = None,
    store: Store | None = None,
    concurrency: int | None = None,
    save_frame: bool = True,
    gemini_budget: int | None = None,
) -> list[Observation]:
    """Observe many cameras, spending a limited Gemini budget where it counts.

    Two phases. Detection fans out across every camera; adjudication then runs
    on the `gemini_budget` most ambiguous frames only. Splitting them is what
    makes the budget spendable at all — you cannot know which frames are near a
    boundary until every frame has been measured.

    Threads rather than processes: every stage is network-bound, so the GIL is
    released while waiting and processes would buy nothing. Concurrency stays
    modest — `nycdot.fetch_snapshot` throttles per camera, not globally, so
    nothing else stops a wide sweep from leaning on a public civic feed harder
    than we would want to explain.
    """
    workers = concurrency or config.SWEEP_CONCURRENCY
    budget = config.GEMINI_SWEEP_BUDGET if gemini_budget is None else gemini_budget

    measured: list[_Measured] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_measure, cam, detector): cam for cam in cameras}
        for future in as_completed(futures):
            camera = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("sweep failed for %s: %s", camera.short_name, exc)
                continue
            if result is not None:
                measured.append(result)

    verdicts: dict[str, SceneVerdict] = {}
    if classifier is not None and budget > 0:
        ranked = sorted(measured, key=ambiguity_of, reverse=True)[:budget]
        log.info("adjudicating %d of %d frames with Gemini", len(ranked), len(measured))
        for item in ranked:
            verdict = classify_with_retry(classifier, item.frame, item.camera.short_name)
            if verdict is not None:
                verdicts[item.camera.id] = verdict

    observations = [
        _to_observation(item, verdicts.get(item.camera.id), save_frame)
        for item in measured
    ]

    # Written in one batch after the fan-out rather than from each worker, so
    # the store does not need to be thread-safe to be correct.
    if store is not None and observations:
        store.save_many(observations)

    observations.sort(key=lambda o: o.camera_name)
    return observations


def ambiguity_of(measured: _Measured) -> float:
    """Sort key wrapper — kept separate so `ambiguity` stays testable on metrics."""
    return ambiguity(measured.metrics)
