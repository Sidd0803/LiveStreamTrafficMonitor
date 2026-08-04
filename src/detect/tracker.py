"""Stationarity tracking by IoU matching across polls.

Deliberately not ByteTrack/DeepSORT. Those carry a motion model tuned for
continuous video at 20-30fps; our frames are ~12s apart, so any velocity
estimate between them is meaningless. What we actually need is much simpler:
"is this the same vehicle, and has it failed to move?"

A vehicle that has not moved for STATIONARY_POLLS consecutive polls while
traffic flows around it is our temporal signal for double parking.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

from src import config
from src.detect.boxes import Box, iou

log = logging.getLogger(__name__)

_track_ids = itertools.count(1)


@dataclass
class Track:
    """One vehicle followed across polls."""

    id: int
    box: Box
    #: Consecutive polls this track has been matched while holding position.
    stationary_polls: int = 1
    #: Consecutive polls this track went unmatched. Cleared on any match.
    misses: int = 0
    #: Total polls this track has been seen at all.
    age: int = 1
    #: IoU against the previous poll, per poll. Useful for debugging drift.
    iou_history: list[float] = field(default_factory=list)

    @property
    def is_stationary(self) -> bool:
        return self.stationary_polls >= config.STATIONARY_POLLS

    @property
    def stationary_seconds(self) -> float:
        """Approximate dwell time, for display and incident records."""
        return self.stationary_polls * config.MIN_POLL_INTERVAL_S


class CameraTracker:
    """Per-camera tracker state. One instance per camera.

    Not thread-safe; the pipeline drives one camera at a time.
    """

    def __init__(self, camera_id: str, iou_threshold: float | None = None) -> None:
        self.camera_id = camera_id
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else config.IOU_MATCH_THRESHOLD
        )
        self.tracks: list[Track] = []

    def update(self, detections: list[Box]) -> list[Track]:
        """Advance one poll. Returns all live tracks after the update.

        Matching is greedy on descending IoU. With a high threshold (~0.6) and
        a dozen well-separated vehicles, greedy and optimal assignment agree
        almost always, and greedy is far easier to reason about when a demo
        misbehaves.
        """
        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = iou(track.box, det)
                if score >= self.iou_threshold:
                    candidates.append((score, ti, di))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for score, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)

            track = self.tracks[ti]
            # Keep the newest box so slow drift doesn't accumulate against a
            # stale anchor, but count the poll as stationary: clearing the
            # threshold means it barely moved.
            track.box = detections[di]
            track.stationary_polls += 1
            track.misses = 0
            track.age += 1
            track.iou_history.append(round(score, 3))

        # Unmatched tracks: either the vehicle left, or detection dropped it
        # for one poll. Tolerate TRACK_MAX_MISSES before forgetting it, but do
        # NOT keep incrementing stationary_polls through the gap.
        surviving: list[Track] = []
        for ti, track in enumerate(self.tracks):
            if ti in matched_tracks:
                surviving.append(track)
                continue
            track.misses += 1
            track.age += 1
            if track.misses <= config.TRACK_MAX_MISSES:
                surviving.append(track)

        # Unmatched detections become new tracks.
        for di, det in enumerate(detections):
            if di not in matched_dets:
                surviving.append(Track(id=next(_track_ids), box=det))

        self.tracks = surviving
        return list(self.tracks)

    def stationary_tracks(self) -> list[Track]:
        """Tracks that have held position long enough to be candidates."""
        return [t for t in self.tracks if t.is_stationary and t.misses == 0]

    def reset(self) -> None:
        self.tracks = []
