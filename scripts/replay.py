"""Run the real pipeline over real consecutive frames — the first time the
tracker and heuristic touch anything that isn't synthetic.

Both modules pass their unit tests, but every box those tests use was
hand-built to make an assertion true. Nothing has been checked against actual
camera frames, and there are two specific reasons to expect trouble:

* `MIN_BOX_CONFIDENCE` is 0.40 while the detector's median confidence on these
  frames is ~0.25. A track needs four *consecutive* matches to count as
  stationary, so a vehicle flickering around the gate never accumulates.
* The fallback curb rule assumes the bottom band of the frame is curbside.
  These cameras look down the roadway, so the near field is travel lane.

Split into two phases on purpose:

    collect   poll a camera every ~12s, cache each frame AND its raw
              detections (unfiltered, low confidence) to disk
    analyze   replay the cache through tracker + heuristic and render it

Caching raw detections is what makes threshold tuning cheap: re-analyzing at a
different confidence is instant and offline, instead of a fresh two-minute poll
and more load on a public civic feed. It is also the frame-sequence corpus the
eval design needs (BUILD_PLAN amendment A4 wants poll sequences, not single
frames) and the replay mode that de-risks a live demo.

    PYTHONPATH=$PWD .venv/bin/python scripts/replay.py --polls 10
    PYTHONPATH=$PWD .venv/bin/python scripts/replay.py --analyze-only --confidence 0.15
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from src import config
from src.detect.boxes import Box
from src.detect.heuristic import find_candidates
from src.detect.roboflow_client import (
    RoboflowUnavailable,
    VehicleDetector,
    _find_predictions,
    _to_boxes,
)
from src.detect.tracker import CameraTracker
from src.sources import nycdot

REPLAY_DIR = config.REPO_ROOT / "runs" / "replay"

# Cache detections well below MIN_BOX_CONFIDENCE so the threshold stays a
# *tuning* knob at analysis time rather than being baked into the capture.
COLLECT_CONFIDENCE = 0.05

STATIONARY_COLOR = (60, 220, 90)
MOVING_COLOR = (120, 170, 255)
CANDIDATE_COLOR = (255, 60, 60)
CURB_COLOR = (255, 200, 40)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --- collect ---------------------------------------------------------------

def collect(camera_names: list[str] | None, polls: int, session: Path) -> None:
    """Poll live cameras and cache frames + raw detections."""
    try:
        detector = VehicleDetector()
    except RoboflowUnavailable as exc:
        sys.exit(f"Roboflow unavailable: {exc}")

    # Widen the confidence gate for capture only. _to_boxes reads this at call
    # time, so restoring it afterwards keeps the rest of the process honest.
    original = config.MIN_BOX_CONFIDENCE
    config.MIN_BOX_CONFIDENCE = COLLECT_CONFIDENCE

    try:
        cameras = nycdot.demo_cameras()
        if camera_names:
            wanted = {n.lower() for n in camera_names}
            cameras = [c for c in cameras if c.name.lower() in wanted]
            if not cameras:
                sys.exit(f"no curated camera matched {camera_names}")

        for cam in cameras:
            out = session / slug(cam.name)
            out.mkdir(parents=True, exist_ok=True)
            (out / "camera.json").write_text(
                json.dumps({"id": cam.id, "name": cam.name, "area": cam.area}, indent=2)
            )
            print(f"\n=== collecting {polls} polls from {cam.name} "
                  f"(~{polls * config.MIN_POLL_INTERVAL_S / 60:.1f} min) ===")

            for i in range(polls):
                t0 = time.time()
                # fetch_snapshot self-throttles to MIN_POLL_INTERVAL_S per camera.
                frame = nycdot.fetch_snapshot(cam)
                if frame is None:
                    print(f"  poll {i:02d}: snapshot unavailable, skipping")
                    continue

                (out / f"frame_{i:02d}.jpg").write_bytes(frame)
                try:
                    result = detector.detect(frame)
                    raw = _find_predictions(result.raw) or []
                except Exception as exc:  # noqa: BLE001 — one bad poll must not end the run
                    print(f"  poll {i:02d}: detection failed ({exc})")
                    raw = []

                (out / f"dets_{i:02d}.json").write_text(json.dumps(raw, indent=2))
                kept = len(_to_boxes(raw))
                print(f"  poll {i:02d}: {len(frame):6d}B  {len(raw):3d} raw dets  "
                      f"{kept:2d} vehicles  ({time.time() - t0:4.1f}s)")
    finally:
        config.MIN_BOX_CONFIDENCE = original


# --- analyze ---------------------------------------------------------------

@dataclass
class PollRow:
    poll: int
    detections: int
    tracks: int
    stationary: int
    candidates: int
    suppressed: bool


def annotate(
    frame_path: Path,
    tracks,
    candidate_ids: set[int],
    out_path: Path,
    frame_w: int,
    frame_h: int,
    poll: int,
    congested: bool,
) -> None:
    img = Image.open(frame_path).convert("RGB")
    scale = 3
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    # The curb band the fallback heuristic uses. Everything at or below this
    # line is assumed curbside and skipped -- worth seeing where it lands.
    curb_y = frame_h * (1.0 - config.CURB_BAND_FRACTION) * scale
    draw.line([(0, curb_y), (img.width, curb_y)], fill=CURB_COLOR, width=2)
    draw.text((4, curb_y + 3), "curb band -> everything below is ignored", fill=CURB_COLOR)

    for t in tracks:
        if t.misses > 0:
            continue
        if t.id in candidate_ids:
            color, width = CANDIDATE_COLOR, 3
        elif t.is_stationary:
            color, width = STATIONARY_COLOR, 2
        else:
            color, width = MOVING_COLOR, 1
        b = t.box
        draw.rectangle([b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale],
                       outline=color, width=width)
        draw.text((b.x1 * scale + 2, max(0, b.y1 * scale - 11)),
                  f"T{t.id} s{t.stationary_polls}", fill=color)

    header = f"poll {poll}  |  green=stationary  blue=moving  red=CANDIDATE"
    if congested:
        header += "  |  CONGESTION: all candidates suppressed"
    draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    draw.text((4, 3), header, fill=(255, 255, 255))
    img.save(out_path)


def analyze(session: Path, confidence: float | None) -> None:
    if confidence is not None:
        print(f"(overriding MIN_BOX_CONFIDENCE {config.MIN_BOX_CONFIDENCE} -> {confidence})")
        config.MIN_BOX_CONFIDENCE = confidence

    cam_dirs = sorted(d for d in session.iterdir() if d.is_dir())
    if not cam_dirs:
        sys.exit(f"no cached cameras under {session}")

    for cam_dir in cam_dirs:
        meta = json.loads((cam_dir / "camera.json").read_text())
        frames = sorted(cam_dir.glob("frame_*.jpg"))
        if not frames:
            continue

        print(f"\n=== {meta['name']} — {len(frames)} cached polls ===")
        print(f"config: conf>={config.MIN_BOX_CONFIDENCE}  IoU>={config.IOU_MATCH_THRESHOLD}  "
              f"stationary_polls>={config.STATIONARY_POLLS}  curb_band={config.CURB_BAND_FRACTION}")

        tracker = CameraTracker(meta["id"])
        rows: list[PollRow] = []
        ann_dir = cam_dir / "annotated"
        ann_dir.mkdir(exist_ok=True)

        for fp in frames:
            i = int(fp.stem.split("_")[1])
            raw = json.loads((cam_dir / f"dets_{i:02d}.json").read_text())
            boxes: list[Box] = _to_boxes(raw)

            with Image.open(fp) as im:
                fw, fh = im.size

            tracks = tracker.update(boxes)
            candidates = find_candidates(meta["id"], tracks, fw, fh)
            cand_ids = {c.track.id for c in candidates}

            stationary = [t for t in tracks if t.is_stationary and t.misses == 0]
            # find_candidates suppresses everything when the frame reads as
            # congested; reproduce that test here purely for reporting.
            congested = (
                len(tracks) >= config.CONGESTION_MIN_TRACKS
                and len(stationary) / len(tracks) >= config.CONGESTION_STATIONARY_FRACTION
            )

            rows.append(PollRow(i, len(boxes), len(tracks), len(stationary),
                                len(candidates), congested))
            annotate(fp, tracks, cand_ids, ann_dir / f"poll_{i:02d}.png",
                     fw, fh, i, congested)

        print(f"\n{'poll':>4} {'dets':>5} {'tracks':>7} {'stationary':>11} {'cands':>6}  note")
        for r in rows:
            note = "congestion -> suppressed" if r.suppressed else ""
            print(f"{r.poll:>4} {r.detections:>5} {r.tracks:>7} {r.stationary:>11} "
                  f"{r.candidates:>6}  {note}")

        # The headline diagnostic: did anything ever hold position long enough?
        best = max((t.stationary_polls for t in tracker.tracks), default=0)
        ever_stationary = any(r.stationary for r in rows)
        ever_candidate = any(r.candidates for r in rows)
        print(f"\n  longest surviving track: {best} consecutive polls "
              f"(need {config.STATIONARY_POLLS})")
        print(f"  any track reached stationary: {ever_stationary}")
        print(f"  any candidate emitted:        {ever_candidate}")

        ious = [v for t in tracker.tracks for v in t.iou_history]
        if ious:
            ious.sort()
            print(f"  match IoU across {len(ious)} matches: "
                  f"min {ious[0]:.2f}  median {ious[len(ious)//2]:.2f}  max {ious[-1]:.2f}  "
                  f"(threshold {config.IOU_MATCH_THRESHOLD})")
        print(f"  annotated frames -> {ann_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", action="append",
                    help="curated camera name; repeatable. Default: the best one.")
    ap.add_argument("--all-cameras", action="store_true")
    ap.add_argument("--polls", type=int, default=8)
    ap.add_argument("--session", default="latest")
    ap.add_argument("--analyze-only", action="store_true",
                    help="replay an existing cache without polling")
    ap.add_argument("--confidence", type=float,
                    help="override MIN_BOX_CONFIDENCE for analysis only")
    args = ap.parse_args()

    session = REPLAY_DIR / args.session

    if not args.analyze_only:
        names = args.camera
        if not names and not args.all_cameras:
            names = ["Amsterdam Ave @ 60 St"]  # the best camera per data/demo_cameras.json
        session.mkdir(parents=True, exist_ok=True)
        collect(names, args.polls, session)

    if not session.exists():
        sys.exit(f"no cached session at {session} — run without --analyze-only first")
    analyze(session, args.confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
