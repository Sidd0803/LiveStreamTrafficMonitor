"""Collect the evaluation corpus: frames plus their raw detections.

Two things make this worth a dedicated script rather than a loop in a notebook.

**Detections are cached at a very low confidence floor.** Tuning
`MIN_BOX_CONFIDENCE` later then costs nothing — re-scoring is offline and
instant, instead of a fresh sweep and more load on a public civic feed. Bake
the production threshold into the capture and every future tuning question
needs new data.

**Sampling is stratified by borough.** Cameras are not evenly distributed and a
naive "first N" sample comes back overwhelmingly Manhattan, which would quietly
mean the eval set measures Manhattan rather than New York. Congestion looks
different on a Queens arterial than on a Midtown grid street, and a metric that
only works on one is not the metric we claim to have.

    PYTHONPATH=$PWD .venv/Scripts/python.exe scripts/collect.py --frames 150
    PYTHONPATH=$PWD .venv/Scripts/python.exe scripts/collect.py --frames 20 --curated
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.detect.roboflow_client import (
    RoboflowUnavailable,
    VehicleDetector,
    _find_predictions,
    _to_boxes,
)
from src.sources import nycdot

CORPUS_DIR = config.EVAL_DIR
FRAMES_DIR = CORPUS_DIR / "frames"
DETECTIONS_DIR = CORPUS_DIR / "detections"
MANIFEST = CORPUS_DIR / "manifest.jsonl"
LABELS = CORPUS_DIR / "labels.json"

# Cache well below config.MIN_BOX_CONFIDENCE so the threshold stays a tuning
# knob at analysis time rather than a property of the corpus.
COLLECT_CONFIDENCE = 0.05

FLOW_STATES = ("clear", "moderate", "jammed")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def stratified_sample(cameras: list, n: int, seed: int) -> list:
    """Pick n cameras spread as evenly as possible across boroughs.

    Round-robins through the boroughs rather than sampling proportionally: the
    goal is coverage of every borough's street character, not a sample that
    mirrors NYC DOT's own camera density (which is itself a Manhattan bias we
    have no reason to inherit).
    """
    by_area: dict[str, list] = defaultdict(list)
    for cam in cameras:
        by_area[cam.area or "Unknown"].append(cam)

    rng = random.Random(seed)
    for pool in by_area.values():
        rng.shuffle(pool)

    picked: list = []
    areas = sorted(by_area)
    while len(picked) < n and any(by_area[a] for a in areas):
        for area in areas:
            if by_area[area] and len(picked) < n:
                picked.append(by_area[area].pop())
    return picked


def collect(frames: int, curated: bool, seed: int) -> int:
    try:
        detector = VehicleDetector()
    except RoboflowUnavailable as exc:
        sys.exit(
            f"Roboflow unavailable: {exc}\n"
            "Set ROBOFLOW_API_KEY in .env — collection needs live detection."
        )

    if curated:
        cameras = nycdot.demo_cameras()
        print(f"Using {len(cameras)} curated cameras")
    else:
        catalog = nycdot.list_cameras(online_only=True)
        cameras = stratified_sample(catalog, frames, seed)
        spread = defaultdict(int)
        for cam in cameras:
            spread[cam.area or "Unknown"] += 1
        print(f"Sampled {len(cameras)} of {len(catalog)} online cameras")
        print("  " + ", ".join(f"{a}: {n}" for a, n in sorted(spread.items())))

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Widen the gate for capture only. _to_boxes reads this at call time, so
    # restoring it afterwards keeps the rest of the process honest.
    original = config.MIN_BOX_CONFIDENCE
    config.MIN_BOX_CONFIDENCE = COLLECT_CONFIDENCE

    written = 0
    try:
        with MANIFEST.open("a", encoding="utf-8") as manifest:
            for cam in cameras[:frames]:
                frame = nycdot.fetch_snapshot(cam)
                if frame is None:
                    print(f"  offline, skipped: {cam.short_name}")
                    continue

                captured = datetime.now(timezone.utc)
                frame_id = f"{slug(cam.short_name)}_{captured:%Y%m%dT%H%M%SZ}"

                try:
                    result = detector.detect(frame)
                except Exception as exc:  # noqa: BLE001 — one bad frame is not fatal
                    print(f"  detection failed, skipped: {cam.short_name} ({exc})")
                    continue

                (FRAMES_DIR / f"{frame_id}.jpg").write_bytes(frame)
                # Persist the raw response, not just our parsed boxes: if the
                # normalization logic changes later, the corpus stays valid.
                predictions = _find_predictions(result.raw) or []
                (DETECTIONS_DIR / f"{frame_id}.json").write_text(
                    json.dumps(
                        {
                            "frame_id": frame_id,
                            "model_id": result.model_id,
                            "inference_id": result.inference_id,
                            "via": result.via,
                            "collect_confidence": COLLECT_CONFIDENCE,
                            "predictions": predictions,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                manifest.write(
                    json.dumps(
                        {
                            "frame_id": frame_id,
                            "camera_id": cam.id,
                            "camera_name": cam.name,
                            "area": cam.area,
                            "latitude": cam.latitude,
                            "longitude": cam.longitude,
                            "captured_at": captured.isoformat(),
                        }
                    )
                    + "\n"
                )
                written += 1
                boxes = _to_boxes(predictions)
                print(f"  {frame_id}: {len(boxes)} vehicles @ conf>={COLLECT_CONFIDENCE}")
    finally:
        config.MIN_BOX_CONFIDENCE = original

    return written


def refresh_labels() -> None:
    """Add a blank label stub for every collected frame, keeping existing ones.

    Deliberately does **not** pre-fill the model's vehicle count. Anchoring a
    human labeller to the model's answer would quietly turn the eval set into a
    measure of agreement rather than of accuracy, and the count is the one
    number the whole index rests on.
    """
    existing: dict[str, dict] = {}
    if LABELS.exists():
        existing = json.loads(LABELS.read_text(encoding="utf-8"))

    frame_ids = sorted(p.stem for p in FRAMES_DIR.glob("*.jpg"))
    added = 0
    for frame_id in frame_ids:
        if frame_id not in existing:
            existing[frame_id] = {"vehicle_count": None, "flow_state": None, "notes": ""}
            added += 1

    LABELS.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    unlabeled = sum(1 for v in existing.values() if v.get("vehicle_count") is None)
    print(f"\n{LABELS.relative_to(config.REPO_ROOT)}: {added} new stubs, {unlabeled} unlabeled")
    print(f"Label each with a vehicle_count (int) and flow_state {FLOW_STATES}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=150, help="how many frames to collect")
    parser.add_argument(
        "--curated", action="store_true",
        help="use the hand-curated demo cameras instead of a stratified network sample",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed, for reproducibility")
    parser.add_argument(
        "--labels-only", action="store_true",
        help="refresh the label stubs without collecting anything new",
    )
    args = parser.parse_args()

    if not args.labels_only:
        written = collect(args.frames, args.curated, args.seed)
        print(f"\nCollected {written} frames into {CORPUS_DIR.relative_to(config.REPO_ROOT)}/")
    refresh_labels()


if __name__ == "__main__":
    main()
