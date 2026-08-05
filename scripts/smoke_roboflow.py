"""Smoke test: does Roboflow actually resolve vehicles in a 352x240 traffic-cam frame?

This answers the project's biggest open question. `roboflow_client.py` is
unit-tested against synthetic responses but has never made a real API call, and
a mid-frame vehicle here is roughly 40x30px — well below what Universe models
trained on high-res photos usually see.

Numbers alone can't settle it: an empty result and a correct result both "run
fine". So this also writes an annotated PNG. Count the boxes against what you
can actually see in the frame.

    PYTHONPATH=$PWD .venv/bin/python scripts/smoke_roboflow.py
    PYTHONPATH=$PWD .venv/bin/python scripts/smoke_roboflow.py --all-cameras
"""

from __future__ import annotations

import argparse
import io
import sys

from PIL import Image, ImageDraw

from src import config
from src.detect.boxes import Box
from src.detect.roboflow_client import RoboflowUnavailable, VehicleDetector, _to_boxes
from src.sources import nycdot

OUT_DIR = config.REPO_ROOT / "runs" / "smoke"


def annotate(frame: bytes, boxes: list[Box], path) -> None:
    """Draw boxes on the frame, upscaled 3x so they're actually inspectable."""
    img = Image.open(io.BytesIO(frame)).convert("RGB")
    scale = 3
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes, 1):
        draw.rectangle(
            [b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale],
            outline=(255, 60, 60),
            width=2,
        )
        draw.text(
            (b.x1 * scale + 2, max(0, b.y1 * scale - 11)),
            f"{i} {b.label} {b.confidence:.2f}",
            fill=(255, 220, 60),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def report(name: str, frame: bytes, detector: VehicleDetector) -> int:
    img = Image.open(io.BytesIO(frame))
    w, h = img.size
    print(f"\n=== {name} — {w}x{h}, {len(frame)} bytes ===")

    result = detector.detect(frame)
    boxes = result.boxes

    # Boxes that survived the class/confidence filters vs. what the model
    # actually returned. A big gap means the filters are the problem, not the
    # model -- worth knowing before concluding the model can't see anything.
    raw_preds = []
    from src.detect.roboflow_client import _find_predictions

    found = _find_predictions(result.raw)
    if found:
        raw_preds = found

    print(f"path: {result.via}  model: {result.model_id}")
    print(f"raw predictions returned: {len(raw_preds)}")
    print(f"boxes after confidence>={config.MIN_BOX_CONFIDENCE} + vehicle-class filter: {len(boxes)}")

    if raw_preds and not boxes:
        labels = sorted({str(p.get("class", "?")).lower() for p in raw_preds})
        confs = [float(p.get("confidence", 0)) for p in raw_preds]
        print("  !! everything was filtered out.")
        print(f"     classes seen: {labels}")
        print(f"     confidence range: {min(confs):.2f}-{max(confs):.2f}")
        print(f"     VEHICLE_WORDS = {sorted(config.VEHICLE_WORDS)}")

    frame_area = float(w * h)
    for i, b in enumerate(boxes, 1):
        frac = b.area / frame_area
        floor = "" if frac >= config.MIN_BOX_AREA_FRACTION else "  <-- below MIN_BOX_AREA_FRACTION"
        print(
            f"  {i:2d}. {b.label:12s} conf={b.confidence:.2f} "
            f"{b.width:5.1f}x{b.height:5.1f}px  area={frac*100:5.2f}%{floor}"
        )

    out = OUT_DIR / f"{name.replace(' ', '_').replace('@', 'at').replace('/', '-')}.png"
    annotate(frame, boxes, out)
    print(f"annotated -> {out}")
    return len(boxes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-cameras", action="store_true",
                    help="test every curated demo camera, not just the best one")
    args = ap.parse_args()

    if not config.ROBOFLOW_API_KEY:
        print("ROBOFLOW_API_KEY is not set. Fill it into .env first.", file=sys.stderr)
        return 2

    try:
        detector = VehicleDetector()
    except RoboflowUnavailable as exc:
        print(f"Roboflow unavailable: {exc}", file=sys.stderr)
        return 2

    print(f"api_url:  {config.ROBOFLOW_API_URL}")
    print(f"workflow: {'yes' if detector.uses_workflow else 'no (direct model inference)'}")

    cameras = nycdot.demo_cameras()
    if not args.all_cameras:
        cameras = cameras[:1]

    total = 0
    for cam in cameras:
        frame = nycdot.fetch_snapshot(cam)
        if frame is None:
            print(f"\n=== {cam.name} — snapshot unavailable, skipping ===")
            continue
        total += report(cam.name, frame, detector)

    print(f"\n{total} vehicle box(es) across {len(cameras)} camera(s).")
    print("Open the annotated PNG(s) and count against what you can see. "
          "That comparison is the actual result -- not the count above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
