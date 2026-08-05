"""Central configuration. Every tunable threshold lives here, not inline.

Phase 2 involves repeatedly tuning IOU_MATCH_THRESHOLD, STATIONARY_POLLS, and
the confidence gates against the labeled eval set, so keep them in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ZONES_DIR = DATA_DIR / "zones"
RUNS_DIR = REPO_ROOT / "runs"
EVAL_DIR = REPO_ROOT / "eval"

# --- Credentials -----------------------------------------------------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

# Direct model inference. Used when no workflow is configured, and as the
# fallback if the workflow call fails.
#
# COCO-pretrained YOLOv8-medium, benchmarked 2026-08-05 against the same live
# frames as vehicle-detection-bz0yu/4 (the model Roboflow originally suggested):
# nearly twice the detections above 0.40 confidence, a higher median confidence
# (0.25 vs 0.19), and ~4x faster. It also has a real "truck" class — the other
# model had only "semi-trailer", and box trucks are the archetypal double-parker.
#
# The prior model was a 203-image public Universe project of unverifiable
# provenance whose training run predated the project itself, with preprocessing
# that stretches input to 640x640 — badly distorting our 352x240 frames.
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "yolov8m-640")

# Hosted Workflow: "one image in, vehicle boxes out". Deliberately detection
# only — tracking, zone logic and incident detection stay in Python, because
# Roboflow's stateful blocks (ByteTrack, Time in Zone) need continuous video
# and our frames are ~12s apart.
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "")
# Input/output key names as defined in the workflow itself.
ROBOFLOW_WORKFLOW_IMAGE_KEY = os.getenv("ROBOFLOW_WORKFLOW_IMAGE_KEY", "image")
ROBOFLOW_WORKFLOW_OUTPUT_KEY = os.getenv("ROBOFLOW_WORKFLOW_OUTPUT_KEY", "vehicle_boxes")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

NY511_API_KEY = os.getenv("NY511_API_KEY", "")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")

# --- Data sources ----------------------------------------------------------
NYCDOT_CAMERA_LIST_URL = "https://webcams.nyctmc.org/api/cameras"
NYCDOT_IMAGE_URL = "https://webcams.nyctmc.org/api/cameras/{camera_id}/image"
NY511_CAMERA_URL = "https://511ny.org/api/getcameras"
NY511_EVENT_URL = "https://511ny.org/api/getevents"

# Identify ourselves. These are public civic feeds; be a good citizen.
USER_AGENT = "double-parking-detector/0.1 (hackathon learning project)"

# Hard floor on polling. NYC DOT snapshots refresh every ~1-2s, but we have no
# need to hammer them and no right to. One poll per camera per this interval.
MIN_POLL_INTERVAL_S = 12.0
REQUEST_TIMEOUT_S = 15.0
MAX_RETRIES = 3

# 511NY throttles at 10 calls / 60s. Only relevant once a key exists.
NY511_MIN_INTERVAL_S = 7.0

# --- Detection tunables ----------------------------------------------------
# Roboflow: drop low-confidence boxes before they reach the tracker.
MIN_BOX_CONFIDENCE = 0.40

# Word-level vocabulary for "a vehicle that could be double parked".
#
# Matching is per-word, not whole-string, because model label vocabularies vary
# more than they look. vehicle-detection-bz0yu emits "pickup-truck",
# "semi-trailer" and "vehicles"; an exact-match set containing "truck",
# "pickup" and "vehicle" silently dropped all three. On NYC local streets
# "semi-trailer" is this model's label for box and delivery trucks — the single
# most common double-parker — so the filter was discarding the primary target.
#
# See is_vehicle_label() in detect/roboflow_client.py for the matching rule:
# labels are lowercased, split on -/_/space, and de-pluralized before lookup.
VEHICLE_WORDS = {
    "car", "truck", "bus", "van", "vehicle", "motorcycle", "motorbike",
    "suv", "pickup", "taxi", "cab", "auto", "automobile", "jeep",
    "trailer", "semi", "lorry", "minivan", "coach", "ambulance",
}

# Words that veto a label even if a vehicle word is also present. Guards
# against things like "truck-stop-sign" or "bus stop" being read as vehicles.
NON_VEHICLE_WORDS = {"stop", "sign", "lane", "light", "signal", "person", "pedestrian"}

# Tracker: two boxes in consecutive polls are "the same vehicle" above this IoU.
# High, because a genuinely stationary vehicle barely moves between polls.
IOU_MATCH_THRESHOLD = 0.60

# Consecutive polls a track must hold position before it counts as stationary.
# At MIN_POLL_INTERVAL_S=12s, 4 polls is roughly 48-60s of not moving.
STATIONARY_POLLS = 4

# Drop a track from memory after this many consecutive polls without a match.
TRACK_MAX_MISSES = 2

# Heuristic: a box whose center sits within this fraction of the frame height
# from the bottom edge is likely curbside (near-field parking) rather than in a
# travel lane. Per-camera zone polygons in data/zones/ override this entirely.
CURB_BAND_FRACTION = 0.18

# Ignore boxes smaller than this fraction of frame area — distant noise.
MIN_BOX_AREA_FRACTION = 0.004

# Congestion guard. At a red light every vehicle is stationary in a travel
# lane, which a naive rule reads as a frame full of double-parking. If at
# least this fraction of tracks are stationary, treat the frame as congested
# and emit nothing — double parking means stopped *while traffic moves*.
CONGESTION_STATIONARY_FRACTION = 0.70
# Below this many tracks the fraction is too noisy to be meaningful.
CONGESTION_MIN_TRACKS = 4

# --- Gemini adjudication ---------------------------------------------------
# Pixels of context to include around a candidate box when cropping. A bare
# box gives the model no road context to judge lane position from.
CROP_PADDING_PX = 60
# Upscale small crops before sending; traffic-cam crops are tiny.
CROP_MIN_DIM_PX = 320
# Below this, treat Gemini's positive verdict as too weak to report.
GEMINI_MIN_CONFIDENCE = 0.55

# --- Demo defaults ---------------------------------------------------------
# Hand-curated camera list, selected by visually inspecting live frames.
# This file is the source of truth for which cameras the demo uses.
DEMO_CAMERAS_FILE = DATA_DIR / "demo_cameras.json"

# Fallback only, used if the curated file is missing. Name matching is a poor
# selector — it happily returns bridge decks and highway ramps, where double
# parking cannot occur. Prefer the curated file.
DEFAULT_CAMERA_NAME_HINTS = [
    "Amsterdam Ave",
    "Northern Blvd",
    "Grand Concourse",
    "Atlantic Ave",
    "Columbus Ave",
    "Flatbush Ave",
]
MAX_DEMO_CAMERAS = 6
