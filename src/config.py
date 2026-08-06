"""Central configuration. Every tunable threshold lives here, not inline.

Phase 2 involves repeatedly tuning the confidence gate and the crowding
thresholds against the labeled eval set, so keep them in one place.
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
# only — the congestion metrics and scene classification stay in Python, where
# they can be unit tested and tuned offline against cached detections.
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
# Roboflow: drop low-confidence boxes before they reach the metrics.
#
# This gate matters more here than it looks. Counting is the base signal, so a
# threshold that drops real vehicles biases every downstream metric toward
# "clear" — and does it silently. Tune against the labeled counts, not by feel.
MIN_BOX_CONFIDENCE = 0.40

# Word-level vocabulary for "a road vehicle we should count".
#
# Matching is per-word, not whole-string, because model label vocabularies vary
# more than they look. vehicle-detection-bz0yu emits "pickup-truck",
# "semi-trailer" and "vehicles"; an exact-match set containing "truck",
# "pickup" and "vehicle" silently dropped all three — 9 delivery trucks
# discarded across four cameras, with no error and no trace.
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

# Ignore boxes smaller than this fraction of frame area — distant noise.
MIN_BOX_AREA_FRACTION = 0.004

# --- Congestion metrics ----------------------------------------------------
# Below this many vehicles a frame is trivially clear and the crowding ratio is
# too noisy to mean anything — one car has no nearest neighbour, two cars give
# a single sample. Report "clear" without computing a ratio.
MIN_VEHICLES_FOR_CROWDING = 3

# Crowding is the core signal: for each vehicle, the distance to its nearest
# neighbour divided by *that vehicle's own* box diagonal, then the median over
# all vehicles. Both terms shrink with distance from the camera, so the ratio
# is scale-free and needs no per-camera calibration — which is what lets this
# run across all ~950 cameras instead of the handful we could hand-tune.
#
# Low ratio = packed. A value near 1.0 means vehicles are roughly one car-length
# apart, i.e. bumper to bumper. Thresholds below are the initial guess and are
# expected to move once the labeled set exists — do not treat them as measured.
CROWDING_JAMMED_MAX = 1.6
CROWDING_MODERATE_MAX = 3.0

# --- Gemini scene classification -------------------------------------------
# Upscale frames before sending; 352x240 is small enough that detail is lost in
# the model's own preprocessing.
SCENE_MIN_DIM_PX = 704
# Below this, treat Gemini's flow-state verdict as too weak to rely on and fall
# back to the geometric classification.
GEMINI_MIN_CONFIDENCE = 0.55

# --- Camera selection ------------------------------------------------------
# Hand-curated list, selected by visually inspecting live frames. Still the
# best set for close-up work and demos.
#
# Note the curation criteria loosened with the change of problem: that list
# rejected bridge decks and highway ramps because double parking cannot happen
# there. Congestion certainly can, so those rejections no longer apply and the
# network-wide sweep deliberately ignores the curated file. What still
# disqualifies a camera is an unusable *view* — skyline framing, a hazed lens —
# not the road type. See data/demo_cameras.json for the recorded reasons.
DEMO_CAMERAS_FILE = DATA_DIR / "demo_cameras.json"
MAX_DEMO_CAMERAS = 6

# Fallback only, used if the curated file is missing.
DEFAULT_CAMERA_NAME_HINTS = [
    "Amsterdam Ave",
    "Northern Blvd",
    "Grand Concourse",
    "Atlantic Ave",
    "Columbus Ave",
    "Flatbush Ave",
]

# Network-wide sweep. The whole point of a scale-free metric is that it runs
# everywhere without per-camera calibration, so the sweep is capped by appetite
# for API calls rather than by curation. At MIN_POLL_INTERVAL_S a full sweep of
# ~950 cameras is well within Roboflow's free tier but is not instant.
SWEEP_MAX_CAMERAS = 250
# Parallel workers for a sweep. Kept modest: these are public civic feeds and
# MIN_POLL_INTERVAL_S is per camera, not global.
SWEEP_CONCURRENCY = 8
