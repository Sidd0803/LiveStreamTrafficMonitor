# NYC Traffic Congestion Index — Scoped Build Plan

**Supersedes** the double-parking build plan (recoverable at `c108e48^`).
See [Why the scope changed](#why-the-scope-changed) for the reasoning, and
[What carried over](#what-carried-over) for the findings that survived intact.

---

## Context

A learning precursor to a hackathon, to get hands-on with **NYC DOT open camera
data, Roboflow, Gemini, and GCP**. A working end-to-end demo is the definition
of done; ambition is secondary to completion.

---

## Why the scope changed

The original problem was double-parking detection. It was abandoned after the
detection half was working, for a structural reason worth stating precisely.

**The test for whether a problem is single-frame solvable: can a human label it
correctly from one still image?**

Double parking fails that test. Shown one frame containing a van in the travel
lane, a human genuinely cannot tell whether it is parked or stopped in traffic.
The information is not in the pixels. Every hard part of that design traced back
to this single root cause — the IoU tracker existed to recover the missing time
axis, the 4-consecutive-detection rule made the confidence gate load-bearing,
and the evaluation could not be scored by frame-level labels because
stationarity is not a property of any frame.

Congestion passes the test. A human reads "backed up" off a single still from
spacing and density alone. That makes frame-level labels *valid*, which
collapses the eval problem, and removes the need for a tracker entirely.

**Consequence:** the ~950-camera network becomes addressable instead of a
hand-tuned handful, because nothing needs per-camera calibration.

---

## Scope

### In scope

- Poll NYC DOT cameras — a curated set for close work, and a **network-wide
  sweep** across hundreds of cameras for the map.
- Roboflow hosted inference (`yolov8m-640`) for vehicle boxes.
- **Geometric congestion metrics** computed per frame: vehicle count,
  occupancy, and a scale-free **crowding** ratio.
- **Gemini classifies flow state** from the frame, judging what counting alone
  cannot, and returns a structured verdict with a reason.
- Streamlit dashboard: live congestion map of NYC, per-camera detail with the
  annotated frame and Gemini's reasoning, and eval metrics.
- **~150 hand-labeled frames** (vehicle count + flow class); report count MAE
  and a 3×3 flow confusion matrix.
- GCP: Cloud Run, frames in Cloud Storage, observations in Firestore.

### Explicitly out of scope

- **Anything requiring the time axis** — tracking, speed estimation, stopped-vs-
  moving, dwell time, incident duration. This is the whole point of the pivot;
  re-introducing it re-introduces the problem that was just removed.
- Custom Roboflow annotation and training (deferred — see Stretch).
- Video stream decoding. Snapshot polling only.
- License plates, vehicle re-identification, any person-level analysis.
- Authentication, multi-user support.

### Definition of the output

For one camera at one instant, an **Observation**: a vehicle count, a set of
geometric metrics, a flow state in `{clear, moderate, jammed}`, and the
reasoning behind it. Time series are built by *aggregating* independent
observations — never by comparing frames to each other.

---

## The core metric

Raw vehicle count is not comparable across cameras. Fifteen vehicles on a wide
six-lane arterial is free-flowing; fifteen on a narrow one-way is gridlock.
Field of view, road width, and camera angle all differ, and there are ~950 of
them, so per-camera calibration is not an option.

**Crowding** solves this without calibration:

```
for each vehicle i:
    r_i = (distance to nearest neighbour's centre) / (vehicle i's own box diagonal)

crowding = median(r_i)
```

Both terms shrink together with distance from the camera, so **the ratio is
scale-free and perspective-robust** — a jam in the far field and a jam in the
near field produce the same number, even though their pixel measurements differ
by an order of magnitude. Dividing by *each vehicle's own* diagonal rather than
a global average is what buys the perspective robustness; a global normalizer
would be dominated by whichever depth had more vehicles.

Interpretation is physical: `crowding ≈ 1.0` means vehicles are roughly one
car-length apart, i.e. bumper to bumper. Larger means more space per vehicle.

Supporting metrics: **count** (the base signal) and **occupancy** (summed box
area over frame area — a crude density that captures "how much of the view is
vehicle").

> Initial thresholds (`CROWDING_JAMMED_MAX = 1.6`, `CROWDING_MODERATE_MAX = 3.0`)
> are a **guess**, not a measurement. They are expected to move once the labeled
> set exists. Do not cite them as results.

---

## Architecture

```
NYC DOT /api/cameras ──> poller ──> frame (JPEG bytes)
                                          │
                            ┌─────────────┴─────────────┐
                            v                           v
              Roboflow yolov8m-640            Gemini 2.5 Flash
              (vehicle boxes)                 (flow state + reason)
                            │                           │
                            v                           │
              density.py: count, occupancy,             │
              crowding -> geometric flow                │
                            │                           │
                            └─────────────┬─────────────┘
                                          v
                              fuse -> Observation
                                          │
                          ┌───────────────┴───────────────┐
                          v                               v
                  GCS (frames)                   Firestore (observations)
                          └───────────────┬───────────────┘
                                          v
                                Streamlit dashboard
```

**The two judgments run in parallel and are both recorded.** This is
deliberate: keeping the geometric and Gemini verdicts side by side in every
observation makes the ablation free at eval time — you can score geometry
alone, Gemini alone, and the fusion, from the same stored data. That comparison
is the most interesting result the project can produce, so the storage schema
is designed to make it impossible to lose.

**Fusion rule:** use Gemini's verdict when its confidence ≥
`GEMINI_MIN_CONFIDENCE`, else fall back to geometric. Both are always stored.

---

## Repo layout

```
double_parking/
  app.py                       # Streamlit entrypoint
  src/
    config.py                  # all thresholds (do not scatter literals)
    sources/nycdot.py          # camera list + snapshot fetch, rate-limited
    detect/
      boxes.py                 # Box geometry, IoU
      roboflow_client.py       # hosted inference wrapper
    analyze/
      density.py               # geometric congestion metrics
      gemini_scene.py          # structured flow-state classification
    storage/
      base.py                  # Observation + Store protocol
      local.py                 # JSONL backend (Phase 1)
      gcp.py                   # GCS + Firestore backend (Phase 3)
    pipeline.py                # one camera -> Observation; sweep -> many
  scripts/
    smoke_roboflow.py          # detection sanity check on live frames
    collect.py                 # sample frames + detections -> eval corpus
  eval/
    frames/ labels.json run_eval.py
```

Storage sits behind **one interface with two implementations**, so Phase 1 runs
fully local and Phase 3 is a config flip rather than a rewrite.

---

## Module contracts

These are fixed so the modules can be built independently and in parallel.
**Do not change a signature without updating this section first.**

### `src/analyze/density.py`

```python
@dataclass(frozen=True)
class FrameMetrics:
    vehicle_count: int
    occupancy: float            # sum(box area) / frame area, 0..1
    crowding: float | None      # None when count < MIN_VEHICLES_FOR_CROWDING
    flow_state: str             # "clear" | "moderate" | "jammed"
    by_class: dict[str, int]    # {"car": 12, "truck": 2}

def frame_metrics(boxes: list[Box], frame_w: int, frame_h: int) -> FrameMetrics
def crowding_ratio(boxes: list[Box]) -> float | None
def classify(crowding: float | None, vehicle_count: int) -> str
```

Pure functions, no network, no I/O.

### `src/analyze/gemini_scene.py`

```python
@dataclass(frozen=True)
class SceneVerdict:
    flow_state: str        # "clear" | "moderate" | "jammed"
    confidence: float      # 0..1
    reason: str            # one sentence, human-readable
    notable: str           # "" or e.g. "construction blocking right lane"
    raw: dict

class GeminiUnavailable(RuntimeError): ...

class SceneClassifier:
    def __init__(self, api_key: str | None = None, model: str | None = None)
    def classify(self, image_bytes: bytes) -> SceneVerdict
```

Structured output via `response_mime_type="application/json"` plus a schema —
**not** prompt-and-parse. Frame is upscaled to `SCENE_MIN_DIM_PX` first.

### `src/storage/base.py`

```python
@dataclass(frozen=True)
class Observation:
    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    area: str                      # borough
    captured_at: datetime          # timezone-aware UTC
    vehicle_count: int
    occupancy: float
    crowding: float | None
    by_class: dict[str, int]
    geometric_flow: str            # density.py's verdict
    gemini_flow: str | None        # None when not called or unavailable
    gemini_confidence: float | None
    gemini_reason: str | None
    gemini_notable: str | None
    final_flow: str                # after the fusion rule
    frame_path: str | None

    def to_dict(self) -> dict      # JSON-safe
    @classmethod
    def from_dict(cls, d: dict) -> "Observation"

class Store(Protocol):
    def save(self, obs: Observation) -> None
    def save_many(self, observations: list[Observation]) -> None
    def recent(self, camera_id: str | None = None, limit: int = 500) -> list[Observation]
    def latest_per_camera(self) -> list[Observation]
```

`latest_per_camera()` is what the map renders, so it must be cheap.

### `src/pipeline.py`

```python
def observe(camera, detector, classifier, store, save_frame=True) -> Observation | None
def sweep(cameras, detector, classifier, store, concurrency=...) -> list[Observation]
```

Returns `None` when the camera is offline — never raises for that. Gemini
failure degrades to geometric-only rather than losing the observation.

---

## Implementation phases

### Phase 1 — Local end-to-end (the demo floor)

1. `analyze/density.py` — metrics and geometric classification. Pure, testable.
2. `analyze/gemini_scene.py` — structured flow-state verdicts.
3. `storage/base.py` + `storage/local.py` — Observation and a JSONL backend.
4. `pipeline.py` — wire detection + metrics + Gemini into an Observation.
5. `app.py` — map, per-camera detail, annotated frame, Gemini's reasoning.

**Demoable:** a live congestion map of NYC with per-camera reasoning.

### Phase 2 — Evaluation

Collect ~150 frames via `scripts/collect.py`, spread across cameras, times of
day, and boroughs. Hand-label each with a **vehicle count** and a **flow class**.

Report:

| Metric | What it tells you |
|---|---|
| Count MAE, and count bias | Whether `MIN_BOX_CONFIDENCE` is systematically under-counting |
| Flow accuracy — geometry only | Whether crowding alone works |
| Flow accuracy — Gemini only | Whether the VLM alone works |
| Flow accuracy — fused | Whether the combination beats either |
| 3×3 confusion matrix | *Which* confusions happen; clear↔moderate is forgivable, clear↔jammed is not |

**The three-way comparison is the ablation, and it is the headline result.** It
answers "is Gemini earning its place, and is the geometry?" — which is the
question this stack exists to teach.

Tune `MIN_BOX_CONFIDENCE` and the crowding thresholds against the **tuning
split only**; hold out a third of the set and score it once, at the end.

With ~150 frames the 95% interval on an accuracy figure is roughly ±8 points.
Report the interval alongside the number.

### Phase 3 — GCP

`storage/gcp.py` against the same protocol: frames to GCS, observations to
Firestore. Containerize, deploy to Cloud Run. **Demoable:** a public URL.

### Stretch

- Fine-tune a detector on labeled NYC DOT frames and compare against
  `yolov8m-640` on the eval set. Highest-value learning left.
- Time-of-day aggregation: hourly congestion profiles per camera. Legitimate
  here because it aggregates independent observations rather than comparing
  frames.
- 511NY highway cameras as an additional source once the key arrives —
  congestion applies to highways, unlike the previous scope.

---

## What carried over

Findings from the double-parking work that remain valid. Full detail in
[PROGRESS.md](PROGRESS.md).

- **511NY covers only highways inside NYC**; NYC DOT runs the ~950-camera local
  street network, public and keyless. Still the right primary source — though
  511NY is now *usable* rather than useless, since highways do congest.
- **Camera curation must be visual.** Name matching returns skyline-framed views
  and hazed lenses. The bridge-deck and highway-ramp rejections no longer apply.
- **Every frame is 352×240.** Resolution remains the binding constraint and the
  strongest argument for the fine-tuning stretch goal.
- **Polling at 12s is safe and verified** — 18/18 byte-distinct frames at 5s.
- **`yolov8m-640` beat five alternatives** on live frames: 25 confident
  detections vs 13, median confidence 0.25 vs 0.19, ~4× faster.
- **Vehicle class matching must be per-word.** Exact matching silently
  discarded `semi-trailer`, `pickup-truck` and `vehicles` — 9 real trucks lost
  with no error. This bug class is the reason count bias is an explicit eval
  metric.
- **Columbus Ave @ 65 St returned 0 detections in rain**, on a frame with an
  unmissable white truck, even at confidence 0.05. Wet-asphalt glare defeats
  detection outright. Keep as a known hard case.

---

## Key technical decisions

- **Crowding normalized per-vehicle, not globally** — the only way the metric
  survives perspective, and the reason no per-camera calibration is needed.
- **Both verdicts always stored** — makes the ablation free and unlosable.
- **Structured Gemini output via schema**, not prompt-and-parse.
- **Detection cached at low confidence** during collection, so threshold tuning
  is offline and instant rather than re-polling a public civic feed.
- **All thresholds in `config.py`** — Phase 2 tunes them repeatedly.
- **Storage behind one protocol** — Phase 3 is a config flip.

---

## Risks

| Risk | Mitigation |
|---|---|
| Crowding thresholds are guessed, not measured | Explicitly labeled as such; Phase 2 sets them from data |
| Count is systematically low at 352×240, biasing everything toward "clear" | Count MAE **and bias** are both reported; `MIN_BOX_CONFIDENCE` is the knob |
| Crowding is undefined for sparse frames | `MIN_VEHICLES_FOR_CROWDING` floor; report "clear" without a ratio |
| Rain/glare defeats detection entirely | Known and documented; Gemini may cover it, which the ablation will show |
| A wide empty arterial reads as congested, or a narrow busy one as clear | Precisely what the scale-free ratio is designed for; the confusion matrix tests it |
| Network sweep is slow or hits rate limits | `SWEEP_MAX_CAMERAS`, `SWEEP_CONCURRENCY`, per-camera throttle |
| Labeling 150 frames is tedious enough to not happen | Collection is automated; only count + one class per frame |

---

## Verification

1. **Unit tests** on `density.py` with synthetic layouts — a packed grid must
   score low crowding, a sparse scatter high, and the same layout at two
   different scales must score *the same*.
2. **Smoke tests** against live frames for Roboflow and Gemini.
3. **`eval/run_eval.py`** — the quantitative gate. Run after every threshold change.
4. **End-to-end local:** `streamlit run app.py`, sweep, confirm the map renders
   and reasoning is coherent.
5. **Post-deploy:** hit the Cloud Run URL, confirm Firestore and GCS writes.

---

## Cost

Roboflow free tier covers this volume. Gemini 2.5 Flash on frames is cents/day.
Cloud Run scales to zero. **Under $5 total.** Set a budget alert anyway.

A full 250-camera sweep is 250 Roboflow calls plus 250 Gemini calls. Do not put
that on a tight schedule without checking quota first.
