# Double-Parking Detector — Scoped Build Plan

> **Approved 2026-08-04.** This is the plan as agreed, kept intact as a record
> of intent. Where implementation has since diverged, the **Amendments** section
> at the bottom is authoritative. For current status see
> [PROGRESS.md](PROGRESS.md).

## Context

This is a **learning precursor to a hackathon**, not a production system. The goal is hands-on familiarity with 511NY / NYC DOT open camera data, Roboflow hosted inference, Gemini vision with structured output, and GCP (Cloud Run + GCS + Firestore). A **working end-to-end demo is the definition of done** — ambition is explicitly secondary to completion.

The project directory is empty, so this is greenfield. No accounts exist yet (511NY, GCP, Roboflow, Gemini), which shapes the sequencing below: **nothing on the critical path may block on an approval queue.**

### One finding that changed the design

The original spec named 511NY as the camera source. 511NY's NYC coverage is **limited-access highways** (Cross Bronx, LIE, BQE, Major Deegan, Staten Island Expwy). Double parking is a *local street* phenomenon — it does not occur on highways. Building purely on 511NY would produce a demo with nothing to detect.

**Resolution (user-confirmed):** NYC DOT's camera system is the detection source — ~950 cameras on local streets (Queens Blvd, Atlantic Ave, signalized intersections), served from a public JSON endpoint with no key and no approval wait. 511NY is retained in the stack as the statewide camera catalog and as a corroborating event feed, so it is still learned, just not load-bearing.

---

## Scope

### In scope
- Poll a **fixed set of 4–6 NYC DOT cameras** on local streets with known double-parking pressure.
- Roboflow hosted inference (pretrained Universe vehicle model) for vehicle boxes.
- **IoU-based stationarity tracking** across polls — a vehicle holding near-identical position for N consecutive polls while traffic moves around it.
- Geometric heuristic to flag *candidate* double-parkers (stationary + in a travel lane, not at the curb).
- **Gemini adjudicates** each candidate crop and returns a structured verdict + reason.
- Streamlit dashboard: camera picker, annotated live frame, incident feed with Gemini's reasoning, map of flagged locations, and live precision/recall from the eval set.
- **~50–100 hand-labeled frames** as an evaluation set; report precision and recall.
- GCP: containerized on Cloud Run, frames/crops in Cloud Storage, incidents in Firestore.

### Explicitly out of scope
- Custom Roboflow annotation and training (deliberately deferred — see Stretch).
- Video stream decoding (MJPEG/HLS). Snapshot polling only.
- License plate reading, vehicle re-identification, enforcement/ticketing integration.
- Pub/Sub, Cloud Scheduler, multi-worker queuing.
- Any authentication, multi-user support, or persistence beyond Firestore.

### Definition of "double parked"
A vehicle that (a) remains in near-identical pixel position across **N consecutive polls** (default N=4, ~60–90s), (b) is positioned **outside the curb/parking zone** per the camera's lane geometry, and (c) is **confirmed by Gemini** from the cropped image. All three must hold. This is the honest definition and gives a temporal signal, which single-frame geometry cannot.

---

## Architecture

```
NYC DOT /api/cameras ──┐
                       ├─> poller ──> frame (JPEG bytes)
511NY event feed ──────┘                  │
                                          v
                         Roboflow hosted inference (vehicle boxes)
                                          │
                                          v
                            IoU tracker (stationarity across polls)
                                          │
                                          v
                            lane heuristic -> candidate crops
                                          │
                                          v
                     Gemini 2.5 Flash (structured verdict + reason)
                                          │
                          ┌───────────────┴───────────────┐
                          v                               v
                  GCS (frames/crops)              Firestore (incidents)
                          └───────────────┬───────────────┘
                                          v
                                Streamlit dashboard
```

**Why this split:** Roboflow does what it is good at (fast, cheap, deterministic box detection). The IoU tracker and lane heuristic do the temporal and spatial reasoning in plain Python — cheap and debuggable. Gemini is called **only on candidates**, which keeps token cost near zero and makes it the adjudicator rather than the detector. This is also the boundary that teaches the most: you see exactly where classical CV ends and VLM reasoning begins.

---

## Data sources (verified)

| Source | Endpoint | Auth | Notes |
|---|---|---|---|
| NYC DOT camera list | `https://webcams.nyctmc.org/api/cameras` | none | JSON array; fields `id` (UUID), `name`, `latitude`, `longitude`, `area` (borough), `isOnline`, `imageUrl` |
| NYC DOT snapshot | `https://webcams.nyctmc.org/api/cameras/<id>/image` | none | Direct JPEG. Low-res, refreshes every ~1–2s |
| 511NY cameras | `https://511ny.org/api/getcameras?key=<KEY>&format=json` | dev key (approval form) | Fields incl. `Id`, `Roadway`, `Direction`, `Latitude`, `Longitude`, `Views[].Url`, `Views[].VideoUrl`. **Throttled: 10 calls / 60s** |
| 511NY events | 511NY data feed | dev key | Used as corroborating context only |

**Sequencing consequence:** submit the 511NY developer access request on day one, then build everything against NYC DOT while it is pending. The 511NY integration is additive and lands whenever the key arrives.

**Responsible use:** poll no faster than every 10–15s per camera, set a descriptive User-Agent, cache aggressively, and never republish raw camera imagery. Rate limiting is a hard requirement in the poller, not a nicety.

---

## Repo layout

```
double_parking/
  README.md
  requirements.txt
  .env.example                 # never commit real keys
  Dockerfile
  app.py                       # Streamlit entrypoint
  src/
    config.py                  # camera IDs, thresholds, env loading
    sources/
      nycdot.py                # camera list + snapshot fetch, rate-limited
      ny511.py                 # 511NY catalog + event feed (additive)
    detect/
      roboflow_client.py       # hosted inference wrapper
      tracker.py               # IoU matching + stationarity state
      heuristic.py             # lane/curb geometry -> candidates
      gemini_judge.py          # structured adjudication
    storage/
      local.py                 # filesystem backend (Phase 1)
      gcp.py                   # GCS + Firestore backend (Phase 3)
    pipeline.py                # orchestrates one poll cycle
  eval/
    frames/                    # ~50-100 saved frames
    labels.json                # hand labels
    run_eval.py                # precision/recall report
  data/
    zones/<camera_id>.json     # optional lane polygons
```

Storage is behind a **single interface** with a local and a GCP implementation. This is the key structural decision: it lets Phase 1 run entirely offline and makes the GCP migration a config flip rather than a rewrite.

---

## Implementation phases

Each phase ends at a **demoable state**. If time runs out at any phase boundary, you still have something to show.

### Phase 0 — Accounts and skeleton
Submit the 511NY developer access request (approval lag). Create Roboflow account (free tier, get API key), Google AI Studio key for Gemini, and a GCP project with billing enabled. Scaffold the repo, `requirements.txt`, `.env.example`.

Verify each key with a one-line smoke test before writing any pipeline code. Do not proceed until all three respond.

### Phase 1 — Local end-to-end (the demo floor)
1. `sources/nycdot.py` — fetch camera list, filter to `isOnline == "true"` and hand-pick 4–6 local-street cameras by `name`/`area`. Snapshot fetch with per-camera rate limiting and retry.
2. `detect/roboflow_client.py` — wrap `inference-sdk`'s `InferenceHTTPClient` against a Roboflow Universe vehicle-detection model. Return normalized `[{x, y, w, h, class, confidence}]` regardless of model quirks.
3. `detect/tracker.py` — per-camera state dict. On each poll, match new boxes to previous boxes by IoU (threshold ~0.6). Increment a `stationary_polls` counter on match, reset on miss. Emit tracks exceeding N polls. **~80 lines, no tracker library.**
4. `detect/heuristic.py` — flag a stationary track as a candidate if it sits outside the curb zone. Start with a simple rule (lateral offset from the modal parked-vehicle row); optionally refine per-camera with hand-drawn polygons in `data/zones/`.
5. `detect/gemini_judge.py` — send the crop (plus a little surrounding context) to `gemini-2.5-flash` with `response_mime_type="application/json"` and a Pydantic `responseSchema`: `{is_double_parked: bool, confidence: float, reason: str, vehicle_type: str}`. Structured output is non-negotiable here — free-text parsing will waste hours.
6. `pipeline.py` + `app.py` — wire it together behind local filesystem storage. Streamlit shows the annotated frame and an incident feed.

**Demoable:** live NYC camera, boxes drawn, incidents appearing with Gemini's reasoning.

### Phase 2 — Evaluation
Save ~50–100 frames spanning the chosen cameras and a range of times of day, including hard negatives (buses at stops, standing traffic, delivery trucks legitimately at the curb). Hand-label in `eval/labels.json`. `run_eval.py` replays them through the pipeline and reports precision, recall, and a confusion matrix. Surface the numbers in the dashboard.

Use this to tune the IoU threshold, N, and the Gemini prompt. **This phase is what makes the project defensible** — "here are our numbers" is the difference between a project and a toy.

### Phase 3 — GCP
Implement `storage/gcp.py` against the same interface: frames and crops to a GCS bucket, incident records to Firestore. Containerize and deploy to Cloud Run. Point Streamlit at the deployed service (or deploy Streamlit itself to Cloud Run — simplest single-service option).

Optionally switch Gemini from the AI Studio key to Vertex AI to keep everything inside the GCP project.

**Demoable:** a public URL.

### Stretch (only if Phases 1–3 are solid)
- Annotate ~150 NYC DOT frames in Roboflow, train a v2 tuned to low-res traffic-cam imagery, compare against the Universe model on the eval set. **This is the highest-value learning left**, and deferring it means the demo is never blocked on annotation.
- 511NY event-feed corroboration once the key arrives.
- Cloud Scheduler for continuous background polling.

---

## Key technical decisions

- **Gemini sees crops, not full frames.** Cheaper, faster, and materially more accurate — a full 352x240 traffic-cam frame gives the model too little signal per vehicle.
- **Gemini is called only on candidates.** With 4–6 cameras and a sane heuristic this is a handful of calls per minute, keeping cost negligible.
- **Structured output via Pydantic schema**, not prompt-and-parse.
- **IoU matching over ByteTrack.** ByteTrack assumes continuous video; on 10–15s-spaced snapshots its motion model is meaningless. Plain IoU is both more appropriate and more instructive here.
- **Thresholds live in `config.py`**, not scattered as literals — you will tune them repeatedly in Phase 2.

## Cost

Roboflow free tier covers hosted inference at this volume. Gemini 2.5 Flash on crops is cents/day. GCP: Cloud Run scales to zero, GCS and Firestore are within free tier at this scale. **Realistically under $5 total.** Set a GCP budget alert anyway.

## Risks

| Risk | Mitigation |
|---|---|
| 511NY key approval is slow | Not on the critical path by design; NYC DOT needs no key |
| Camera imagery too low-res for reliable detection | Pick cameras by visual inspection first; prefer well-framed street views |
| No double-parking occurs during the demo | Cache a set of known-good frames as a replay mode — build this in Phase 2 alongside the eval set, it is nearly free |
| Camera goes offline mid-demo | Check `isOnline`, fall back to the next camera, and keep the replay mode |
| Heuristic produces excessive false positives | This is what the eval set is for; Gemini is the second gate |

## Verification

1. **Per-module smoke tests** as you go: fetch a camera list and assert non-empty; fetch one snapshot and assert valid JPEG bytes; run one image through Roboflow and assert boxes; send one crop to Gemini and assert schema-valid JSON.
2. **Tracker unit test** with synthetic boxes: a fixed box across 5 polls must flag stationary; a box translating each poll must not.
3. **`eval/run_eval.py`** — the primary quantitative gate. Precision and recall on the labeled set, run after every threshold change.
4. **End-to-end local:** `streamlit run app.py`, watch a real camera for ~10 minutes, confirm incidents appear with coherent Gemini reasoning and no crashes on offline cameras.
5. **Post-deploy:** hit the Cloud Run URL, confirm incidents land in Firestore and frames in GCS, and confirm cold start doesn't break the UI.

---

## Amendments

Changes made after the plan was approved, as implementation contact with the
real data forced them. Each supersedes the corresponding section above.

### A1 — Camera selection is hand-curated, not name-matched (2026-08-04)

The plan said "hand-pick 4–6 local-street cameras by `name`/`area`." Name
matching proved actively misleading: it returned bridge decks, elevated highway
ramps, a skyline-facing camera and one with a hazed lens.

Selection is now a curated file, [`data/demo_cameras.json`](../data/demo_cameras.json),
built by fetching live frames and inspecting them, recording rejections and
reasons alongside selections. Curated IDs are cross-checked against the live
catalog on load, since a stale hardcoded ID is otherwise a silent demo failure.
Name matching survives only as a degraded fallback.

### A2 — Polling interval validated empirically (2026-08-04)

The plan assumed snapshots refresh every ~1–2s without checking. Measured
directly: **18/18 byte-distinct frames** across three cameras polled every 5s
for 90s. Frames are genuinely fresh, so the 12s poll interval carries no
false-stationarity risk — which would have silently corrupted the tracker had
it been wrong.

Frames are confirmed **352x240**, which makes the padded-and-upscaled crop
strategy for Gemini load-bearing rather than an optimization.

### A3 — Congestion guard added to the heuristic (2026-08-04)

Not anticipated in the plan. At a red light *every* vehicle is stationary in a
travel lane, which the plan's rule would read as a frame full of double-parking.
The heuristic now suppresses all candidates when ≥70% of tracks are stationary
(`CONGESTION_STATIONARY_FRACTION`). This follows from the definition itself:
double parking is stopping *while traffic moves around you*.

### A4 — Eval design is unresolved and blocks Phase 2 (open)

The plan specifies "~50–100 hand-labeled frames." **Frame-level labels cannot
score a tracker** — stationarity is a property of a vehicle across ~4
consecutive polls, not of any single image. Such a set can evaluate the
detector and Gemini's judgment on a crop, but not the temporal logic that is
the core of the approach.

Probable correction: label **poll sequences** (e.g. 20 sequences × 6 consecutive
frames per camera) instead of independent frames. Also unresolved: whether to
score precision/recall per *incident* or per *frame* — an incident spanning 10
polls should not count as 10 successes.

**Decide this before building `eval/run_eval.py`.** Flagged for review.
