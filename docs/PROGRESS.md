# Progress Log

Running record of what is built, what was learned, and what to do next.
**Read this first when picking the project back up.**

Last updated: **2026-08-04**

---

## Status at a glance

| Component | Status | Commit |
|---|---|---|
| Repo scaffold, config, dependencies | ✅ Done | `7021e02` |
| NYC DOT camera source | ✅ Done, verified live | `e6889a5` |
| Box geometry + IoU | ✅ Done | `baa6205` |
| IoU tracker (stationarity) | ✅ Done, 19 tests pass | `baa6205` |
| Lane/curb heuristic | ✅ Done, 19 tests pass | `baa6205` |
| Docs + GitHub remote | ✅ Done | `2030aac` |
| Roboflow client | 🟡 **Written, 17 tests pass — unverified against live API** | `64ee564` |
| Gemini adjudicator | ⛔ **Blocked — needs `GEMINI_API_KEY`** | — |
| Pipeline orchestration | ⬜ Not started | — |
| Streamlit dashboard | ⬜ Not started | — |
| Eval set + metrics (Phase 2) | ⬜ Not started — **design open, see below** | — |
| GCP deployment (Phase 3) | ⬜ Not started | — |

**Nothing is blocked on the 511NY approval queue** — that was a deliberate
sequencing decision. NYC DOT needs no key.

---

## Pick up here

### 0. Get running on the new machine (~2 min)

```bash
git clone https://github.com/Sidd0803/LiveStreamTrafficMonitor.git
cd LiveStreamTrafficMonitor
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
cp .env.example .env
```

Confirm the environment is sound before touching anything — this needs no keys
and no network:

```bash
python -m pytest tests/ -q
```

Expect **36 passed**. Then confirm live camera access, also keyless:

```bash
python -c "from src.sources import nycdot; c=nycdot.demo_cameras(); print(len(c), c[0].name, len(nycdot.fetch_snapshot(c[0])))"
```

> **Gotcha:** imports are repo-rooted, so `PYTHONPATH` must include the repo
> root. `pytest` picks it up automatically from the working directory; ad-hoc
> `python -c` may not. If you hit `ModuleNotFoundError: No module named 'src'`,
> set `PYTHONPATH=$PWD` (PowerShell: `$env:PYTHONPATH = $PWD`).

### 1. Verify Roboflow against a live frame ← **next action**

Fill in `.env`:

```
ROBOFLOW_API_KEY=          # app.roboflow.com -> Settings -> API Keys
ROBOFLOW_WORKSPACE=        # from the workflow URL: app.roboflow.com/<workspace>/workflows/<id>
ROBOFLOW_WORKFLOW_ID=      #                                                      ^^^^
```

Then run the detector on a real Amsterdam Ave frame. `roboflow_client.py` is
written and unit-tested but **has never made a real API call** — this is the
first thing to prove.

What to check: does `vehicle-detection-bz0yu/4` actually resolve vehicles at
352x240? Count boxes against what you can see in the frame. This is the single
biggest unknown in the project (see risks below).

### 2. Build `src/detect/gemini_judge.py`

Crop with `Box.padded(config.CROP_PADDING_PX, ...)`, upscale to
`config.CROP_MIN_DIM_PX`, send with a Pydantic `responseSchema`
(`is_double_parked`, `confidence`, `reason`, `vehicle_type`). Use structured
output, not free-text parsing — `response_mime_type="application/json"` plus a
schema. Free-text parsing will waste hours.

### 3. Wire `src/pipeline.py` + `app.py`

Then the eval set — but resolve the open eval-design question first.

---

## Decisions waiting on you

Two things are genuinely open and should be settled before the work that
depends on them.

### Eval design — blocks Phase 2

Frame-level labels cannot score a tracker. Full detail in the section below and
in [BUILD_PLAN.md](BUILD_PLAN.md) amendment A4. **You said you wanted to review
the eval strategy** — this is that conversation.

### Is `vehicle-detection-bz0yu/4` your own trained model?

Roboflow described it as "your existing Vehicle Detection model." Unanswered:
did you train it, and if so on what imagery? A model trained on traffic-cam-like
frames is far more trustworthy at 352x240 than one trained on high-res photos,
and the answer changes whether Phase 4 fine-tuning is a stretch goal or a
necessity.

---

## Findings from the work so far

These are the things that would cost time to rediscover.

### 1. 511NY cannot see double parking — source changed to NYC DOT

The original spec named 511NY as the camera source. 511NY's NYC coverage is
**limited-access highways** (Cross Bronx, LIE, BQE, Major Deegan, Staten Island
Expwy). Double parking is a local-street phenomenon; it does not happen there.

NYC DOT runs a **separate ~950-camera system on local streets**, public and
keyless:

- `GET https://webcams.nyctmc.org/api/cameras` → JSON catalog
- `GET https://webcams.nyctmc.org/api/cameras/{id}/image` → JPEG snapshot

511NY is retained as the statewide catalog and event feed (Phase 3, additive).

### 2. Camera curation must be visual, not by name

Matching street names against the catalog returned, among others:

| Camera | Why it's useless |
|---|---|
| WBB-20 @ Above Bedford Ave | Williamsburg Bridge deck — limited access |
| Canal Street @ Chrystie Street | Elevated bridge approach ramp, not the surface street |
| Queens Plaza N @ Northern Blvd | Framed on skyline; roadway is a sliver of frame |
| Queens Blvd @ 39 St - East | Hazed/dirty lens, washed out |
| Broadway @ 45 St | Times Square — heavy pedestrian occlusion, atypical traffic |

12 live frames were pulled and inspected as a contact sheet; 6 usable cameras
were kept. Selections **and rejections with reasons** are in
[`data/demo_cameras.json`](../data/demo_cameras.json).

**Amsterdam Ave @ 60 St** is the best camera — delivery vans double-parked in
the travel lane were visible in the very first frame sampled.

Curated IDs are cross-checked against the live catalog on load, because a stale
hardcoded ID would otherwise be a silent demo failure.

> Re-verify the list before any demo. Cameras get repositioned and go offline.

### 3. Snapshot refresh is ~1-5s — polling at 12s is safe

Worth checking because the burned-in timestamps looked suspiciously identical
across cameras (they weren't — all 12 were simply fetched in the same second).

Measured: **18/18 byte-distinct frames** across three cameras polled every 5s
for 90s. So frames are genuinely fresh and there is **no false-stationarity
risk from stale images**, which would have silently corrupted the tracker.

Caveat: byte-distinctness alone is weak evidence, since the burned-in timestamp
banner changes every second regardless of scene. The banner advancing is what
actually confirms fresh capture.

### 4. Resolution is the binding constraint

**Every NYC DOT snapshot is 352x240.** A mid-frame vehicle is roughly 40x30px.
This drives several design choices:

- Gemini gets **padded, upscaled crops**, never full frames.
- There is a minimum box-area floor (`MIN_BOX_AREA_FRACTION`) to drop specks.
- It is the most likely cause of poor detection quality, and the strongest
  argument for the Phase 4 stretch goal of fine-tuning on real traffic-cam
  frames rather than using a Universe model trained on high-res imagery.

### 5. Roboflow Workflows cannot do our temporal logic

Roboflow proposed a workflow that would also handle zone flagging, incident
detection and Vision Events logging. Declined, for a concrete reason: Roboflow's
stateful blocks (**ByteTrack**, **Time in Zone**) require continuous video. Per
Roboflow's own docs, on a still image "there is no meaningful history to track,
compare, aggregate, or visualize."

Our input is independent snapshots ~12s apart, so those blocks don't apply, and
single-frame zone occupancy would flag every vehicle in the lane — including
moving traffic and red-light queues. That is exactly the single-frame heuristic
rejected during scoping.

Settled boundary: **Roboflow detects, Python decides, Gemini adjudicates.** The
hosted workflow is detection-only — one image in, vehicle boxes out. Keeping
incident logic in Python also preserves the congestion guard and keeps Firestore
meaningful as the incident store.

### 6. Two false-positive traps the heuristic handles explicitly

- **Red lights / gridlock.** At a red light *every* vehicle is stationary in a
  travel lane — a naive rule reads that as a frame full of double-parking. If
  ≥70% of tracks are stationary the frame is treated as congested and all
  candidates suppressed. Double parking means stopped *while traffic moves*.
- **Legal curbside parking.** Parked cars are also stationary and roadside. The
  fallback requires lateral offset from the curb band *plus* an adjacent
  stopped vehicle (the defining pattern: stopping alongside a parked car).

---

## Design decisions and why

- **IoU matching over ByteTrack/DeepSORT.** Those carry motion models tuned for
  20-30fps video. Our frames are ~12s apart, so inter-frame velocity is
  meaningless. Plain IoU is more appropriate *and* more instructive.
- **Greedy matching over Hungarian assignment.** At a 0.6 IoU threshold with
  well-separated vehicles they agree almost always, and greedy is far easier to
  debug mid-demo.
- **Pillow over OpenCV.** Lighter, no wheel issues on Python 3.14, and crop +
  draw is all that's needed.
- **Every threshold in `src/config.py`.** Phase 2 involves tuning these
  repeatedly; scattered literals would make that painful.
- **Storage behind one interface.** Phase 1 runs fully local; Phase 3 is a
  config flip, not a rewrite.
- **Missed detections tolerated for 2 polls**, but `stationary_polls` does not
  increment through the gap — a flickering detector shouldn't manufacture dwell
  time.

---

## Open questions and risks

### Eval design is unresolved — decide before building Phase 2

The build plan says "label 50-100 frames," but **frame-level labels cannot
score a tracker.** Stationarity is a property of a vehicle across ~4 consecutive
polls, not of any single image. A frame-labeled set can evaluate the *detector*
and *Gemini's judgment on a crop*, but not the temporal logic that is the core
of the approach.

Likely correction: label **poll sequences per camera** (e.g. 20 sequences of 6
consecutive frames) rather than independent frames. Costs more to collect but
is the only thing that measures the actual system. **Flagged for review; not
yet decided.**

Also worth deciding: report precision/recall per *incident* or per *frame*? An
incident lasting 10 polls shouldn't count as 10 successes.

### Other open risks

| Risk | Mitigation |
|---|---|
| **Model may not resolve vehicles at 352x240** — still the top unknown | Test `vehicle-detection-bz0yu/4` against live frames before building on it; fine-tuning is the fallback |
| Workflow output key may not be `vehicle_boxes` | Client searches the response for prediction geometry rather than hardcoding a path, and falls back to direct model inference |
| Nothing double-parks during a live demo | Build a **replay mode** over cached frames (near-free alongside the eval set) |
| Camera offline mid-demo | `demo_cameras()` cross-checks the live catalog and falls back |
| Heuristic false-positive rate unknown | This is what the eval set is for; Gemini is the second gate |

---

## Verification performed

- ✅ Live camera catalog fetch — 969 online cameras across 5 boroughs
- ✅ Live snapshot fetch — valid JPEG, 352x240, ~9-22KB
- ✅ Refresh-rate probe — 18/18 distinct frames at 5s cadence
- ✅ Curated camera loading with live cross-check — 6/6 resolved
- ✅ `pytest tests/ -q` — **36 passed**
- ⛔ Roboflow live inference — **not yet run**, no key

Tests deliberately cover both directions: the negative heuristic cases would
all pass trivially if the heuristic simply rejected everything, so there is an
explicit positive test for the genuine double-parking pattern.

---

## Environment notes

Things about the dev setup that are easy to trip over.

- **Python 3.14** on the original machine. Nothing depends on 3.14 specifically;
  3.11+ should be fine.
- **`PYTHONPATH` must include the repo root.** Imports are rooted at `src.*`.
- **Pillow, not OpenCV.** Chosen deliberately — lighter, no wheel issues on
  3.14, and crop + draw is all that's needed. Don't add `cv2` without a reason.
- **`.env` is gitignored** and does not exist in the repo. Copy `.env.example`.
- **The repo is public.** No secrets in tracked files (verified by scan). Keep
  it that way — everything credential-shaped reads from env via `config.py`.
- **NYC DOT needs no key**, so camera ingestion and all 36 tests run
  immediately after clone. Only detection and adjudication need credentials.

---

## Commit history

| Commit | What |
|---|---|
| `7021e02` | Scaffold: layout, dependencies, central config |
| `e6889a5` | NYC DOT camera source + hand-curated camera list |
| `baa6205` | Box geometry, IoU tracker, lane heuristic, 19 tests |
| `2030aac` | README, build plan, progress log |
| `64ee564` | Roboflow client (workflow + direct model paths), 17 more tests |

---

## Session log

**2026-08-04** — Project scoped via interview and greenlit. Built camera
ingestion, tracking, heuristic, and the Roboflow client. Key course-correction:
the camera source moved from 511NY to NYC DOT once it became clear 511NY only
covers highways inside the city. Roboflow proposed owning zone logic and
incident detection; declined after confirming its stateful blocks need
continuous video. Ended blocked on API keys, with the eval design flagged as
unresolved.
