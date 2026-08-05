# Progress Log

Running record of what is built, what was learned, and what to do next.
**Read this first when picking the project back up.**

Last updated: **2026-08-05**

---

## Status at a glance

| Component | Status | Commit |
|---|---|---|
| Repo scaffold, config, dependencies | ✅ Done | `7021e02` |
| NYC DOT camera source | ✅ Done, verified live | `e6889a5` |
| Box geometry + IoU | ✅ Done | `baa6205` |
| IoU tracker (stationarity) | ✅ Done, unit tests pass | `baa6205` |
| Lane/curb heuristic | ✅ Done, unit tests pass | `baa6205` |
| Docs + GitHub remote | ✅ Done | `2030aac` |
| Roboflow client | ✅ **Verified against the live API** | `64ee564` + 2026-08-05 |
| Detection model | ✅ **Swapped to COCO `yolov8m-640`** | 2026-08-05 |
| Vehicle class filter | ✅ **Bug fixed — was dropping delivery trucks** | 2026-08-05 |
| Replay harness (`scripts/replay.py`) | 🟡 **Built and runs; only smoke-tested at 3 polls** | 2026-08-05 |
| Gemini adjudicator | ⬜ Not started — key is now in `.env` | — |
| Pipeline orchestration | ⬜ Not started | — |
| Streamlit dashboard | ⬜ Not started | — |
| Eval set + metrics (Phase 2) | ⬜ Not started — **design still open, see below** | — |
| GCP deployment (Phase 3) | ⬜ Not started | — |

**Nothing is blocked on the 511NY approval queue** — that was a deliberate
sequencing decision. NYC DOT needs no key.

> **Still true, and the thing to hold onto:** no double-parking detection
> happens yet. Detection works well; the tracker and heuristic are written but
> nothing in `src/` calls them (only the tests do), so they have never run on a
> real frame sequence. `pipeline.py`, `gemini_judge.py` and `app.py` do not
> exist. `scripts/replay.py` is the bridge — see step 1 below.

---

## Pick up here

### 0. Get running on the new machine (~2 min)

```bash
git clone https://github.com/Sidd0803/LiveStreamTrafficMonitor.git
cd LiveStreamTrafficMonitor
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
```

> **Use Python 3.12, not 3.14.** `inference-sdk` declares `<3.13` and has no
> wheel for 3.14, so `pip install -r requirements.txt` fails outright on a 3.14
> interpreter. An earlier note here said the original machine ran 3.14 — that
> only held because Roboflow had never actually been installed or called there.

Confirm the environment is sound before touching anything — this needs no keys
and no network:

```bash
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q
```

Expect **59 passed**. Then confirm live camera access, also keyless:

```bash
python -c "from src.sources import nycdot; c=nycdot.demo_cameras(); print(len(c), c[0].name, len(nycdot.fetch_snapshot(c[0])))"
```

> **Gotcha:** imports are repo-rooted, so `PYTHONPATH` must include the repo
> root. `pytest` picks it up automatically from the working directory; ad-hoc
> `python -c` may not. If you hit `ModuleNotFoundError: No module named 'src'`,
> set `PYTHONPATH=$PWD` (PowerShell: `$env:PYTHONPATH = $PWD`).

### 1. Run the replay harness for real ← **next action**

`scripts/replay.py` exists and works, but has only been run for **3 polls** —
enough to prove the plumbing, not enough to reach `STATIONARY_POLLS = 4`. So
the tracker has still never actually declared anything stationary on real data.

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/replay.py --polls 12
```

That polls Amsterdam Ave every ~12s for ~2.5 minutes, caches every frame and
its raw detections, then replays them through tracker + heuristic and renders
annotated PNGs to `runs/replay/latest/<camera>/annotated/`.

Re-tuning is offline and instant once collected, because raw detections are
cached at confidence 0.05:

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/replay.py --analyze-only --confidence 0.15
```

**What to look for, in order:**

1. Does any track reach 4 consecutive polls? If not, `MIN_BOX_CONFIDENCE` is
   still starving it — see finding 8.
2. Where does the yellow curb-band line actually land in the annotated frames?
   Finding 9 predicts it lands in the travel lane, which would be backwards.
3. Are candidates emitted, and are they plausible when you look at the crop?

Only after watching those fail or pass is it worth changing thresholds. The
cached sequences are also the corpus the eval design needs (amendment A4).

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

### ~~Is `vehicle-detection-bz0yu/4` your own trained model?~~ — RESOLVED, model dropped

Investigated 2026-08-05 via the Roboflow REST API. Almost certainly not yours,
and no longer used either way. See finding 7.

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

### 7. The suggested Roboflow model was a poor fit — swapped for COCO

Roboflow suggested `vehicle-detection-bz0yu/4` and called it "your existing
Vehicle Detection model," but could not explain its provenance. What the REST
API says:

- It **resolves under your workspace** (`siddharth-saha-rm6rg/...`) and bogus
  project names 404, so the namespacing is real — hence Roboflow's phrasing.
- But your workspace's project list comes back **empty**, and the model's
  training run (2024-01-04 → 2024-01-06) **predates the project's creation**
  (2024-01-29) by three weeks. That pattern says fork of a public Universe
  project, not something you trained.
- Caveat: the control query used a nonexistent *workspace*, which 404s for its
  own reason, so cross-workspace resolution was not cleanly ruled out.
- **203 images total.** Version 4's own record claims `images: 3` with splits
  of 1/1/1, which is incoherent for a real run — more inherited clone metadata.
- Reported mAP 75.36 / recall 69.11 / precision 93.64, measured on its own
  held-out split, not on traffic-cam imagery.
- Preprocessing **stretches input to 640x640**. Our frames are 352x240 (aspect
  1.47), so every vehicle is distorted vertically ~1.8x versus training
  geometry — a plausible direct cause of its low confidence scores.

Benchmarked COCO alternatives on identical live frames (3 cameras):

| model | vehicles | at conf ≥0.40 | median conf | time |
|---|---|---|---|---|
| vehicle-detection-bz0yu/4 | 43 | 13 | 0.19 | 3.5s |
| yolov8s-640 | 83 | 20 | 0.20 | 0.8s |
| **yolov8m-640** ← now default | 71 | **25** | **0.25** | 0.8s |
| yolov11s-640 | 80 | 17 | 0.15 | 0.6s |

`yolov8m-640` nearly doubles confident detections, raises median confidence,
runs ~4x faster, and has a real `truck` class. COCO aliases are hosted by
Roboflow and need no project or workflow — the same API key just works.

**This makes Phase 4 fine-tuning look like a necessity rather than a stretch
goal** if accuracy ever needs to be defensible: even yolov8m is a general model
being asked to read 352x240 civic-camera frames.

### 8. The class filter was silently dropping delivery trucks

The worst bug found so far, and it left no trace. `VEHICLE_CLASSES` was an
exact-match set containing `truck`, `pickup`, `vehicle`. The model emitted
`pickup-truck`, `semi-trailer`, `vehicles`. **None matched.** Measured across
four cameras: 9 `semi-trailer` detections discarded.

On NYC local streets "semi-trailer" is that model's label for box and delivery
trucks — the single most common double-parker, and the exact thing Amsterdam
Ave was chosen for.

Fixed by `is_vehicle_label()` in `detect/roboflow_client.py`: labels are
lowercased, split on `-`/`_`/space, and de-pluralized before lookup against
`config.VEHICLE_WORDS`, with a `NON_VEHICLE_WORDS` veto so "bus stop" and
"truck stop sign" don't slip through. Deliberately permissive — a wrongly-kept
box costs one Gemini call at the adjudication gate, while a wrongly-dropped box
is invisible and silently removes a real double-parker. 23 regression tests.

### 9. Two threshold concerns, one confirmed harmless, one still open

**IoU matching is fine.** Measured across real consecutive polls: min 0.96,
median 1.00. Boxes on genuinely parked vehicles are essentially pixel-identical
between polls, so `IOU_MATCH_THRESHOLD = 0.60` has enormous headroom. An
earlier worry that loose boxes would break matching was specific to the old
model; yolov8m's boxes are tight.

**`MIN_BOX_CONFIDENCE = 0.40` is still suspect.** The detector's median
confidence on these frames is ~0.25, and on one camera 17 detections collapsed
to 3 under the gate. Because `stationary_polls` does not increment through a
miss, a track needs **4 consecutive** detections: at ~50% per-poll detection
odds that is 0.5⁴ ≈ 6%. A genuinely double-parked van could produce no incident
and no error. **Not changed** — lowering it trades false positives for recall,
which is exactly what the eval set exists to settle. Guessing a number now
would invent a result. Suggested starting point when tuning: ~0.15, leaning on
Gemini as the second gate.

**The curb band is probably backwards.** `data/zones/` does not exist, so every
camera uses the crude fallback: anything in the bottom 18% of frame is assumed
curbside and skipped. But these cameras look *down* the roadway, so the near
field is travel lane, not curb. That would reject genuine near-field
double-parkers and pass mid-frame moving traffic. `scripts/replay.py` draws the
band on each annotated frame specifically so this can be judged by eye.

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
| ~~Model may not resolve vehicles at 352x240~~ — **resolved**: it does, at modest confidence | Swapped to `yolov8m-640`; see finding 7 |
| **Confidence gate may starve the tracker** — now the top unknown | Replay harness reports whether any track reaches 4 polls; tune with `--analyze-only --confidence` |
| **Curb-band rule may be inverted for these camera angles** | Band is drawn on every annotated replay frame; hand-drawn zone polygons in `data/zones/` are the real fix |
| Workflow output key may not be `vehicle_boxes` | Moot while `ROBOFLOW_WORKSPACE`/`ROBOFLOW_WORKFLOW_ID` are blank — the client calls the model directly. Fallback logic retained |
| Nothing double-parks during a live demo | Build a **replay mode** over cached frames (near-free alongside the eval set) |
| Camera offline mid-demo | `demo_cameras()` cross-checks the live catalog and falls back |
| Heuristic false-positive rate unknown | This is what the eval set is for; Gemini is the second gate |

---

## Verification performed

- ✅ Live camera catalog fetch — 966 online cameras (2026-08-05 re-check)
- ✅ Live snapshot fetch — valid JPEG, 352x240, ~9-28KB
- ✅ Refresh-rate probe — 18/18 distinct frames at 5s cadence
- ✅ Curated camera loading with live cross-check — **6/6 still resolved on 2026-08-05**
- ✅ `pytest tests/ -q` — **59 passed**
- ✅ **Roboflow live inference — verified on all 6 cameras, both models**
- ✅ **Model benchmark** — 7 models on identical frames (finding 7)
- ✅ **Replay harness** — 3-poll run end to end; IoU min 0.96 / median 1.00
- ⬜ Replay at ≥12 polls — **not yet run**, so nothing has reached `STATIONARY_POLLS`
- ⬜ Gemini — key present in `.env`, but `gemini_judge.py` does not exist yet

One camera is worth knowing about: **Columbus Ave @ 65 St returned 0 detections**
in heavy rain, on a frame containing an unmissable white utility truck, even at
confidence 0.05. Wet-asphalt glare at dusk appears to defeat detection entirely.
Not a threshold problem. Consider dropping it from the demo set, or treat it as
a known hard negative.

Tests deliberately cover both directions: the negative heuristic cases would
all pass trivially if the heuristic simply rejected everything, so there is an
explicit positive test for the genuine double-parking pattern.

---

## Environment notes

Things about the dev setup that are easy to trip over.

- **Python 3.12.** Not 3.14: `inference-sdk` declares `<3.13` and publishes no
  wheel for 3.14, so the install fails outright. 3.10–3.12 are fine.
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

**2026-08-05** — Set up on a second machine (macOS, Python 3.12). Verified the
whole keyless half still works a day on: 966 cameras live, all 6 curated ones
still resolving. Got Roboflow and Gemini keys in and made the **first real
inference calls in the project's history** — the answer to the top open risk is
that vehicles *are* detectable at 352x240, at modest confidence.

Then three course corrections, in increasing order of importance. The model
Roboflow suggested turned out to be a 203-image fork of unverifiable
provenance, so it was benchmarked against six alternatives and replaced with
COCO `yolov8m-640`. The vehicle class filter was found to be silently
discarding `semi-trailer` detections — delivery trucks, the project's primary
target — and was rewritten to match per word. And a question worth recording:
*"the boxes are all on cars driving normally — what's doing the double-parking
detection?"* Nothing was. That is still true, and it is why `scripts/replay.py`
exists now.

Ended with detection genuinely solid and the temporal half still unproven.

**2026-08-04** — Project scoped via interview and greenlit. Built camera
ingestion, tracking, heuristic, and the Roboflow client. Key course-correction:
the camera source moved from 511NY to NYC DOT once it became clear 511NY only
covers highways inside the city. Roboflow proposed owning zone logic and
incident detection; declined after confirming its stateful blocks need
continuous video. Ended blocked on API keys, with the eval design flagged as
unresolved.
