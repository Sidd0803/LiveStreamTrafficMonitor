# LiveStreamTrafficMonitor — NYC Double-Parking Detector

Detects double-parked vehicles in live NYC traffic-camera feeds by combining
object detection, temporal tracking, and vision-model adjudication.

A learning project built as a precursor to a hackathon, to get hands-on with
**NYC DOT / 511NY open camera data, Roboflow, Gemini, and GCP**. A working
end-to-end demo is the definition of done; ambition is secondary to completion.

> **Status:** Phase 1 in progress. Camera ingestion, tracking, and the spatial
> heuristic are built and verified. Detection and adjudication are next and
> need API keys. See [docs/PROGRESS.md](docs/PROGRESS.md).

---

## How it works

```
NYC DOT /api/cameras ──┐
                       ├─> poller ──> frame (352x240 JPEG)
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

Roboflow does fast deterministic box detection. The tracker and heuristic do
temporal and spatial reasoning in plain Python — cheap and debuggable. Gemini is
called **only on candidates**, so it acts as an adjudicator rather than a
detector, which keeps cost near zero.

### What counts as "double parked"

All three must hold:

1. The vehicle holds near-identical position across **4 consecutive polls** (~48-60s).
2. It sits **outside the curb/parking zone** per the camera's lane geometry.
3. **Gemini confirms** it from the cropped image.

---

## Setup on a new machine

```bash
git clone https://github.com/Sidd0803/LiveStreamTrafficMonitor.git
cd LiveStreamTrafficMonitor
```

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

Copy the env template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Needed for | Where to get it |
|---|---|---|
| `ROBOFLOW_API_KEY` | Phase 1 detection | [app.roboflow.com](https://app.roboflow.com) → Settings → API Keys |
| `GEMINI_API_KEY` | Phase 1 adjudication | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `NY511_API_KEY` | Phase 3, optional | [511ny.org/developers/resources](https://511ny.org/developers/resources) — **manual approval, request early** |
| `GCP_PROJECT_ID`, `GCS_BUCKET` | Phase 3 | Your GCP project |

**NYC DOT camera data needs no key** — that half runs immediately after clone.

### Running

Imports are rooted at the repo, so `PYTHONPATH` must include it:

```bash
export PYTHONPATH=$PWD && python -m pytest tests/ -q
```

On Windows PowerShell:

```bash
$env:PYTHONPATH = $PWD; python -m pytest tests/ -q
```

The Streamlit dashboard (once Phase 1 completes):

```bash
streamlit run app.py
```

---

## Layout

```
src/
  config.py              every tunable threshold — tuned against the eval set
  sources/nycdot.py      camera catalog + rate-limited snapshot fetch
  sources/ny511.py       511NY catalog + event feed (additive, Phase 3)
  detect/boxes.py        Box geometry + IoU
  detect/tracker.py      IoU matching, stationarity counters
  detect/heuristic.py    curb/lane geometry -> candidates
  detect/roboflow_client.py
  detect/gemini_judge.py
  storage/               local filesystem and GCP backends behind one interface
  pipeline.py            orchestrates one poll cycle
data/demo_cameras.json   hand-curated cameras, with rejections and reasons
eval/                    labeled set + precision/recall harness
tests/                   tracker and heuristic tests (no keys, no network)
docs/                    build plan and progress log
```

Storage sits behind a single interface with local and GCP implementations, so
Phase 1 runs entirely offline and the GCP migration is a config flip.

---

## Responsible use

These are public civic camera feeds. The poller enforces a **12-second minimum
interval per camera**, sets a descriptive User-Agent, and the project does not
republish raw camera imagery. 511NY additionally throttles at 10 calls/60s.

This is a learning project, not an enforcement tool. It identifies vehicle
positions, not vehicles, people, or plates.

---

## Docs

- [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) — full scope, phases, and decisions
- [docs/PROGRESS.md](docs/PROGRESS.md) — what's done, findings, and what's next
