"""NYC Traffic Congestion Index — Streamlit dashboard.

The read side of the pipeline. A sweep polls NYC DOT cameras, runs each frame
through Roboflow for vehicle boxes and Gemini for a scene verdict, and writes an
`Observation` per camera per instant. This app renders those observations:

* a **congestion map** of the city, one dot per camera, coloured by flow state;
* a **per-camera detail** panel showing the cached frame, the geometric metrics,
  and the two verdicts **side by side**;
* a **summary strip** and a **recent observations table**.

The side-by-side layout is the point, not a layout convenience. The project
exists to ask whether a scale-free geometric ratio and a vision model agree
about what "jammed" means, so the UI is built to make disagreement between them
impossible to miss rather than quietly averaging it away into a single verdict.

Run it:

    streamlit run app.py                 # live store, demo fallback offered
    streamlit run app.py -- --demo       # force synthetic data

Demo mode generates plausible observations across real NYC coordinates with no
network, no API keys and no stored sweep, so the dashboard is developable and
demoable from a fresh clone. It is labelled loudly on screen wherever it is in
effect: a demo that silently looks live is worse than no demo at all.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src import demo_data

# --- Presentation constants ------------------------------------------------
# One colour per flow state, used by the map, the summary strip and the
# verdict panels alike. Defined once so a dot on the map and the badge in the
# detail panel can never disagree about what "moderate" looks like.
FLOW_COLORS: dict[str, str] = {
    "clear": "#1a9850",     # green
    "moderate": "#f2a900",  # amber
    "jammed": "#d73027",    # red
}
FLOW_RGB: dict[str, list[int]] = {
    "clear": [26, 152, 80],
    "moderate": [242, 169, 0],
    "jammed": [215, 48, 39],
}
UNKNOWN_COLOR = "#9e9e9e"
UNKNOWN_RGB = [158, 158, 158]

FLOW_ORDER = ("clear", "moderate", "jammed")

# Roughly the centre of the five boroughs, zoomed to fit them.
NYC_CENTER = (40.7128, -73.935)
NYC_ZOOM = 9.6


# --- Time helpers ----------------------------------------------------------
def to_eastern(dt: datetime) -> tuple[datetime, str]:
    """Convert a UTC observation timestamp to New York local time.

    Observations are stored in UTC, which is correct and also unreadable to the
    audience: everyone in the room is thinking in Eastern time. Returns the
    converted value and the label to print beside it, because a bare clock time
    with no zone is exactly how this kind of dashboard misleads people.

    Falls back to UTC if the tz database is unavailable (a real possibility on a
    bare Windows install without `tzdata`) rather than raising.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return dt.astimezone(ZoneInfo("America/New_York")), "ET"
    except Exception:
        return dt.astimezone(timezone.utc), "UTC"


def format_ts(dt: datetime) -> str:
    local, label = to_eastern(dt)
    return f"{local:%Y-%m-%d %H:%M:%S} {label}"


# --- Data loading ----------------------------------------------------------
class StoreUnavailable(RuntimeError):
    """The storage backend could not be reached or does not exist yet."""


def _open_store():
    """Import and open the store lazily.

    Lazy because `src.storage.local` is a sibling module under active
    construction; an import at module scope would take the whole dashboard down
    with it, including demo mode, which has no business depending on storage.
    """
    try:
        from src.storage.local import get_store
    except Exception as exc:  # not yet built, or broken
        raise StoreUnavailable(f"storage backend unavailable: {exc}") from exc
    try:
        return get_store()
    except Exception as exc:
        raise StoreUnavailable(f"could not open the store: {exc}") from exc


@st.cache_data(show_spinner="Loading observations…")
def load_live(nonce: int, limit: int) -> tuple[list, list]:
    """Read the latest-per-camera set and the recent feed from the store.

    `nonce` is not used in the body — it exists so the Refresh button can bust
    the cache by incrementing it. Without it, caching would make the refresh
    button a lie.
    """
    store = _open_store()
    latest = list(store.latest_per_camera())
    recent = list(store.recent(limit=limit))
    return latest, recent


@st.cache_data(show_spinner="Generating demo observations…")
def load_demo(nonce: int, seed: int) -> tuple[list, list]:
    """Synthetic observations, seeded so a rehearsed demo stays rehearsed."""
    observations = demo_data.demo_observations(seed=seed)
    return demo_data.latest_per_camera(observations), observations


# --- Frame helpers ---------------------------------------------------------
def resolve_frame(frame_path: str | None) -> Path | None:
    """Return an on-disk path for a stored frame, or None.

    `frame_path` may be relative to the repo root (local store) or a `gs://`
    URI (Phase 3, GCS). Only local files can be rendered here; anything else
    returns None so the caller shows a placeholder instead of a broken image.
    """
    if not frame_path:
        return None
    if "://" in frame_path:
        return None
    path = Path(frame_path)
    if not path.is_absolute():
        path = config.REPO_ROOT / path
    return path if path.exists() else None


# --- Rendering: shared pieces ----------------------------------------------
def flow_badge(state: str | None) -> str:
    """Coloured inline badge for a flow state. Markdown, so it works anywhere."""
    if not state:
        return f"<span style='color:{UNKNOWN_COLOR};font-weight:600'>no verdict</span>"
    color = FLOW_COLORS.get(state, UNKNOWN_COLOR)
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-weight:600'>{state.upper()}</span>"
    )


def observations_frame(observations: list) -> pd.DataFrame:
    """Flatten Observations into a DataFrame for the map and the table."""
    rows = []
    # Resolve the zone label once: it must be identical for every row, or the
    # column name would vary and pandas would produce a ragged frame.
    _, tz_label = to_eastern(datetime.now(timezone.utc))
    for obs in observations:
        local, _ = to_eastern(obs.captured_at)
        rows.append(
            {
                "camera_id": obs.camera_id,
                "camera": obs.camera_name,
                "borough": obs.area,
                "latitude": obs.latitude,
                "longitude": obs.longitude,
                f"captured ({tz_label})": local.strftime("%Y-%m-%d %H:%M:%S"),
                "_captured_at": obs.captured_at,
                "vehicles": obs.vehicle_count,
                "occupancy": obs.occupancy,
                "crowding": obs.crowding,
                "geometric": obs.geometric_flow,
                "gemini": obs.gemini_flow,
                "gemini conf": obs.gemini_confidence,
                "final": obs.final_flow,
                "agree": (
                    None
                    if obs.gemini_flow is None
                    else obs.gemini_flow == obs.geometric_flow
                ),
                "color": FLOW_COLORS.get(obs.final_flow, UNKNOWN_COLOR),
                "_rgb": FLOW_RGB.get(obs.final_flow, UNKNOWN_RGB),
            }
        )
    return pd.DataFrame(rows)


# --- Rendering: the four views ---------------------------------------------
def render_summary(latest: list, is_demo: bool) -> None:
    """Counts by flow state, total vehicles, and when the data was captured."""
    counts = {state: 0 for state in FLOW_ORDER}
    for obs in latest:
        counts[obs.final_flow] = counts.get(obs.final_flow, 0) + 1

    total_vehicles = sum(obs.vehicle_count for obs in latest)
    newest = max((obs.captured_at for obs in latest), default=None)

    cols = st.columns(6)
    cols[0].metric("Cameras", len(latest))
    cols[1].metric("Clear", counts.get("clear", 0))
    cols[2].metric("Moderate", counts.get("moderate", 0))
    cols[3].metric("Jammed", counts.get("jammed", 0))
    cols[4].metric("Vehicles seen", total_vehicles)
    cols[5].metric(
        "Last updated",
        format_ts(newest).split(" ", 1)[1] if newest else "—",
        help=format_ts(newest) if newest else None,
    )

    if newest is not None:
        source = "SYNTHETIC DEMO DATA" if is_demo else "live sweep"
        st.caption(f"Newest observation: {format_ts(newest)} · source: {source}")


def render_map(latest: list) -> None:
    """The headline: every camera as a dot, coloured by its final flow state."""
    st.subheader("Congestion map")

    df = observations_frame(latest)
    # Cameras with no fix would otherwise land in the Atlantic off Africa.
    df = df[(df["latitude"].abs() > 0.01) & (df["longitude"].abs() > 0.01)]
    if df.empty:
        st.info("No cameras have usable coordinates.")
        return

    # Colour alone is not self-explanatory, so the legend is part of the view
    # rather than a caption someone has to hunt for.
    legend = " &nbsp;&nbsp; ".join(
        f"<span style='color:{FLOW_COLORS[s]};font-size:20px'>●</span> {s}"
        for s in FLOW_ORDER
    )
    st.markdown(
        f"{legend} &nbsp;&nbsp; <span style='color:{UNKNOWN_COLOR};font-size:20px'>"
        "●</span> unknown",
        unsafe_allow_html=True,
    )

    try:
        import pydeck as pdk  # ships with Streamlit; not an added dependency

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["longitude", "latitude"],
            get_fill_color="_rgb",
            get_radius=180,
            radius_min_pixels=6,
            radius_max_pixels=22,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
            pickable=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=NYC_CENTER[0],
                longitude=NYC_CENTER[1],
                zoom=NYC_ZOOM,
                pitch=0,
            ),
            tooltip={
                "html": (
                    "<b>{camera}</b><br/>{borough}<br/>"
                    "flow: <b>{final}</b><br/>"
                    "geometric: {geometric} &nbsp; gemini: {gemini}<br/>"
                    "vehicles: {vehicles}"
                )
            },
        )
        st.pydeck_chart(deck, width="stretch")
    except Exception as exc:
        # pydeck's basemap tiles need network. The dots still render without
        # them, but if the deck fails outright, st.map is the simpler fallback.
        st.warning(f"Falling back to the simple map ({exc}).")
        st.map(df, latitude="latitude", longitude="longitude", color="color", size=90)


def render_detail(latest: list) -> None:
    """One camera in full: the frame, the metrics, and both verdicts."""
    st.subheader("Camera detail")

    if not latest:
        return

    by_label = {f"{o.camera_name} — {o.area}": o for o in sorted(
        latest, key=lambda o: (o.area, o.camera_name)
    )}
    label = st.selectbox("Camera", list(by_label), key="camera_picker")
    obs = by_label[label]

    left, right = st.columns([1, 1])

    with left:
        frame = resolve_frame(obs.frame_path)
        if frame is not None:
            st.image(str(frame), caption=f"{obs.camera_name} · {format_ts(obs.captured_at)}")
        elif obs.frame_path:
            st.info(
                f"Frame not available locally: `{obs.frame_path}`\n\n"
                "Remote frames (GCS) are not rendered in the local dashboard."
            )
        else:
            st.info("No cached frame for this observation.")

    with right:
        st.markdown(f"**Final verdict** {flow_badge(obs.final_flow)}", unsafe_allow_html=True)
        st.caption(format_ts(obs.captured_at))

        m = st.columns(3)
        m[0].metric("Vehicles", obs.vehicle_count)
        m[1].metric("Occupancy", f"{obs.occupancy:.1%}")
        m[2].metric(
            "Crowding",
            f"{obs.crowding:.2f}" if obs.crowding is not None else "—",
            help=(
                "Median (nearest-neighbour distance / own box diagonal). "
                "Scale-free: ~1.0 is bumper to bumper. Undefined below "
                f"{config.MIN_VEHICLES_FOR_CROWDING} vehicles."
            ),
        )
        if obs.by_class:
            st.caption(
                "By class: "
                + ", ".join(f"{k} {v}" for k, v in sorted(obs.by_class.items()))
            )

    st.markdown("#### The two judgments")
    st.caption(
        "Both are stored for every observation. Comparing them is the point of "
        "the project, so neither is hidden behind the fused result."
    )

    geo_col, gem_col = st.columns(2)

    with geo_col:
        st.markdown("**Geometric** (crowding ratio)", unsafe_allow_html=True)
        st.markdown(flow_badge(obs.geometric_flow), unsafe_allow_html=True)
        if obs.crowding is not None:
            st.caption(
                f"crowding {obs.crowding:.2f} against thresholds "
                f"jammed ≤ {config.CROWDING_JAMMED_MAX}, "
                f"moderate ≤ {config.CROWDING_MODERATE_MAX}"
            )
        else:
            st.caption(
                f"Fewer than {config.MIN_VEHICLES_FOR_CROWDING} vehicles — "
                "the ratio is too noisy to compute, so the frame reads clear."
            )

    with gem_col:
        st.markdown(f"**Gemini** ({config.GEMINI_MODEL})", unsafe_allow_html=True)
        st.markdown(flow_badge(obs.gemini_flow), unsafe_allow_html=True)
        if obs.gemini_flow is None:
            st.caption("Not called, or unavailable for this frame.")
        else:
            conf = obs.gemini_confidence
            st.caption(f"confidence {conf:.2f}" if conf is not None else "confidence —")
            if obs.gemini_reason:
                st.write(f"_{obs.gemini_reason}_")
            if obs.gemini_notable:
                st.warning(f"Notable: {obs.gemini_notable}")

    # Disagreement is the interesting case, so it gets called out explicitly
    # rather than left for the reader to spot by comparing two badges.
    if obs.gemini_flow is None:
        st.info(
            f"No Gemini verdict — the fused result falls back to geometry "
            f"({obs.geometric_flow})."
        )
    elif obs.gemini_flow != obs.geometric_flow:
        conf = obs.gemini_confidence or 0.0
        winner = (
            f"Gemini wins (confidence {conf:.2f} ≥ {config.GEMINI_MIN_CONFIDENCE})"
            if conf >= config.GEMINI_MIN_CONFIDENCE
            else f"geometry wins (confidence {conf:.2f} < {config.GEMINI_MIN_CONFIDENCE})"
        )
        st.error(
            f"**Verdicts disagree** — geometry says *{obs.geometric_flow}*, "
            f"Gemini says *{obs.gemini_flow}*. Fusion rule: {winner}, "
            f"so the final verdict is **{obs.final_flow}**."
        )
    else:
        st.success(f"Both judgments agree: **{obs.final_flow}**.")


def render_recent(recent: list) -> None:
    """The raw feed, newest first."""
    st.subheader("Recent observations")
    if not recent:
        st.info("No observations recorded yet.")
        return

    df = observations_frame(recent).sort_values("_captured_at", ascending=False)
    df = df.drop(columns=["_captured_at", "_rgb", "color", "latitude", "longitude"])
    st.dataframe(df, width="stretch", hide_index=True)


# --- Empty / error states --------------------------------------------------
def render_no_data(reason: str) -> None:
    """Never a traceback. An empty store is a normal state on a fresh clone."""
    st.info(
        f"**No observations to show.** {reason}\n\n"
        "Run a sweep to populate the store, or switch on **Demo data** in the "
        "sidebar to explore the dashboard with synthetic observations."
    )
    st.code("PYTHONPATH=$PWD python -m src.pipeline", language="bash")


# --- Main ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="NYC Traffic Congestion Index",
        page_icon="🚦",
        layout="wide",
    )

    # `streamlit run app.py -- --demo` for a machine with no data and no network.
    cli_demo = "--demo" in sys.argv

    st.sidebar.title("Controls")
    demo_toggle = st.sidebar.toggle(
        "Demo data (synthetic)",
        value=cli_demo,
        help="Generate plausible observations locally. No network, no API keys, "
             "no stored sweep required.",
    )
    seed = st.sidebar.number_input(
        "Demo seed", min_value=0, max_value=9999, value=7, step=1,
        help="Same seed, same map — so a rehearsed demo stays rehearsed.",
        disabled=not demo_toggle,
    )
    limit = st.sidebar.slider("Recent rows", 50, 1000, 200, step=50)

    if "nonce" not in st.session_state:
        st.session_state.nonce = 0
    if st.sidebar.button("Refresh", width="stretch"):
        # Clearing the cache is what makes this button actually refresh; the
        # nonce then guarantees a distinct cache key even within the same run.
        st.cache_data.clear()
        st.session_state.nonce += 1

    st.title("🚦 NYC Traffic Congestion Index")

    is_demo = demo_toggle
    latest: list = []
    recent: list = []
    failure: str | None = None

    if demo_toggle:
        latest, recent = load_demo(st.session_state.nonce, int(seed))
    else:
        try:
            latest, recent = load_live(st.session_state.nonce, int(limit))
        except StoreUnavailable as exc:
            failure = str(exc)
        except Exception as exc:  # a broken store must not be a traceback
            failure = f"error reading the store: {exc}"

    if is_demo:
        # Loud, unmissable, and above the fold. A demo that silently looks live
        # is worse than no demo.
        st.error(
            "🧪 **DEMO MODE — every number on this page is synthetic.** "
            "Generated locally by `src/demo_data.py` at real NYC coordinates. "
            "No cameras were polled, no model was called. Turn off *Demo data* "
            "in the sidebar to read the live store."
        )

    if not latest and not recent:
        render_no_data(failure or "The store is empty.")
        return

    render_summary(latest, is_demo)
    st.divider()
    render_map(latest)
    st.divider()
    render_detail(latest)
    st.divider()
    render_recent(recent[: int(limit)])

    st.sidebar.divider()
    st.sidebar.caption(
        "Flow states come from two independent judgments — a scale-free "
        "crowding ratio and a Gemini scene verdict — fused at "
        f"confidence ≥ {config.GEMINI_MIN_CONFIDENCE}. Both are always stored."
    )


if __name__ == "__main__":
    main()
