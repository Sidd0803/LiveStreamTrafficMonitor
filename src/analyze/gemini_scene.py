"""Gemini flow-state classification — the judgment the geometry cannot make.

`density.py` measures what is measurable from boxes alone: how many vehicles,
how much of the frame they cover, and how far apart they sit relative to their
own size. What it has no access to is **the road**. Crowding tells you vehicles
are two car-lengths apart; it cannot tell you whether that is a comfortable gap
on a six-lane arterial or a standing queue on a one-way side street, because
nothing in a list of boxes says how wide the roadway is or how many lanes it
has. Fifteen well-spaced vehicles on Queens Boulevard and five bumper-to-bumper
on a Village side street can produce similar numbers and opposite answers.

Reading road capacity off the image is exactly what a VLM is good at and a
geometric metric structurally cannot do, and that is the entire reason Gemini is
in this pipeline. Both verdicts are recorded on every observation so the eval
can score geometry alone, Gemini alone, and the fusion — see BUILD_PLAN.

Two things this module treats as non-negotiable:

* **Structured output via a schema**, not prompt-and-parse. Free-text parsing
  fails on a public feed at 3am with no one watching.
* **A bad response degrades, it does not raise.** An unrecognized `flow_state`
  or a missing field yields a zero-confidence verdict, which lands below
  `GEMINI_MIN_CONFIDENCE` and makes the fusion rule fall back to geometry on its
  own. The pipeline needs no special case for it.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel

from src import config

log = logging.getLogger(__name__)

try:  # keep import failure actionable rather than a bare ImportError at startup
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


#: The only flow states the rest of the system understands. Shared with
#: density.py's classify() so geometric and Gemini verdicts are comparable —
#: the 3x3 confusion matrix in Phase 2 depends on the two vocabularies matching.
FLOW_STATES: tuple[str, str, str] = ("clear", "moderate", "jammed")

#: Where an unrecognized verdict lands. Paired with confidence 0.0 below, so it
#: is never actually used by the fusion rule — it exists to keep `flow_state` a
#: valid member of FLOW_STATES for anything that indexes on it (the confusion
#: matrix, the map's colour scale) rather than to assert anything about traffic.
FALLBACK_FLOW_STATE = "moderate"

#: Models paraphrase. When the paraphrase is unambiguous, recovering the meaning
#: beats discarding a good verdict over vocabulary — "heavy" plainly means
#: jammed. Anything outside this map is treated as an unusable answer rather
#: than guessed at.
_FLOW_SYNONYMS: dict[str, str] = {
    "free": "clear", "free-flowing": "clear", "flowing": "clear",
    "light": "clear", "empty": "clear", "open": "clear",
    "medium": "moderate", "busy": "moderate", "slow": "moderate",
    "heavy": "jammed", "congested": "jammed", "congestion": "jammed",
    "gridlock": "jammed", "gridlocked": "jammed", "jam": "jammed",
    "jammed-up": "jammed", "blocked": "jammed", "standstill": "jammed",
}

#: `reason` renders in a Streamlit card and `notable` in a map tooltip. Neither
#: has room for a paragraph, and a model that ignores "one sentence" should cost
#: us a truncated string rather than a broken layout.
_MAX_TEXT_CHARS = 300

#: Re-encode quality for the upscaled frame. High, because the input is already
#: a lossy 352x240 JPEG and stacking a second round of compression artifacts on
#: top of the first is exactly the detail this upscale exists to preserve.
_JPEG_QUALITY = 92

#: Deterministic verdicts. The eval set is scored by re-running frames, so
#: sampling noise would show up as accuracy noise and be indistinguishable from
#: a real threshold change.
_TEMPERATURE = 0.0

#: gemini-2.5-flash thinks by default. Left enabled but capped: the wide-road
#: vs narrow-road judgment below is a genuine reasoning step and gets visibly
#: worse without any budget, while an uncapped budget makes a 250-camera sweep
#: slow and costly for no measured gain.
_THINKING_BUDGET_TOKENS = 512


@dataclass(frozen=True)
class SceneVerdict:
    """Gemini's read of one frame."""

    flow_state: str
    #: 0..1, the model's own stated reliability. Below GEMINI_MIN_CONFIDENCE the
    #: fusion rule discards the verdict in favour of the geometric one.
    confidence: float
    #: One human-readable sentence, shown next to the frame in the dashboard.
    reason: str
    #: "" unless something genuinely unusual is visible. Empty is the norm.
    notable: str
    #: The JSON the model actually returned, kept for eval-time forensics.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class GeminiUnavailable(RuntimeError):
    """Raised when no usable Gemini configuration exists."""


class _SceneResponse(BaseModel):
    """The response schema handed to the API.

    `flow_state` is a Literal so the schema constrains generation to the three
    values rather than merely asking for them. The tolerant parsing below is
    kept anyway — a schema constrains the decoder, it does not guarantee the
    SDK's own validation succeeded, and google-genai silently leaves
    `response.parsed` as None when it does not.
    """

    flow_state: Literal["clear", "moderate", "jammed"]
    confidence: float
    reason: str
    notable: str


SCENE_PROMPT = """\
You are judging traffic FLOW STATE in a single still frame from a New York City \
DOT street camera. Reply with one JSON object matching the schema and nothing else.

CLASSIFY the busiest roadway visible in the frame as exactly one of:
  clear     Vehicles are spread out with several car-lengths of open road \
between them. No standing queue.
  moderate  Vehicles are noticeably closer together than free flow, or a short \
queue has formed at a signal, but gaps remain and the road is still usable.
  jammed    Vehicles are packed nose-to-tail with little or no gap across the \
usable lanes, or a long standing queue fills the roadway, or something is \
blocking it with no room to pass.

YOU CANNOT SEE MOTION. This is one still image, not video. Speed, direction, and \
whether any vehicle is moving at all are not available to you, so do not infer \
them. Judge only from spatial evidence actually present in the pixels:
  - the gap between vehicles, measured in car-lengths rather than pixels
  - whether vehicles form a continuous queue or are scattered independently
  - how much of each usable lane is covered by vehicle rather than open road
  - how many lanes carry traffic versus how many lanes the road has

DENSITY RELATIVE TO ROAD CAPACITY, NOT VEHICLE COUNT. This is the single most \
important instruction here. A wide arterial holding fifteen well-spaced vehicles \
is CLEAR, because a wide road carries a lot of traffic comfortably. A narrow \
one-way street with five vehicles bumper-to-bumper is JAMMED, despite the far \
smaller number. Raw count is not the signal. Ask yourself: is this road carrying \
more than a road of this width and this many lanes carries comfortably? Judge \
the moving lanes; a solid row of cars parked at the kerb is normal NYC street \
furniture and is not congestion.

IMAGE CONDITIONS. These frames are low-resolution NYC traffic camera stills, \
enlarged before you see them, so they are soft and blocky rather than sharp. \
Night, rain, glare, lens haze and water on the housing are all common. Distant \
vehicles may be only a few pixels of headlight. When conditions genuinely stop \
you from reading spacing — you cannot separate one vehicle from the next, or \
cannot tell roadway from sidewalk — LOWER YOUR CONFIDENCE instead of guessing. \
A confident wrong answer is far worse here than an honestly uncertain one. \
Clusters of headlights and taillights at night are still usable evidence of \
queue length; a frame that is simply dark or washed out is not.

FIELDS:
  flow_state  Exactly one of: clear, moderate, jammed.
  confidence  0.0 to 1.0. Your honest reliability given what is legible in THIS \
frame, not how typical the scene is. Use 0.8 or above only when individual \
vehicles and the gaps between them are clearly resolvable. Use below 0.4 when \
the frame is too dark, blurred, obstructed or empty of road to read.
  reason      ONE sentence of plain English citing the spatial evidence you \
used, e.g. "about eight cars queued nose-to-tail across both northbound lanes \
with no gaps". Do not restate these instructions.
  notable     Empty string "" unless something genuinely unusual and visible is \
obstructing or distorting traffic: construction closing a lane, a collision, a \
bus or truck stopped in a travel lane, emergency vehicles, flooding, a closure. \
Ordinary parked cars, ordinary traffic and ordinary weather are NOT notable. \
When in doubt, leave it empty."""


def upscale_for_scene(
    image_bytes: bytes, min_dim_px: int = config.SCENE_MIN_DIM_PX
) -> bytes:
    """Enlarge a frame so its long edge is at least `min_dim_px`, then re-encode.

    Every NYC DOT frame is 352x240. That is small enough that the model's own
    image preprocessing does most of the harm before the vision encoder ever
    sees it: at native size a whole car in the mid-field occupies fewer pixels
    than one encoder patch, so several vehicles and the gaps between them get
    averaged into a single token — and the gaps are the entire signal this
    module is asked to read.

    Resampling adds no information. What it buys is that the information already
    present lands on a patch grid fine enough to resolve individual vehicles,
    instead of being destroyed by a downscale we do not control. Lanczos is used
    because it preserves the edges between vehicle and road surface; a bilinear
    enlargement smears exactly the boundaries that spacing is judged from.

    Images already at or above the target are returned byte-identical. Never
    downscale — that would throw away the detail this exists to protect. The
    long edge is used rather than the short one so the aspect ratio is preserved
    without ever exceeding the target on either axis.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.load()
        width, height = img.size
        long_edge = max(width, height)
        if long_edge <= 0:
            raise ValueError("image has zero extent")
        if long_edge >= min_dim_px:
            return image_bytes

        scale = min_dim_px / long_edge
        # round() rather than int() so the smaller edge does not drift a whole
        # pixel away from the true aspect ratio on odd scale factors.
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        # Gemini wants RGB; NYC DOT frames are already RGB JPEG, but a camera
        # returning greyscale or a palette image must not crash the sweep.
        enlarged = img.convert("RGB").resize(target, Image.LANCZOS)

    buffer = io.BytesIO()
    enlarged.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return buffer.getvalue()


def normalize_flow_state(value: object) -> tuple[str, bool]:
    """Coerce a model-supplied flow state. Returns (state, was_understood).

    The caller needs the second element because "we could not understand the
    answer" and "the answer was moderate" must not be recorded identically —
    the first has to be stripped of its confidence so the fusion rule ignores it.
    """
    if not isinstance(value, str):
        return FALLBACK_FLOW_STATE, False

    # "Free Flowing" / "free_flowing" / "FREE-FLOWING" all reduce to one key.
    key = re.sub(r"[\s_]+", "-", value.strip().lower()).strip("-")
    if key in FLOW_STATES:
        return key, True
    if key in _FLOW_SYNONYMS:
        recovered = _FLOW_SYNONYMS[key]
        log.debug("recovered flow_state %r as %r", value, recovered)
        return recovered, True
    return FALLBACK_FLOW_STATE, False


def clamp_confidence(value: object) -> float:
    """Force any supplied confidence into 0..1.

    Models occasionally answer on a 0-100 scale or emit a bare string. Clamping
    rather than rejecting is deliberate: an out-of-range number still carries
    the model's intent at the ends of the scale, and this value is a gate
    (`>= GEMINI_MIN_CONFIDENCE`), not a calibrated probability. NaN is the one
    case with no sensible clamp, so it becomes 0.0 — no confidence at all.
    """
    try:
        confidence = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _clean_text(value: object) -> str:
    """One-line, length-capped text, or "" for anything unusable."""
    if not isinstance(value, str):
        return ""
    collapsed = re.sub(r"\s+", " ", value).strip()
    if len(collapsed) > _MAX_TEXT_CHARS:
        collapsed = collapsed[: _MAX_TEXT_CHARS - 1].rstrip() + "…"
    return collapsed


def verdict_from_payload(payload: object) -> SceneVerdict:
    """Build a SceneVerdict from whatever the model returned.

    Total function by design — there is no input that raises. A malformed
    verdict costs one camera its Gemini signal for one poll; an exception here
    would cost the whole sweep, and this runs unattended against a live feed.
    """
    raw: dict[str, Any] = payload if isinstance(payload, dict) else {"response": payload}

    flow_state, understood = normalize_flow_state(raw.get("flow_state"))
    confidence = clamp_confidence(raw.get("confidence"))
    if not understood:
        # Zeroing rather than keeping it is what makes this degrade cleanly: the
        # fusion rule already drops anything under GEMINI_MIN_CONFIDENCE, so an
        # unintelligible verdict routes itself to the geometric answer with no
        # extra branch anywhere downstream.
        log.warning("unusable flow_state %r — verdict zeroed", raw.get("flow_state"))
        confidence = 0.0

    return SceneVerdict(
        flow_state=flow_state,
        confidence=confidence,
        reason=_clean_text(raw.get("reason")),
        notable=_clean_text(raw.get("notable")),
        raw=raw,
    )


def _payload_from_text(text: str | None) -> object:
    """Last-resort JSON decode of the response body."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("Gemini returned non-JSON despite response_mime_type: %.200r", text)
        return {}


class SceneClassifier:
    """Frame in, flow-state verdict out."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL

        if genai is None:
            raise GeminiUnavailable(
                "google-genai is not installed — run: pip install google-genai"
            )
        if not self.api_key:
            raise GeminiUnavailable(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

        self.client = genai.Client(api_key=self.api_key)
        self._request_config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_SceneResponse,
            temperature=_TEMPERATURE,
            thinking_config=genai_types.ThinkingConfig(
                thinking_budget=_THINKING_BUDGET_TOKENS
            ),
        )

    def classify(self, image_bytes: bytes) -> SceneVerdict:
        """Classify flow state in a single JPEG frame.

        Transport and quota failures propagate — the pipeline catches them and
        records a geometry-only observation, which is the right response to
        "Gemini is down". Only *content* problems are absorbed here, because a
        response that arrived but made no sense is not an outage.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                genai_types.Part.from_bytes(
                    data=upscale_for_scene(image_bytes), mime_type="image/jpeg"
                ),
                SCENE_PROMPT,
            ],
            config=self._request_config,
        )

        # The SDK validates against _SceneResponse for us and leaves `parsed` as
        # None if that fails, so falling through to the raw body is the schema
        # violation path rather than a redundant second attempt.
        if isinstance(response.parsed, _SceneResponse):
            return verdict_from_payload(response.parsed.model_dump())
        return verdict_from_payload(_payload_from_text(response.text))
