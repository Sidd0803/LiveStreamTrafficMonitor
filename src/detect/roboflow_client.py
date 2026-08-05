"""Roboflow vehicle detection — the only job here is frame in, boxes out.

Two call paths, both normalizing to the same `Box` list:

* **Workflow** (preferred when configured) — the hosted "one image in, vehicle
  boxes out" workflow. Returns `inference_id` / `model_id` alongside the boxes,
  which is useful for tracing a bad detection back to a specific run.
* **Direct model** — plain `infer()` against `ROBOFLOW_MODEL_ID`.

The workflow is deliberately detection-only. Tracking, zone logic and incident
detection live in Python (`tracker.py`, `heuristic.py`) because Roboflow's
stateful blocks (ByteTrack, Time in Zone) require continuous video, and our
frames are ~12s apart — there is no meaningful inter-frame history for them to
work with.

Direct model inference is kept as an automatic fallback: the workflow adds a
config dependency (workspace, workflow id, output key) that can drift, and a
demo should degrade rather than die.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src import config
from src.detect.boxes import Box

log = logging.getLogger(__name__)

try:  # keep import failure actionable rather than a bare ImportError at startup
    from inference_sdk import InferenceHTTPClient
except ImportError:  # pragma: no cover
    InferenceHTTPClient = None  # type: ignore[assignment]


@dataclass
class DetectionResult:
    """Boxes plus the provenance the workflow gives us."""

    boxes: list[Box]
    #: Roboflow's id for this inference run, when available.
    inference_id: str | None = None
    model_id: str | None = None
    #: "workflow" or "model" — which path produced this.
    via: str = "model"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class RoboflowUnavailable(RuntimeError):
    """Raised when no usable Roboflow configuration exists."""


def _find_predictions(node: Any, depth: int = 0) -> list[dict] | None:
    """Recursively locate a Roboflow predictions list in a response.

    Workflow responses nest differently depending on how the workflow's outputs
    were named and whether blocks were chained, so rather than hardcoding one
    path we search for the first list of dicts that carries detection geometry.
    Direct `infer()` responses hit this on the first level.
    """
    if depth > 6:
        return None

    if isinstance(node, list):
        if node and isinstance(node[0], dict) and _looks_like_prediction(node[0]):
            return node
        for item in node:
            found = _find_predictions(item, depth + 1)
            if found is not None:
                return found
        return None

    if isinstance(node, dict):
        preds = node.get("predictions")
        if isinstance(preds, list) and (
            not preds or (isinstance(preds[0], dict) and _looks_like_prediction(preds[0]))
        ):
            return preds
        # Check the configured output key first, then everything else.
        keys = [config.ROBOFLOW_WORKFLOW_OUTPUT_KEY] + [
            k for k in node if k != config.ROBOFLOW_WORKFLOW_OUTPUT_KEY
        ]
        for key in keys:
            if key in node:
                found = _find_predictions(node[key], depth + 1)
                if found is not None:
                    return found
    return None


def _looks_like_prediction(item: dict) -> bool:
    """Roboflow object-detection predictions are center-based boxes."""
    return all(k in item for k in ("x", "y", "width", "height"))


def _scan(node: Any, key: str, depth: int = 0) -> str | None:
    """Pull a scalar like inference_id/model_id from anywhere in the response."""
    if depth > 6:
        return None
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
        for sub in node.values():
            found = _scan(sub, key, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _scan(item, key, depth + 1)
            if found:
                return found
    return None


def is_vehicle_label(label: str) -> bool:
    """Does this class label denote a vehicle that could be double parked?

    Word-level rather than whole-string, because label vocabularies differ per
    model in ways exact matching handles badly: "pickup-truck", "semi-trailer"
    and "vehicles" all denote vehicles but match none of "pickup", "truck",
    "vehicle" exactly. Splitting on separators and de-pluralizing catches them.

    Kept deliberately permissive. A wrongly-kept box costs one Gemini call at
    the adjudication gate; a wrongly-dropped box is invisible and removes a
    real double-parker from the pipeline before anything can flag it.
    """
    words = {w for w in re.split(r"[-_/\s]+", label.strip().lower()) if w}
    # "vehicles" -> "vehicle", "buses" -> "buse" (harmless; "bus" also present)
    words |= {w[:-1] for w in words if len(w) > 3 and w.endswith("s")}

    if words & config.NON_VEHICLE_WORDS:
        return False
    return bool(words & config.VEHICLE_WORDS)


def _to_boxes(predictions: list[dict]) -> list[Box]:
    """Convert Roboflow center-based predictions to our corner-form boxes."""
    boxes: list[Box] = []
    for pred in predictions:
        label = str(pred.get("class", "vehicle")).strip().lower()
        confidence = float(pred.get("confidence", 0.0))

        if confidence < config.MIN_BOX_CONFIDENCE:
            continue
        # Models carry varied label vocabularies; COCO-trained ones include
        # people, traffic lights and signs. Keep only things that could be
        # double parked.
        if not is_vehicle_label(label):
            log.debug("dropping non-vehicle class %r", label)
            continue

        try:
            boxes.append(
                Box.from_center(
                    x=float(pred["x"]),
                    y=float(pred["y"]),
                    width=float(pred["width"]),
                    height=float(pred["height"]),
                    label=label,
                    confidence=confidence,
                )
            )
        except (KeyError, TypeError, ValueError):
            log.warning("skipping malformed prediction: %r", pred)
    return boxes


class VehicleDetector:
    """Frame in, vehicle boxes out."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        workspace: str | None = None,
        workflow_id: str | None = None,
    ) -> None:
        self.api_key = api_key or config.ROBOFLOW_API_KEY
        self.model_id = model_id or config.ROBOFLOW_MODEL_ID
        self.workspace = workspace if workspace is not None else config.ROBOFLOW_WORKSPACE
        self.workflow_id = (
            workflow_id if workflow_id is not None else config.ROBOFLOW_WORKFLOW_ID
        )

        if InferenceHTTPClient is None:
            raise RoboflowUnavailable(
                "inference-sdk is not installed — run: pip install inference-sdk"
            )
        if not self.api_key:
            raise RoboflowUnavailable(
                "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

        self.client = InferenceHTTPClient(
            api_url=config.ROBOFLOW_API_URL, api_key=self.api_key
        )
        self._workflow_failed = False

    @property
    def uses_workflow(self) -> bool:
        return bool(self.workspace and self.workflow_id) and not self._workflow_failed

    def detect(self, image_bytes: bytes) -> DetectionResult:
        """Detect vehicles in a single JPEG frame."""
        if self.uses_workflow:
            try:
                return self._detect_via_workflow(image_bytes)
            except Exception as exc:  # noqa: BLE001 — any failure falls back
                # Latch the failure so we stop retrying the workflow every poll;
                # a misconfigured output key would otherwise cost a round trip
                # per frame for the life of the demo.
                self._workflow_failed = True
                log.warning(
                    "workflow call failed (%s) — falling back to direct model "
                    "inference against %s for the rest of this session",
                    exc, self.model_id,
                )
        return self._detect_via_model(image_bytes)

    def _detect_via_workflow(self, image_bytes: bytes) -> DetectionResult:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.run_workflow(
            workspace_name=self.workspace,
            workflow_id=self.workflow_id,
            images={config.ROBOFLOW_WORKFLOW_IMAGE_KEY: encoded},
        )
        predictions = _find_predictions(response)
        if predictions is None:
            raise ValueError(
                f"no predictions found in workflow response; "
                f"expected output key {config.ROBOFLOW_WORKFLOW_OUTPUT_KEY!r}. "
                f"Top-level shape: {_describe(response)}"
            )
        return DetectionResult(
            boxes=_to_boxes(predictions),
            inference_id=_scan(response, "inference_id"),
            model_id=_scan(response, "model_id") or self.model_id,
            via="workflow",
            raw=response if isinstance(response, dict) else {"response": response},
        )

    def _detect_via_model(self, image_bytes: bytes) -> DetectionResult:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.infer(encoded, model_id=self.model_id)
        predictions = _find_predictions(response) or []
        return DetectionResult(
            boxes=_to_boxes(predictions),
            inference_id=_scan(response, "inference_id"),
            model_id=self.model_id,
            via="model",
            raw=response if isinstance(response, dict) else {"response": response},
        )


def _describe(node: Any, depth: int = 0) -> str:
    """Compact structural summary, for making a bad response debuggable."""
    if depth > 2:
        return "..."
    if isinstance(node, dict):
        return "{" + ", ".join(f"{k}: {_describe(v, depth + 1)}" for k in list(node)[:6] for v in [node[k]]) + "}"
    if isinstance(node, list):
        return f"[{len(node)} x {_describe(node[0], depth + 1) if node else 'empty'}]"
    return type(node).__name__
