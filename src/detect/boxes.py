"""Shared box geometry. Everything downstream speaks this type.

Roboflow returns center-based boxes (x, y, width, height); this module
normalizes to corner form (x1, y1, x2, y2) once, at the boundary, so no other
module has to remember which convention it is holding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in pixel coordinates, corner form."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "vehicle"
    confidence: float = 1.0

    @classmethod
    def from_center(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str = "vehicle",
        confidence: float = 1.0,
    ) -> "Box":
        """Build from Roboflow's center-based convention."""
        return cls(
            x1=x - width / 2,
            y1=y - height / 2,
            x2=x + width / 2,
            y2=y + height / 2,
            label=label,
            confidence=confidence,
        )

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clipped(self, frame_w: int, frame_h: int) -> "Box":
        return Box(
            x1=max(0.0, min(self.x1, frame_w)),
            y1=max(0.0, min(self.y1, frame_h)),
            x2=max(0.0, min(self.x2, frame_w)),
            y2=max(0.0, min(self.y2, frame_h)),
            label=self.label,
            confidence=self.confidence,
        )

    def padded(self, pad: float, frame_w: int, frame_h: int) -> "Box":
        """Expand by `pad` px on all sides, clipped to the frame.

        Used when cropping for Gemini: a tight crop gives the model no road
        context to judge lane position against.
        """
        return Box(
            x1=self.x1 - pad,
            y1=self.y1 - pad,
            x2=self.x2 + pad,
            y2=self.y2 + pad,
            label=self.label,
            confidence=self.confidence,
        ).clipped(frame_w, frame_h)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))


def iou(a: Box, b: Box) -> float:
    """Intersection over union. 0.0 when the boxes do not overlap."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)

    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0
