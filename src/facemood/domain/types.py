"""Core value types — immutable data across perception → affect → policy → audio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class FaceObservation:
    """Single frame: geometry + optional MediaPipe blendshape scores (0–1)."""

    bbox: BoundingBox
    landmarks_norm: Tuple[Tuple[float, float], ...]
    blendshapes: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionEstimate:
    """Output of affect layer before temporal smoothing."""

    label: str
    confidence: float


@dataclass(frozen=True)
class SmoothedEmotion:
    """After temporal fusion — stable label for policy decisions."""

    label: str
    confidence: float
    raw_label: str


@dataclass(frozen=True)
class PlaybackDecision:
    """Policy output: whether to start a new mood-aligned playback."""

    should_trigger: bool
    emotion: str
    reason: str
