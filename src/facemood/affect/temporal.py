"""Temporal fusion — dampens jitter before policy acts on affect."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from facemood.domain.types import EmotionEstimate, SmoothedEmotion


class EmotionSmoother:
    """
    Sliding-window majority vote + exponential confidence smoothing.

    Reduces spurious playlist triggers from single-frame misclassification.
    """

    def __init__(
        self,
        window: int = 7,
        ema_alpha: float = 0.42,
        min_confidence: float = 0.38,
    ) -> None:
        self._window = max(3, window)
        self._alpha = min(0.95, max(0.05, ema_alpha))
        self._min_confidence = min_confidence
        self._labels: Deque[str] = deque(maxlen=self._window)
        self._ema_conf: float = 0.5
        self._last_raw: str = "neutral"
        self.last_smoothed: Optional[SmoothedEmotion] = None

    def push(self, estimate: EmotionEstimate) -> SmoothedEmotion:
        self._last_raw = estimate.label
        self._labels.append(estimate.label)
        self._ema_conf = self._alpha * estimate.confidence + (1.0 - self._alpha) * self._ema_conf

        if len(self._labels) < self._window:
            self.last_smoothed = SmoothedEmotion(
                label=estimate.label,
                confidence=float(self._ema_conf),
                raw_label=estimate.label,
            )
            return self.last_smoothed

        counts: dict[str, int] = {}
        for lab in self._labels:
            counts[lab] = counts.get(lab, 0) + 1
        winner = max(counts, key=counts.get)

        if self._ema_conf < self._min_confidence and winner != "neutral":
            winner = "neutral"

        self.last_smoothed = SmoothedEmotion(
            label=winner,
            confidence=float(self._ema_conf),
            raw_label=self._last_raw,
        )
        return self.last_smoothed

    def reset(self) -> None:
        self._labels.clear()
        self._ema_conf = 0.5
        self._last_raw = "neutral"
        self.last_smoothed = None
