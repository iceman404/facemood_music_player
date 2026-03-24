"""When to trigger mood music — hysteresis separate from raw CV."""

from __future__ import annotations

import time
from typing import Optional

from facemood.domain.types import PlaybackDecision, SmoothedEmotion


class PlaybackPolicy:
    """
    Requires stable non-neutral affect + confidence, enforces cooldown.

    Decouples "what the face looks like" from "should we change the playlist now".
    """

    def __init__(self, config) -> None:
        self._stable_needed = int(config.get("policy.stable_frames", 12))
        self._cooldown_s = float(config.get("policy.cooldown_seconds", 25.0))
        self._min_conf = float(config.get("policy.min_confidence", 0.42))
        self._neutral_holdoff = int(config.get("policy.neutral_reset_frames", 6))

        self._last_trigger_mono: float = 0.0
        self._streak_label: Optional[str] = None
        self._streak_count = 0
        self._neutral_count = 0
        self._last_played: Optional[str] = None
        self._diverge_frames = 0

    def evaluate(self, smoothed: SmoothedEmotion) -> PlaybackDecision:
        lab = smoothed.label
        conf = smoothed.confidence

        # Allow the same mood to trigger again after user clearly left it (v1 semantics).
        if self._last_played is not None and lab != self._last_played:
            self._diverge_frames += 1
            if self._diverge_frames >= self._neutral_holdoff:
                self._last_played = None
                self._diverge_frames = 0
        else:
            self._diverge_frames = 0

        if lab == "neutral":
            self._neutral_count += 1
            if self._neutral_count >= self._neutral_holdoff:
                self._streak_label = None
                self._streak_count = 0
            return PlaybackDecision(False, lab, "neutral_hold")

        self._neutral_count = 0

        if conf < self._min_conf:
            self._streak_label = None
            self._streak_count = 0
            return PlaybackDecision(False, lab, "low_confidence")

        if lab != self._streak_label:
            self._streak_label = lab
            self._streak_count = 1
        else:
            self._streak_count += 1

        if self._streak_count < self._stable_needed:
            return PlaybackDecision(False, lab, f"unstable_{self._streak_count}/{self._stable_needed}")

        now = time.monotonic()
        if now - self._last_trigger_mono < self._cooldown_s:
            return PlaybackDecision(False, lab, "cooldown")

        if lab == self._last_played:
            return PlaybackDecision(False, lab, "same_as_last_played")

        self._last_trigger_mono = now
        self._last_played = lab
        self._streak_count = 0
        return PlaybackDecision(True, lab, "trigger")

    def forget_last_played(self) -> None:
        """Allow immediate re-trigger (e.g. user skipped)."""
        self._last_played = None
