"""Shared affect pipeline + playback — used by CLI and GUI."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from facemood.affect import AffectEngine, EmotionSmoother
from facemood.hud_overlay import render_hud
from facemood.domain.types import EmotionEstimate
from facemood.perception import FaceAnalysisService
from facemood.playback import PlaybackService
from facemood.policy import PlaybackPolicy
from facemood.telemetry import SessionStatistics
from facemood.utils import normalize_emotion

if TYPE_CHECKING:
    from facemood.config import Config

logger = logging.getLogger("facemood.session")


class FaceMoodSession:
    def __init__(self, config: "Config") -> None:
        self.config = config
        self.face = FaceAnalysisService(config)
        self.affect = AffectEngine(config)
        self.smoother = EmotionSmoother(
            window=int(config.get("temporal.window", 7)),
            ema_alpha=float(config.get("temporal.ema_alpha", 0.42)),
            min_confidence=float(config.get("temporal.min_confidence", 0.38)),
        )
        self.policy = PlaybackPolicy(config)
        self.playback = PlaybackService(config)
        self.stats = SessionStatistics()

        self._last_policy_reason = ""
        self._last_infer = 0.0
        self._infer_interval = float(config.get("emotion.update_interval", 0.12))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run perception → affect → policy → playback; draw landmarks + HUD. Returns same frame."""
        now = time.time()
        obs = self.face.analyze(frame)

        display_label = "no_face"
        display_conf = 0.0
        raw = "—"

        if obs and obs.landmarks_norm:
            if now - self._last_infer >= self._infer_interval:
                self._last_infer = now
                est = self.affect.estimate(obs)
                est = EmotionEstimate(normalize_emotion(est.label), est.confidence)
                smoothed = self.smoother.push(est)
                decision = self.policy.evaluate(smoothed)
                self._last_policy_reason = decision.reason
                display_label = smoothed.label
                display_conf = smoothed.confidence
                raw = smoothed.raw_label
                self.stats.record_emotion(smoothed.label)

                if decision.should_trigger:
                    if self.playback.play_emotion(decision.emotion):
                        self.stats.record_music_play(decision.emotion)
            elif self.smoother.last_smoothed is not None:
                ls = self.smoother.last_smoothed
                display_label = ls.label
                display_conf = ls.confidence
                raw = ls.raw_label

            frame = self.face.draw_landmarks(frame, obs)
            self._draw_hud(frame, display_label, display_conf, raw)
        else:
            self._draw_hud(frame, display_label, display_conf, raw)

        return frame

    def _draw_hud(self, frame: np.ndarray, label: str, confidence: float, raw_label: str) -> None:
        if not self.config.get("display.futuristic_hud", True):
            import cv2

            fs = self.config.get("display.font_scale", 1.0)
            col = tuple(self.config.get("display.font_color", [0, 255, 0]))
            cv2.putText(
                frame,
                f"{label.upper()} {confidence:.0%}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                col,
                2,
            )
            return

        render_hud(
            frame,
            label=label,
            confidence=confidence,
            raw_label=raw_label,
            policy_reason=self._last_policy_reason,
            track=self.playback.get_current_track(),
            downloading=self.playback.is_downloading(),
            download_pct=self.playback.get_download_progress(),
            volume=self.playback.get_volume(),
            source_tag=self.playback.source_tag(),
            footer_hint=self.config.get("display.footer_hint"),
        )

    def close(self) -> None:
        self.playback.cleanup()
        self.face.close()
