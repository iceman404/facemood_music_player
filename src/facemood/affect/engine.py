"""
Affect inference: prefer MediaPipe blendshapes (learned facial action units),
fallback to geometric heuristics on normalized landmarks.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from facemood.domain.types import EmotionEstimate, FaceObservation

logger = logging.getLogger("facemood.affect")


class AffectEngine:
    """Maps FaceObservation → EmotionEstimate."""

    # MediaPipe Face Mesh landmark indices (478 points) — geometry fallback
    _LM = {
        "left_mouth": 61,
        "right_mouth": 291,
        "upper_lip": 13,
        "lower_lip": 14,
        "left_brow_inner": 70,
        "right_brow_inner": 300,
        "left_brow_outer": 105,
        "right_brow_outer": 334,
        "left_eye_top": 159,
        "left_eye_bottom": 145,
        "right_eye_top": 386,
        "right_eye_bottom": 374,
    }

    def __init__(self, config) -> None:
        self._config = config

    def estimate(self, observation: FaceObservation) -> EmotionEstimate:
        if observation.blendshapes:
            est = self._from_blendshapes(observation.blendshapes)
            if est.label != "neutral" or est.confidence >= 0.45:
                return est
        if len(observation.landmarks_norm) >= 468:
            return self._from_geometry(observation.landmarks_norm)
        return EmotionEstimate("neutral", 0.35)

    def _from_blendshapes(self, bs: Dict[str, float]) -> EmotionEstimate:
        """Competitive scoring over coarse emotion classes (MediaPipe ARKit-style names)."""
        g = bs.get

        happy = (
            g("mouthSmileLeft", 0.0)
            + g("mouthSmileRight", 0.0)
            + 0.35 * (g("cheekSquintLeft", 0.0) + g("cheekSquintRight", 0.0))
        )
        sad = g("mouthFrownLeft", 0.0) + g("mouthFrownRight", 0.0) + 0.6 * g("browInnerUp", 0.0)
        surprised = (
            g("jawOpen", 0.0)
            + 0.45 * (g("eyeWideLeft", 0.0) + g("eyeWideRight", 0.0))
            + 0.25 * g("browOuterUpLeft", 0.0)
            + 0.25 * g("browOuterUpRight", 0.0)
        )
        angry = g("browDownLeft", 0.0) + g("browDownRight", 0.0) + 0.4 * (
            g("noseSneerLeft", 0.0) + g("noseSneerRight", 0.0)
        )

        scores = {
            "happy": happy,
            "sad": sad,
            "surprised": surprised,
            "angry": angry,
        }
        best = max(scores, key=scores.get)
        peak = scores[best]
        runner = sorted(scores.values(), reverse=True)
        second = runner[1] if len(runner) > 1 else 0.0
        margin = peak - second

        if peak < 0.08:
            return EmotionEstimate("neutral", 0.4)

        # Confidence: strength + margin (calibrated heuristically)
        confidence = float(min(0.98, 0.35 + peak * 1.4 + margin * 0.5))
        return EmotionEstimate(best, confidence)

    def _from_geometry(self, lm: Tuple[Tuple[float, float], ...]) -> EmotionEstimate:
        """Legacy rule-based path — same structure as v1, full-frame normalized coords."""
        L = self._LM

        def p(i: int) -> np.ndarray:
            return np.array(lm[i])

        left_mouth, right_mouth = p(L["left_mouth"]), p(L["right_mouth"])
        upper_lip, lower_lip = p(L["upper_lip"]), p(L["lower_lip"])
        left_brow_inner = lm[L["left_brow_inner"]]
        right_brow_inner = lm[L["right_brow_inner"]]
        left_brow_outer = lm[L["left_brow_outer"]]
        right_brow_outer = lm[L["right_brow_outer"]]

        mouth_distance = float(np.linalg.norm(left_mouth - right_mouth))
        mouth_height = float(np.linalg.norm(upper_lip - lower_lip))
        eyebrow_inner_distance = abs(left_brow_inner[1] - right_brow_inner[1])
        eyebrow_outer_distance = abs(left_brow_outer[1] - right_brow_outer[1])
        eye_openness_left = abs(lm[L["left_eye_top"]][1] - lm[L["left_eye_bottom"]][1])
        eye_openness_right = abs(lm[L["right_eye_top"]][1] - lm[L["right_eye_bottom"]][1])

        sad_c = self._config.get("emotion.sad", {})
        happy_c = self._config.get("emotion.happy", {})
        sur_c = self._config.get("emotion.surprised", {})
        ang_c = self._config.get("emotion.angry", {})

        if (
            mouth_height < sad_c.get("mouth_threshold", 0.1)
            and eyebrow_inner_distance > sad_c.get("brow_inner_threshold", 0.05)
            and eyebrow_outer_distance < sad_c.get("brow_outer_threshold", 0.07)
            and eye_openness_left < sad_c.get("eye_openness_threshold", 0.15)
            and eye_openness_right < sad_c.get("eye_openness_threshold", 0.15)
        ):
            return EmotionEstimate("sad", 0.62)

        if mouth_height > sur_c.get("mouth_height_threshold", 0.15):
            return EmotionEstimate("surprised", 0.65)

        if mouth_distance > happy_c.get("mouth_distance_threshold", 0.45):
            return EmotionEstimate("happy", 0.62)

        if left_brow_inner[1] > ang_c.get("brow_position_threshold", 0.6) and right_brow_inner[1] > ang_c.get(
            "brow_position_threshold", 0.6
        ):
            return EmotionEstimate("angry", 0.58)

        return EmotionEstimate("neutral", 0.5)
