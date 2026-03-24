"""
Face perception: BlazeFace + Face Landmarker on full frames.

Runs landmarker on the full image (not a tight crop) so landmark indices and
blendshapes stay consistent with MediaPipe's training distribution.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from facemood.domain.types import BoundingBox, FaceObservation

logger = logging.getLogger("facemood.perception")


def _blendshapes_face_to_dict(face_bs: Any) -> Dict[str, float]:
    """
    MediaPipe Python bindings differ: face blendshapes for one face may be
    a ClassificationList (`.categories`), a plain list of Category objects,
    or an iterable of protos with category_name/score.
    """
    out: Dict[str, float] = {}
    if face_bs is None:
        return out

    cats = getattr(face_bs, "categories", None)
    if cats is not None:
        for c in cats:
            out[getattr(c, "category_name", str(c))] = float(getattr(c, "score", 0.0))
        return out

    if isinstance(face_bs, (list, tuple)):
        for item in face_bs:
            if hasattr(item, "category_name") and hasattr(item, "score"):
                out[str(item.category_name)] = float(item.score)
            elif isinstance(item, dict):
                out[str(item.get("category_name", item.get("label", "")))] = float(
                    item.get("score", item.get("probability", 0.0))
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                out[str(item[0])] = float(item[1])
        return out

    return out


class FaceAnalysisService:
    """Detects face bbox + dense landmarks + 52 blendshape scores per frame."""

    def __init__(self, config) -> None:
        self._config = config
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision

        model_dir = Path.home() / ".facemood_models"
        model_dir.mkdir(exist_ok=True)

        face_detector_model = self._download_model(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            model_dir / "face_detector.tflite",
        )
        face_landmarker_model = self._download_model(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_dir / "face_landmarker.task",
        )

        min_det = config.get("face_detection.min_detection_confidence", 0.5)
        min_track = config.get("face_detection.min_tracking_confidence", 0.5)

        base_det = python.BaseOptions(model_asset_path=str(face_detector_model))
        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=base_det,
                running_mode=vision.RunningMode.VIDEO,
                min_detection_confidence=min_det,
            )
        )

        base_lm = python.BaseOptions(model_asset_path=str(face_landmarker_model))
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_lm,
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=min_det,
                min_tracking_confidence=min_track,
            )
        )

        self._ts_ms = 0
        logger.info("FaceAnalysisService ready (blendshapes ON, full-frame landmarker)")

    def _download_model(self, url: str, output_path: Path) -> Path:
        if output_path.exists():
            return output_path
        logger.info("Downloading %s ...", output_path.name)
        try:
            urllib.request.urlretrieve(url, output_path)
        except Exception:
            import requests

            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            output_path.write_bytes(r.content)
        return output_path

    def analyze(self, frame_bgr: np.ndarray) -> Optional[FaceObservation]:
        """BGR frame → optional FaceObservation."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)

        self._ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        det = self._detector.detect_for_video(mp_image, self._ts_ms)
        if not det.detections:
            return None

        bbox = det.detections[0].bounding_box
        box = BoundingBox(
            int(max(0, bbox.origin_x)),
            int(max(0, bbox.origin_y)),
            int(min(w, bbox.origin_x + bbox.width)),
            int(min(h, bbox.origin_y + bbox.height)),
        )

        lm_result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        if not lm_result.face_landmarks:
            return FaceObservation(bbox=box, landmarks_norm=tuple(), blendshapes={})

        face_lm = lm_result.face_landmarks[0]
        landmarks_norm = tuple((lm.x, lm.y) for lm in face_lm)

        blendshapes: Dict[str, float] = {}
        if lm_result.face_blendshapes:
            blendshapes = _blendshapes_face_to_dict(lm_result.face_blendshapes[0])

        return FaceObservation(bbox=box, landmarks_norm=landmarks_norm, blendshapes=blendshapes)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        observation: FaceObservation,
    ) -> np.ndarray:
        if not self._config.get("display.show_landmarks", True):
            return frame
        if not observation.landmarks_norm:
            return frame

        h, w = frame.shape[:2]
        color = tuple(self._config.get("display.landmark_color", [0, 255, 0]))
        size = self._config.get("display.landmark_size", 2)
        for x, y in observation.landmarks_norm:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), size, color, -1)
        return frame

    def close(self) -> None:
        try:
            self._detector.close()
            self._landmarker.close()
        except Exception as e:
            logger.debug("close: %s", e)
