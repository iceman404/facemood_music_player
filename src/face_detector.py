"""
Face detection module using MediaPipe Tasks API.
"""

import cv2
import numpy as np
import os
import urllib.request
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger('facemood.face_detector')


class FaceDetector:
    """Face detection and landmark extraction using MediaPipe Tasks API."""
    
    def __init__(self, config):
        """
        Initialize face detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        try:
            # Import MediaPipe Tasks modules
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            self.mp = mp
            self.vision = vision
            
            # Download model files if not exists
            model_dir = Path.home() / '.facemood_models'
            model_dir.mkdir(exist_ok=True)
            
            # Model URLs (public MediaPipe models)
            face_detector_model = self._download_model(
                'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
                model_dir / 'face_detector.tflite'
            )
            
            face_landmarker_model = self._download_model(
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                model_dir / 'face_landmarker.task'
            )
            
            min_detection_confidence = config.get('face_detection.min_detection_confidence', 0.5)
            min_tracking_confidence = config.get('face_detection.min_tracking_confidence', 0.5)
            
            # Initialize face detector
            base_options = python.BaseOptions(model_asset_path=str(face_detector_model))
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_detection_confidence=min_detection_confidence
            )
            self.detector = vision.FaceDetector.create_from_options(options)
            
            # Initialize face landmarker
            base_options_lm = python.BaseOptions(model_asset_path=str(face_landmarker_model))
            options_lm = vision.FaceLandmarkerOptions(
                base_options=base_options_lm,
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options_lm)
            
            logger.info("Face detector initialized (MediaPipe Tasks API)")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            print(f"DEBUG: MediaPipe initialization error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _download_model(self, url: str, output_path: Path) -> Path:
        """Download model file if it doesn't exist."""
        if output_path.exists():
            logger.info(f"Model already exists: {output_path}")
            return output_path
        
        logger.info(f"Downloading model from {url}...")
        try:
            # Download with progress
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownloading... {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
            print()  # New line after progress
            
            logger.info(f"Model downloaded to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            
            # Try alternative download method
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Model downloaded (alternative method): {output_path}")
            return output_path
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame and return bounding box.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Bounding box (x_min, y_min, x_max, y_max) or None
        """
        try:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Get timestamp in milliseconds
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            # Detect faces
            detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.detections:
                detection = detection_result.detections[0]
                bbox = detection.bounding_box
                
                # MediaPipe Tasks API returns absolute coordinates
                x_min = bbox.origin_x
                y_min = bbox.origin_y
                x_max = x_min + bbox.width
                y_max = y_min + bbox.height
                
                return (x_min, y_min, x_max, y_max)
        except Exception as e:
            logger.error(f"Error detecting face: {e}")
        
        return None
    
    def extract_landmarks(self, face_crop: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Extract facial landmarks from face crop.
        
        Args:
            face_crop: Cropped face image (BGR)
            
        Returns:
            List of landmark coordinates [(x, y), ...] or None
        """
        try:
            # Resize for consistent processing
            face_crop_resized = cv2.resize(face_crop, (200, 200))
            face_crop_rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=face_crop_rgb)
            
            # Get timestamp in milliseconds
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            # Detect landmarks
            landmark_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if landmark_result.face_landmarks:
                face_landmarks = landmark_result.face_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in face_landmarks]
                return landmarks
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
        
        return None
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float]], 
                      bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Draw landmarks on frame.
        
        Args:
            frame: Frame to draw on
            landmarks: List of landmark coordinates
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            Frame with landmarks drawn
        """
        if not self.config.get('display.show_landmarks', True):
            return frame
        
        try:
            x_min, y_min, x_max, y_max = bbox
            color = tuple(self.config.get('display.landmark_color', [0, 255, 0]))
            size = self.config.get('display.landmark_size', 2)
            
            for landmark in landmarks:
                cx = int(landmark[0] * (x_max - x_min) + x_min)
                cy = int(landmark[1] * (y_max - y_min) + y_min)
                cv2.circle(frame, (cx, cy), size, color, -1)
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
        
        return frame
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'detector'):
                self.detector.close()
            if hasattr(self, 'landmarker'):
                self.landmarker.close()
        except:
            pass
        
        self.detector = None
        self.landmarker = None
        logger.info("Face detector cleaned up")
