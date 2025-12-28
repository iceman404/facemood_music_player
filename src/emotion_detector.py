"""
Emotion detection module based on facial landmarks.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger('facemood.emotion_detector')


class EmotionDetector:
    """Detect emotions from facial landmarks."""
    
    # Landmark indices for MediaPipe Face Mesh
    LANDMARKS = {
        'left_mouth': 61,
        'right_mouth': 291,
        'upper_lip': 13,
        'lower_lip': 14,
        'left_eye': 33,
        'right_eye': 133,
        'left_brow_inner': 70,
        'right_brow_inner': 300,
        'left_brow_outer': 105,
        'right_brow_outer': 334,
        'left_eye_top': 159,
        'left_eye_bottom': 145,
        'right_eye_top': 386,
        'right_eye_bottom': 374
    }
    
    def __init__(self, config):
        """
        Initialize emotion detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("Emotion detector initialized")
    
    def calculate_distances(self, landmarks: list) -> Tuple[float, ...]:
        """
        Calculate facial feature distances from landmarks.
        
        Args:
            landmarks: List of landmark coordinates [(x, y), ...]
            
        Returns:
            Tuple of calculated distances
        """
        try:
            # Get landmark points
            left_mouth = np.array(landmarks[self.LANDMARKS['left_mouth']])
            right_mouth = np.array(landmarks[self.LANDMARKS['right_mouth']])
            upper_lip = np.array(landmarks[self.LANDMARKS['upper_lip']])
            lower_lip = np.array(landmarks[self.LANDMARKS['lower_lip']])
            left_brow_inner = landmarks[self.LANDMARKS['left_brow_inner']]
            right_brow_inner = landmarks[self.LANDMARKS['right_brow_inner']]
            left_brow_outer = landmarks[self.LANDMARKS['left_brow_outer']]
            right_brow_outer = landmarks[self.LANDMARKS['right_brow_outer']]
            
            # Calculate distances
            mouth_distance = np.linalg.norm(left_mouth - right_mouth)  # Mouth width
            mouth_height = np.linalg.norm(upper_lip - lower_lip)  # Mouth height
            eyebrow_inner_distance = abs(left_brow_inner[1] - right_brow_inner[1])
            eyebrow_outer_distance = abs(left_brow_outer[1] - right_brow_outer[1])
            eye_openness_left = abs(
                landmarks[self.LANDMARKS['left_eye_top']][1] - 
                landmarks[self.LANDMARKS['left_eye_bottom']][1]
            )
            eye_openness_right = abs(
                landmarks[self.LANDMARKS['right_eye_top']][1] - 
                landmarks[self.LANDMARKS['right_eye_bottom']][1]
            )
            
            return (
                mouth_distance, mouth_height, eyebrow_inner_distance,
                eyebrow_outer_distance, eye_openness_left, eye_openness_right,
                left_brow_inner, right_brow_inner
            )
        except Exception as e:
            logger.error(f"Error calculating distances: {e}")
            return (0.0,) * 8
    
    def detect_emotion(self, distances: Tuple[float, ...]) -> str:
        """
        Detect emotion from facial feature distances.
        
        Args:
            distances: Tuple of calculated distances
            
        Returns:
            Detected emotion string
        """
        if len(distances) < 8:
            return 'neutral'
        
        (mouth_distance, mouth_height, eyebrow_inner_distance,
         eyebrow_outer_distance, eye_openness_left, eye_openness_right,
         left_brow_inner, right_brow_inner) = distances
        
        # Get thresholds from config
        sad_config = self.config.get('emotion.sad', {})
        happy_config = self.config.get('emotion.happy', {})
        surprised_config = self.config.get('emotion.surprised', {})
        angry_config = self.config.get('emotion.angry', {})
        
        # Sadness detection
        if (mouth_height < sad_config.get('mouth_threshold', 0.1) and
            eyebrow_inner_distance > sad_config.get('brow_inner_threshold', 0.05) and
            eyebrow_outer_distance < sad_config.get('brow_outer_threshold', 0.07) and
            eye_openness_left < sad_config.get('eye_openness_threshold', 0.15) and
            eye_openness_right < sad_config.get('eye_openness_threshold', 0.15)):
            return 'sad'
        
        # Surprise detection
        if mouth_height > surprised_config.get('mouth_height_threshold', 0.15):
            return 'surprised'
        
        # Happiness detection
        if mouth_distance > happy_config.get('mouth_distance_threshold', 0.45):
            return 'happy'
        
        # Anger detection
        if (left_brow_inner[1] > angry_config.get('brow_position_threshold', 0.6) and
            right_brow_inner[1] > angry_config.get('brow_position_threshold', 0.6)):
            return 'angry'
        
        return 'neutral'
    
    def get_emotion_confidence(self, distances: Tuple[float, ...], emotion: str) -> float:
        """
        Calculate confidence score for detected emotion.
        
        Args:
            distances: Tuple of calculated distances
            emotion: Detected emotion
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Simple confidence calculation based on how well features match thresholds
        # This is a basic implementation - can be enhanced with ML models
        if emotion == 'neutral':
            return 0.5
        
        # For now, return a moderate confidence
        # In a production system, this would use a trained model
        return 0.7

