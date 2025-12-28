"""
Statistics tracking module for emotion detection.
"""

from collections import defaultdict
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger('facemood.statistics')


class Statistics:
    """Track and manage emotion detection statistics."""
    
    def __init__(self):
        """Initialize statistics tracker."""
        self.emotion_counts: Dict[str, int] = defaultdict(int)
        self.emotion_durations: Dict[str, float] = defaultdict(float)
        self.total_detections = 0
        self.music_plays: Dict[str, int] = defaultdict(int)
        self.session_start = datetime.now()
        self.last_emotion_time: Dict[str, datetime] = {}
        
        logger.info("Statistics tracker initialized")
    
    def record_emotion(self, emotion: str) -> None:
        """
        Record an emotion detection.
        
        Args:
            emotion: Detected emotion
        """
        self.emotion_counts[emotion] += 1
        self.total_detections += 1
        self.last_emotion_time[emotion] = datetime.now()
    
    def record_music_play(self, emotion: str) -> None:
        """
        Record a music play event.
        
        Args:
            emotion: Emotion that triggered music
        """
        self.music_plays[emotion] += 1
        logger.info(f"Music played for emotion: {emotion} (total: {self.music_plays[emotion]})")
    
    def get_summary(self) -> Dict:
        """
        Get statistics summary.
        
        Returns:
            Dictionary with statistics summary
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        # Calculate percentages
        emotion_percentages = {}
        if self.total_detections > 0:
            for emotion, count in self.emotion_counts.items():
                emotion_percentages[emotion] = (count / self.total_detections) * 100
        
        return {
            'total_detections': self.total_detections,
            'emotion_counts': dict(self.emotion_counts),
            'emotion_percentages': emotion_percentages,
            'music_plays': dict(self.music_plays),
            'session_duration_seconds': session_duration,
            'session_start': self.session_start.isoformat()
        }
    
    def get_most_common_emotion(self) -> Optional[str]:
        """
        Get the most frequently detected emotion.
        
        Returns:
            Most common emotion or None
        """
        if not self.emotion_counts:
            return None
        return max(self.emotion_counts.items(), key=lambda x: x[1])[0]
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.emotion_counts.clear()
        self.emotion_durations.clear()
        self.total_detections = 0
        self.music_plays.clear()
        self.session_start = datetime.now()
        self.last_emotion_time.clear()
        logger.info("Statistics reset")
    
    def print_summary(self) -> None:
        """Print statistics summary to console."""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("EMOTION DETECTION STATISTICS")
        print("="*50)
        print(f"Total Detections: {summary['total_detections']}")
        print(f"Session Duration: {summary['session_duration_seconds']:.1f} seconds")
        print("\nEmotion Distribution:")
        for emotion, count in summary['emotion_counts'].items():
            percentage = summary['emotion_percentages'].get(emotion, 0)
            print(f"  {emotion.capitalize()}: {count} ({percentage:.1f}%)")
        print("\nMusic Plays:")
        for emotion, count in summary['music_plays'].items():
            print(f"  {emotion.capitalize()}: {count}")
        print("="*50 + "\n")

