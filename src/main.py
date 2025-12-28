"""
Main application entry point for Face Mood Music Player.
"""

import os
import cv2
import time
import signal
import sys
from pathlib import Path

# Disable GPU for TensorFlow if present
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from .config import Config
from .utils import setup_logging, normalize_emotion
from .face_detector import FaceDetector
from .emotion_detector import EmotionDetector
from .music_player import MusicPlayer
from .statistics import Statistics


class FaceMoodApp:
    """Main application class."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.logger = setup_logging(self.config)
        self.running = False
        
        # Initialize components
        self.face_detector = FaceDetector(self.config)
        self.emotion_detector = EmotionDetector(self.config)
        self.music_player = MusicPlayer(self.config)
        self.statistics = Statistics()
        
        # Emotion tracking
        self.emotion_counter = {}
        self.last_played_emotion = None
        self.emotion_threshold_count = self.config.get('emotion.threshold_count', 40)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Face Mood Music Player initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)
    
    def _initialize_emotion_counter(self):
        """Initialize emotion counter with all supported emotions."""
        emotions = ['happy', 'sad', 'surprised', 'angry', 'neutral']
        self.emotion_counter = {emotion: 0 for emotion in emotions}
    
    def _update_emotion_counter(self, emotion: str) -> bool:
        """
        Update emotion counter and check if threshold is reached.
        
        Args:
            emotion: Detected emotion
            
        Returns:
            True if threshold reached and music should play
        """
        if emotion not in self.emotion_counter:
            return False
        
        self.emotion_counter[emotion] += 1
        
        # Check if threshold reached and emotion changed
        if (self.emotion_counter[emotion] >= self.emotion_threshold_count and
            emotion != self.last_played_emotion and
            emotion != 'neutral'):
            
            # Reset all counters
            for key in self.emotion_counter.keys():
                self.emotion_counter[key] = 0
            
            return True
        
        return False
    
    def _draw_ui(self, frame, emotion: str, confidence: float = 0.0):
        """
        Draw UI elements on frame.
        
        Args:
            frame: Frame to draw on
            emotion: Current detected emotion
            confidence: Detection confidence
        """
        font_scale = self.config.get('display.font_scale', 1.0)
        font_color = tuple(self.config.get('display.font_color', [0, 255, 0]))
        font_thickness = self.config.get('display.font_thickness', 2)
        
        # Draw emotion text
        emotion_text = f'Emotion: {emotion.upper()}'
        if confidence > 0:
            emotion_text += f' ({confidence:.0%})'
        cv2.putText(frame, emotion_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        
        # Draw counter information
        counter_text = f'Count: {self.emotion_counter.get(emotion, 0)}/{self.emotion_threshold_count}'
        cv2.putText(frame, counter_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, font_color, font_thickness)
        
        # Draw current track
        current_track = self.music_player.get_current_track()
        if current_track:
            track_text = f'Now: {current_track[:35]}...' if len(current_track) > 35 else f'Now: {current_track}'
            cv2.putText(frame, track_text, (10, frame.shape[0] - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, font_color, font_thickness)
        
        # Draw download progress if downloading
        if self.music_player.is_downloading():  # FIXED: This is now a method call
            progress = self.music_player.get_download_progress()  # FIXED: This is now a method call
            download_text = f'Downloading: {progress}%'
            cv2.putText(frame, download_text, (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 0), font_thickness)
        
        # Draw instructions
        instructions = "Press 'q' to quit | 's' for stats | 'p' pause | 't' toggle streaming"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (255, 255, 255), 1)
        
        # Draw volume
        volume_text = f'Volume: {int(self.music_player.get_volume() * 100)}%'
        cv2.putText(frame, volume_text, (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), 1)
        
        # Draw streaming status
        streaming_status = "Streaming: ON" if self.music_player.use_streaming else "Streaming: OFF"
        status_color = (0, 255, 0) if self.music_player.use_streaming else (0, 0, 255)
        cv2.putText(frame, streaming_status, (frame.shape[1] - 120, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, status_color, 1)
    
    def run(self):
        """Run the main application loop."""
        self.running = True
        self._initialize_emotion_counter()
        
        # Get camera settings
        device_id = self.config.get('camera.device_id', 0)
        width = self.config.get('camera.width', 640)
        height = self.config.get('camera.height', 480)
        
        try:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera {device_id}")
                return
            
            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            self.logger.info("Starting main loop")
            last_update_time = time.time()
            update_interval = self.config.get('emotion.update_interval', 1.0)
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    break
                
                current_time = time.time()
                
                # Detect face
                bbox = self.face_detector.detect_face(frame)
                
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    face_crop = frame[y_min:y_max, x_min:x_max]
                    
                    # Extract landmarks
                    landmarks = self.face_detector.extract_landmarks(face_crop)
                    
                    if landmarks:
                        # Calculate distances and detect emotion
                        distances = self.emotion_detector.calculate_distances(landmarks)
                        emotion = self.emotion_detector.detect_emotion(distances)
                        emotion = normalize_emotion(emotion)
                        confidence = self.emotion_detector.get_emotion_confidence(distances, emotion)
                        
                        # Update statistics
                        self.statistics.record_emotion(emotion)
                        
                        # Update emotion counter and play music if needed
                        if current_time - last_update_time >= update_interval:
                            if self._update_emotion_counter(emotion):
                                if self.music_player.play_emotion(emotion):
                                    self.last_played_emotion = emotion
                                    self.statistics.record_music_play(emotion)
                            last_update_time = current_time
                        
                        # Draw landmarks
                        frame = self.face_detector.draw_landmarks(frame, landmarks, bbox)
                        
                        # Draw UI
                        self._draw_ui(frame, emotion, confidence)
                    else:
                        self._draw_ui(frame, 'no_landmarks')
                else:
                    self._draw_ui(frame, 'no_face')
                
                # Display frame
                cv2.imshow('Face Mood Music Player', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.statistics.print_summary()
                elif key == ord('p'):
                    if self.music_player.is_playing():
                        self.music_player.pause()
                    else:
                        self.music_player.unpause()
                elif key == ord('t'):
                    self.music_player.toggle_streaming()
                elif key == ord('+') or key == ord('='):
                    # Increase volume
                    new_volume = min(1.0, self.music_player.get_volume() + 0.1)
                    self.music_player.set_volume(new_volume)
                elif key == ord('-'):
                    # Decrease volume
                    new_volume = max(0.0, self.music_player.get_volume() - 0.1)
                    self.music_player.set_volume(new_volume)
                elif key == ord('n'):
                    # Skip to next track
                    self.music_player.stop()
                    if self.last_played_emotion:
                        self.music_player.play_emotion(self.last_played_emotion)
                elif key == ord('0'):
                    # Mute
                    self.music_player.set_volume(0.0)
                elif key == ord('9'):
                    # Max volume
                    self.music_player.set_volume(1.0)
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the application and cleanup resources."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping application...")
        
        # Print final statistics
        self.statistics.print_summary()
        
        # Cleanup components
        self.music_player.cleanup()
        self.face_detector.cleanup()
        
        self.logger.info("Application stopped")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Mood Music Player')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--music-path', type=str, default=None,
                       help='Override music base path')
    parser.add_argument('--no-streaming', action='store_true',
                       help='Disable streaming mode')
    
    args = parser.parse_args()
    
    app = FaceMoodApp(args.config)
    
    if args.music_path:
        app.config.set('music.base_path', args.music_path)
    
    if args.no_streaming:
        app.music_player.use_streaming = False
        app.logger.info("Streaming disabled via command line")
    
    try:
        app.run()
    except KeyboardInterrupt:
        app.logger.info("Interrupted by user")
        app.stop()
    except Exception as e:
        app.logger.error(f"Fatal error: {e}", exc_info=True)
        app.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
