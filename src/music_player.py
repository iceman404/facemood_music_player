"""
Music Player with background threading for streaming downloads.
"""

import os
import random
import pygame
import time
import tempfile
import threading
import queue
from pathlib import Path
from typing import Optional, List, Dict
import logging

from .utils import get_supported_audio_files

logger = logging.getLogger('facemood.music_player')


class MusicPlayer:
    """Music player with local files and background streaming downloads."""
    
    def __init__(self, config):
        """
        Initialize music player.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.current_emotion: Optional[str] = None
        self.current_track: Optional[str] = None
        self.temp_file: Optional[str] = None
        self.local_playlist: List[Path] = []
        self.streaming_playlist: List[Dict] = []
        self.use_streaming = True  # Set to False to disable streaming
        self.volume = config.get('music.volume', 0.7)
        self.fade_duration = config.get('music.fade_duration', 1.0)
        self.supported_formats = config.get('music.supported_formats', ['.mp3', '.wav', '.ogg', '.flac'])
        
        # Threading and queues
        self.download_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        self.running = True
        self._is_downloading = False  # Changed to private attribute
        self._download_progress = 0   # Changed to private attribute
        
        # Emotion to search queries
        self.emotion_queries = {
            'happy': ['happy music', 'upbeat pop', 'feel good songs'],
            'sad': ['sad songs', 'emotional music', 'melancholic'],
            'surprised': ['epic music', 'cinematic soundtrack', 'orchestral'],
            'angry': ['rock music', 'metal songs', 'intense music'],
            'neutral': ['lofi beats', 'ambient music', 'background music']
        }
        
        # Check for required packages
        self.yt_dlp_available = self._check_yt_dlp()
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            pygame.mixer.music.set_volume(self.volume)
            logger.info("Music player initialized")
            logger.info(f"Streaming available: {self.yt_dlp_available}")
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            raise
        
        # Start background threads
        self._start_threads()
    
    def _check_yt_dlp(self) -> bool:
        """Check if yt-dlp is installed."""
        try:
            import yt_dlp
            return True
        except ImportError:
            logger.warning("yt-dlp not installed. Streaming will not work.")
            return False
    
    def _start_threads(self):
        """Start background threads."""
        # Download thread
        self.download_thread = threading.Thread(
            target=self._download_worker,
            daemon=True,
            name="DownloadWorker"
        )
        self.download_thread.start()
        
        # Playback thread
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="PlaybackWorker"
        )
        self.playback_thread.start()
        
        logger.info("Background threads started")
    
    def _download_worker(self):
        """Background worker for downloading music."""
        while self.running:
            try:
                # Get task from queue
                task = self.download_queue.get(timeout=0.5)
                if task is None:
                    break
                
                emotion, video_id, title = task
                logger.info(f"Downloading: {title[:40]}...")
                self._is_downloading = True
                self._download_progress = 10
                
                # Download the audio
                temp_file = self._download_audio(video_id, title)
                
                self._download_progress = 90
                
                if temp_file and os.path.exists(temp_file):
                    # Queue for playback
                    self.playback_queue.put((emotion, temp_file, title))
                    logger.info(f"Download complete: {title[:40]}...")
                else:
                    logger.error(f"Download failed: {title[:40]}...")
                
                self._is_downloading = False
                self._download_progress = 0
                self.download_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Download worker error: {e}")
                self._is_downloading = False
                self._download_progress = 0
    
    def _playback_worker(self):
        """Background worker for playing music."""
        while self.running:
            try:
                # Get task from queue
                task = self.playback_queue.get(timeout=0.5)
                if task is None:
                    break
                
                emotion, temp_file, title = task
                logger.info(f"Playing: {title[:40]}...")
                
                # Play the file
                success = self._play_audio_file(temp_file)
                
                if success:
                    self.current_emotion = emotion
                    self.current_track = title
                    self.temp_file = temp_file
                    logger.info(f"Now playing: {title[:40]}...")
                else:
                    logger.error(f"Failed to play: {title[:40]}...")
                    # Clean up failed file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                self.playback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback worker error: {e}")
    
    def _download_audio(self, video_id: str, title: str) -> Optional[str]:
        """Download audio using yt-dlp."""
        if not self.yt_dlp_available:
            return None
        
        try:
            import yt_dlp
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            temp_dir = tempfile.gettempdir()
            
            # Create a safe filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:30]
            temp_file = os.path.join(temp_dir, f"facemood_{video_id}_{safe_title}.mp3")
            
            # Check if already downloaded
            if os.path.exists(temp_file):
                logger.info(f"Using cached file: {temp_file}")
                return temp_file
            
            # Download options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, f"facemood_{video_id}"),
                'quiet': False,
                'no_warnings': False,
                'progress_hooks': [self._download_progress_hook],
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }],
            }
            
            # Download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self._download_progress = 20
                ydl.download([url])
                self._download_progress = 80
            
            # Rename to include title
            downloaded_file = os.path.join(temp_dir, f"facemood_{video_id}.mp3")
            if os.path.exists(downloaded_file):
                os.rename(downloaded_file, temp_file)
                return temp_file
            
            return None
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    def _download_progress_hook(self, d):
        """Progress hook for downloads."""
        if d['status'] == 'downloading':
            try:
                downloaded = d.get('downloaded_bytes', 0)
                total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                if total > 0:
                    self._download_progress = 20 + int((downloaded / total) * 60)
            except:
                pass
    
    def _play_audio_file(self, filepath: str) -> bool:
        """Play an audio file."""
        try:
            # Fade out if playing
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.fadeout(int(self.fade_duration * 1000))
                time.sleep(self.fade_duration)
            
            # Load and play
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            return True
            
        except Exception as e:
            logger.error(f"Error playing file: {e}")
            return False
    
    def search_youtube(self, query: str, limit: int = 3) -> List[Dict]:
        """Search YouTube for music."""
        if not self.yt_dlp_available:
            return []
        
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'default_search': f'ytsearch{limit}',
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(query, download=False)
                
                if not result or 'entries' not in result:
                    return []
                
                tracks = []
                for entry in result['entries'][:limit]:
                    if entry and entry.get('duration', 0) <= 300:  # 5 min max
                        tracks.append({
                            'id': entry.get('id'),
                            'title': entry.get('title', 'Unknown'),
                            'artist': entry.get('uploader', 'Unknown'),
                            'duration': entry.get('duration', 0),
                            'url': f"https://www.youtube.com/watch?v={entry.get('id')}"
                        })
                
                return tracks
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def load_streaming_playlist(self, emotion: str) -> bool:
        """Load streaming playlist for an emotion."""
        if not self.use_streaming or not self.yt_dlp_available:
            return False
        
        try:
            queries = self.emotion_queries.get(emotion, ['music'])
            query = random.choice(queries)
            
            logger.info(f"Searching for: {query}")
            self.streaming_playlist = self.search_youtube(query + " music", limit=3)
            
            if not self.streaming_playlist:
                logger.warning(f"No streaming tracks found for: {emotion}")
                return False
            
            logger.info(f"Found {len(self.streaming_playlist)} tracks for {emotion}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading streaming: {e}")
            return False
    
    def load_local_playlist(self, emotion: str) -> bool:
        """Load local playlist for an emotion."""
        try:
            music_path = self.config.get_music_path(emotion)
            self.local_playlist = get_supported_audio_files(music_path, self.supported_formats)
            
            if not self.local_playlist:
                logger.warning(f"No local files for: {emotion}")
                return False
            
            logger.info(f"Loaded {len(self.local_playlist)} local tracks for {emotion}")
            return True
        except Exception as e:
            logger.error(f"Error loading local: {e}")
            return False
    
    def play_emotion(self, emotion: str) -> bool:
        """
        Play music for an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            True if music started or queued
        """
        # If same emotion, continue
        if emotion == self.current_emotion:
            if not pygame.mixer.music.get_busy():
                return self.play_next()
            return True
        
        self.current_emotion = emotion
        
        # Try streaming
        if self.use_streaming:
            if self.load_streaming_playlist(emotion):
                return self.queue_streaming_track()
        
        # Fallback to local
        if self.load_local_playlist(emotion):
            return self.play_next_local()
        
        return False
    
    def queue_streaming_track(self) -> bool:
        """Queue a streaming track for download/playback."""
        if not self.streaming_playlist:
            return False
        
        try:
            track = random.choice(self.streaming_playlist)
            
            # Add to download queue (non-blocking)
            self.download_queue.put((
                self.current_emotion,
                track['id'],
                track['title']
            ))
            
            logger.info(f"Queued for download: {track['title'][:40]}...")
            return True  # Return immediately
            
        except Exception as e:
            logger.error(f"Error queueing track: {e}")
            return False
    
    def play_next_local(self) -> bool:
        """Play next local track."""
        if not self.local_playlist:
            return False
        
        try:
            track = random.choice(self.local_playlist)
            self.current_track = track.name
            
            # Stop current
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.fadeout(int(self.fade_duration * 1000))
                time.sleep(self.fade_duration)
            
            # Play new
            pygame.mixer.music.load(str(track))
            pygame.mixer.music.play()
            
            logger.info(f"Playing local: {track.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error playing local: {e}")
            return False
    
    def play_next(self) -> bool:
        """Play next track."""
        if self.use_streaming and self.streaming_playlist:
            return self.queue_streaming_track()
        elif self.local_playlist:
            return self.play_next_local()
        
        return False
    
    # ======== PUBLIC METHODS ========
    
    def stop(self) -> None:
        """Stop music."""
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            logger.error(f"Error stopping: {e}")
    
    def pause(self) -> None:
        """Pause music."""
        try:
            pygame.mixer.music.pause()
        except Exception as e:
            logger.error(f"Error pausing: {e}")
    
    def unpause(self) -> None:
        """Resume music."""
        try:
            pygame.mixer.music.unpause()
        except Exception as e:
            logger.error(f"Error unpausing: {e}")
    
    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        try:
            pygame.mixer.music.set_volume(self.volume)
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
    
    def get_volume(self) -> float:
        """Get current volume."""
        return self.volume
    
    def is_playing(self) -> bool:
        """Check if music is playing."""
        return pygame.mixer.music.get_busy()
    
    def get_current_track(self) -> Optional[str]:
        """Get current track name."""
        return self.current_track
    
    # FIXED: Changed from attribute to methods
    def is_downloading(self) -> bool:
        """Check if downloading."""
        return self._is_downloading
    
    def get_download_progress(self) -> int:
        """Get download progress (0-100)."""
        return self._download_progress
    
    def toggle_streaming(self) -> bool:
        """Toggle streaming mode."""
        self.use_streaming = not self.use_streaming
        status = "ON" if self.use_streaming else "OFF"
        logger.info(f"Streaming mode: {status}")
        return self.use_streaming
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.running = False
        
        # Signal threads to stop
        self.download_queue.put(None)
        self.playback_queue.put(None)
        
        # Wait for threads
        if self.download_thread:
            self.download_thread.join(timeout=1.0)
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        # Stop pygame
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            logger.info("Music player cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
