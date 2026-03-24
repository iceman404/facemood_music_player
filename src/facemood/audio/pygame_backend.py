"""Audio: pygame.mixer when available; else ffplay/mpv (typical when Pygame is built without SDL2_mixer)."""

from __future__ import annotations

import logging
import os
import queue
import random
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import pygame

from facemood.utils import get_supported_audio_files

logger = logging.getLogger("facemood.audio")


class MusicPlayer:
    """Music player with local files and optional yt-dlp streaming."""

    def __init__(self, config) -> None:
        self.config = config
        self.current_emotion: Optional[str] = None
        self.current_track: Optional[str] = None
        self.temp_file: Optional[str] = None
        self.local_playlist: List[Path] = []
        self.streaming_playlist: List[Dict] = []
        self.use_streaming = True
        self.volume = config.get("music.volume", 0.7)
        self.fade_duration = config.get("music.fade_duration", 1.0)
        self.supported_formats = config.get("music.supported_formats", [".mp3", ".wav", ".ogg", ".flac"])

        self.download_queue: queue.Queue = queue.Queue()
        self.playback_queue: queue.Queue = queue.Queue()
        self.running = True
        self._is_downloading = False
        self._download_progress = 0

        self.emotion_queries = {
            "happy": ["happy music", "upbeat pop", "feel good songs"],
            "sad": ["sad songs", "emotional music", "melancholic"],
            "surprised": ["epic music", "cinematic soundtrack", "orchestral"],
            "angry": ["rock music", "metal songs", "intense music"],
            "neutral": ["lofi beats", "ambient music", "background music"],
        }

        self.yt_dlp_available = self._check_yt_dlp()

        # pygame.mixer often missing on Pygame built from source without sdl2_mixer (e.g. Python 3.14)
        self._audio_backend = "pygame"
        self._external_bin: Optional[str] = None
        self._playproc: Optional[subprocess.Popen] = None
        self._play_lock = threading.Lock()

        self._init_audio_engine()

        logger.info(
            "Music player: backend=%s streaming=%s",
            self._audio_backend,
            self.yt_dlp_available,
        )

        self._start_threads()

    def _init_audio_engine(self) -> None:
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            pygame.mixer.music.set_volume(self.volume)
            self._audio_backend = "pygame"
            return
        except (NotImplementedError, ModuleNotFoundError, RuntimeError) as e:
            logger.warning("pygame.mixer unavailable (%s) — trying ffplay/mpv", e)

        for cmd in ("ffplay", "mpv"):
            path = shutil.which(cmd)
            if path:
                self._audio_backend = "external"
                self._external_bin = path
                logger.info("Using external player: %s (install sdl2_mixer + reinstall pygame to use pygame.mixer)", path)
                return

        raise RuntimeError(
            "No audio output available: pygame.mixer is missing (common on Python 3.14 when Pygame "
            "builds without SDL2_mixer) and neither ffplay nor mpv was found.\n\n"
            "Fix options:\n"
            "  • sudo pacman -S sdl2_mixer ffmpeg  &&  pip install --force-reinstall --no-cache-dir pygame\n"
            "  • or: python3.12 -m venv .venv && pip install -r requirements.txt\n"
            "  • or: install ffmpeg (provides ffplay) so this fallback can play audio."
        ) from e

    def _check_yt_dlp(self) -> bool:
        try:
            import yt_dlp  # noqa: F401

            return True
        except ImportError:
            logger.warning("yt-dlp not installed — streaming disabled.")
            return False

    def _start_threads(self) -> None:
        self.download_thread = threading.Thread(target=self._download_worker, daemon=True, name="DownloadWorker")
        self.download_thread.start()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True, name="PlaybackWorker")
        self.playback_thread.start()

    def _stop_external(self) -> None:
        with self._play_lock:
            if self._playproc is not None and self._playproc.poll() is None:
                self._playproc.terminate()
                try:
                    self._playproc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._playproc.kill()
            self._playproc = None

    def _music_busy(self) -> bool:
        if self._audio_backend == "pygame":
            return bool(pygame.mixer.music.get_busy())
        with self._play_lock:
            return self._playproc is not None and self._playproc.poll() is None

    def _play_audio_file(self, filepath: str) -> bool:
        if self._audio_backend == "pygame":
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.fadeout(int(self.fade_duration * 1000))
                    time.sleep(self.fade_duration)
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                return True
            except Exception as e:
                logger.error("Play file: %s", e)
                return False

        return self._play_external_file(filepath)

    def _play_external_file(self, filepath: str) -> bool:
        assert self._external_bin is not None
        self._stop_external()
        vol = max(0, min(100, int(self.volume * 100)))
        try:
            if "ffplay" in self._external_bin:
                args = [
                    self._external_bin,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-volume",
                    str(vol),
                    filepath,
                ]
            else:
                # mpv
                args = [
                    self._external_bin,
                    "--no-video",
                    "--really-quiet",
                    f"--volume={vol}",
                    filepath,
                ]
            with self._play_lock:
                self._playproc = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return True
        except Exception as e:
            logger.error("External play: %s", e)
            return False

    def _download_worker(self) -> None:
        while self.running:
            try:
                task = self.download_queue.get(timeout=0.5)
                if task is None:
                    break
                emotion, video_id, title = task
                logger.info("Downloading: %s...", title[:40])
                self._is_downloading = True
                self._download_progress = 10
                temp_file = self._download_audio(video_id, title)
                self._download_progress = 90
                if temp_file and os.path.exists(temp_file):
                    self.playback_queue.put((emotion, temp_file, title))
                else:
                    logger.error("Download failed: %s", title[:40])
                self._is_downloading = False
                self._download_progress = 0
                self.download_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Download worker: %s", e)
                self._is_downloading = False
                self._download_progress = 0

    def _playback_worker(self) -> None:
        while self.running:
            try:
                task = self.playback_queue.get(timeout=0.5)
                if task is None:
                    break
                emotion, temp_file, title = task
                success = self._play_audio_file(temp_file)
                if success:
                    self.current_emotion = emotion
                    self.current_track = title
                    self.temp_file = temp_file
                else:
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
                self.playback_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Playback worker: %s", e)

    def _download_audio(self, video_id: str, title: str) -> Optional[str]:
        if not self.yt_dlp_available:
            return None
        try:
            import yt_dlp

            url = f"https://www.youtube.com/watch?v={video_id}"
            temp_dir = tempfile.gettempdir()
            safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()[:30]
            temp_file = os.path.join(temp_dir, f"facemood_{video_id}_{safe_title}.mp3")
            if os.path.exists(temp_file):
                return temp_file
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(temp_dir, f"facemood_{video_id}"),
                "quiet": False,
                "no_warnings": False,
                "progress_hooks": [self._download_progress_hook],
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self._download_progress = 20
                ydl.download([url])
                self._download_progress = 80
            downloaded_file = os.path.join(temp_dir, f"facemood_{video_id}.mp3")
            if os.path.exists(downloaded_file):
                os.rename(downloaded_file, temp_file)
                return temp_file
            return None
        except Exception as e:
            logger.error("Download error: %s", e)
            return None

    def _download_progress_hook(self, d: Dict) -> None:
        if d.get("status") == "downloading":
            try:
                downloaded = d.get("downloaded_bytes", 0)
                total = d.get("total_bytes", 0) or d.get("total_bytes_estimate", 0)
                if total > 0:
                    self._download_progress = 20 + int((downloaded / total) * 60)
            except Exception:
                pass

    def search_youtube(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.yt_dlp_available:
            return []
        try:
            import yt_dlp

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "default_search": f"ytsearch{limit}",
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(query, download=False)
                if not result or "entries" not in result:
                    return []
                tracks = []
                for entry in result["entries"][:limit]:
                    if entry and entry.get("duration", 0) <= 300:
                        tracks.append(
                            {
                                "id": entry.get("id"),
                                "title": entry.get("title", "Unknown"),
                                "artist": entry.get("uploader", "Unknown"),
                                "duration": entry.get("duration", 0),
                                "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                            }
                        )
                return tracks
        except Exception as e:
            logger.error("Search: %s", e)
            return []

    def load_streaming_playlist(self, emotion: str) -> bool:
        if not self.use_streaming or not self.yt_dlp_available:
            return False
        try:
            queries = self.emotion_queries.get(emotion, ["music"])
            query = random.choice(queries)
            self.streaming_playlist = self.search_youtube(query + " music", limit=3)
            return bool(self.streaming_playlist)
        except Exception as e:
            logger.error("Streaming: %s", e)
            return False

    def load_local_playlist(self, emotion: str) -> bool:
        try:
            music_path = self.config.get_music_path(emotion)
            self.local_playlist = get_supported_audio_files(music_path, self.supported_formats)
            return bool(self.local_playlist)
        except Exception as e:
            logger.error("Local: %s", e)
            return False

    def play_emotion(self, emotion: str) -> bool:
        if emotion == self.current_emotion:
            if not self._music_busy():
                return self.play_next()
            return True
        self.current_emotion = emotion
        if self.use_streaming and self.load_streaming_playlist(emotion):
            return self.queue_streaming_track()
        if self.load_local_playlist(emotion):
            return self.play_next_local()
        return False

    def queue_streaming_track(self) -> bool:
        if not self.streaming_playlist:
            return False
        try:
            track = random.choice(self.streaming_playlist)
            self.download_queue.put((self.current_emotion, track["id"], track["title"]))
            return True
        except Exception as e:
            logger.error("Queue: %s", e)
            return False

    def play_next_local(self) -> bool:
        if not self.local_playlist:
            return False
        try:
            track = random.choice(self.local_playlist)
            self.current_track = track.name
            if self._audio_backend == "pygame":
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.fadeout(int(self.fade_duration * 1000))
                    time.sleep(self.fade_duration)
                pygame.mixer.music.load(str(track))
                pygame.mixer.music.play()
                return True
            return self._play_external_file(str(track))
        except Exception as e:
            logger.error("Local play: %s", e)
            return False

    def play_next(self) -> bool:
        if self.use_streaming and self.streaming_playlist:
            return self.queue_streaming_track()
        if self.local_playlist:
            return self.play_next_local()
        return False

    def stop(self) -> None:
        try:
            if self._audio_backend == "pygame":
                pygame.mixer.music.stop()
            else:
                self._stop_external()
        except Exception as e:
            logger.error("stop: %s", e)

    def pause(self) -> None:
        try:
            if self._audio_backend == "pygame":
                pygame.mixer.music.pause()
            else:
                with self._play_lock:
                    if self._playproc and self._playproc.poll() is None and hasattr(signal, "SIGSTOP"):
                        self._playproc.send_signal(signal.SIGSTOP)
        except Exception as e:
            logger.debug("pause: %s", e)

    def unpause(self) -> None:
        try:
            if self._audio_backend == "pygame":
                pygame.mixer.music.unpause()
            else:
                with self._play_lock:
                    if self._playproc and self._playproc.poll() is None and hasattr(signal, "SIGCONT"):
                        self._playproc.send_signal(signal.SIGCONT)
        except Exception as e:
            logger.debug("unpause: %s", e)

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))
        try:
            if self._audio_backend == "pygame":
                pygame.mixer.music.set_volume(self.volume)
        except Exception as e:
            logger.error("volume: %s", e)

    def get_volume(self) -> float:
        return self.volume

    def is_playing(self) -> bool:
        return self._music_busy()

    def get_current_track(self) -> Optional[str]:
        return self.current_track

    def is_downloading(self) -> bool:
        return self._is_downloading

    def get_download_progress(self) -> int:
        return self._download_progress

    def toggle_streaming(self) -> bool:
        self.use_streaming = not self.use_streaming
        return self.use_streaming

    def cleanup(self) -> None:
        self.running = False
        self.download_queue.put(None)
        self.playback_queue.put(None)
        if self.download_thread:
            self.download_thread.join(timeout=1.0)
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        try:
            self._stop_external()
            if self._audio_backend == "pygame":
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception as e:
            logger.error("cleanup: %s", e)
