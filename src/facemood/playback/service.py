"""
Playback routing: local files, yt-dlp stream, browser watch URL, or in-app audio (yt-dlp | ffplay).

YouTube watch: yt-dlp resolves the top search hit → watch URL + autoplay=1.

SoundCloud: yt-dlp scsearch1 → first hit webpage URL (actual track/playlist page).

Spotify: optional Web API (music.spotify_client_id / spotify_client_secret) → track URL; else
Songlink/Odesli from the first YouTube search hit → Spotify when available; else search page.

Each mood change stops any in-app yt-dlp→ffplay stream first. External browser uses webbrowser only
(no extra hidden Chromium). For one controllable stream without tabs, use youtube_browser_mode "audio"
or music.source youtube_stream.
"""

from __future__ import annotations

import base64
import logging
import random
import shutil
import subprocess
import sys
import urllib.parse
import webbrowser
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("facemood.playback")

DEFAULT_QUERIES: Dict[str, List[str]] = {
    "happy": ["happy upbeat pop music", "feel good songs 2024"],
    "sad": ["sad emotional piano music", "melancholic songs"],
    "surprised": ["epic cinematic soundtrack", "orchestral trailer music"],
    "angry": ["intense rock workout music", "metal energy"],
    "neutral": ["lofi hip hop beats", "ambient focus music"],
}


class PlaybackService:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._player: Optional[Any] = None
        self._last_track: Optional[str] = None
        self._queries: Dict[str, List[str]] = dict(DEFAULT_QUERIES)
        custom = config.get("music.emotion_queries", None)
        if isinstance(custom, dict):
            for k, v in custom.items():
                if isinstance(v, list) and v:
                    self._queries[str(k).lower()] = [str(x) for x in v]

        self._yt_audio_pipe: Optional[Tuple[subprocess.Popen, subprocess.Popen]] = None

    def source_id(self) -> str:
        return str(self.config.get("music.source", "youtube_browser"))

    def source_tag(self) -> str:
        if self.source_id() == "youtube_browser":
            mode = self.config.get("music.youtube_browser_mode", "watch")
            if mode == "audio":
                return "YT·AUDIO"
            if mode == "search":
                return "YT·FIND"
            return "YT·WEB"
        m = {
            "local": "LOCAL",
            "youtube_stream": "YT·STREAM",
            "spotify_browser": "SPOTIFY",
            "soundcloud_browser": "SOUNDCLOUD",
        }
        return m.get(self.source_id(), self.source_id().upper()[:12])

    def _pick_query(self, emotion: str) -> str:
        opts = self._queries.get(emotion.lower(), self._queries["neutral"])
        return random.choice(opts)

    @staticmethod
    def _yt_dlp_available() -> bool:
        if shutil.which("yt-dlp"):
            return True
        try:
            import yt_dlp  # noqa: F401

            return True
        except ImportError:
            return False

    def _ytdlp_argv(self) -> Optional[List[str]]:
        exe = shutil.which("yt-dlp")
        if exe:
            return [exe]
        try:
            import yt_dlp  # noqa: F401

            return [sys.executable, "-m", "yt_dlp"]
        except ImportError:
            return None

    def _ensure_player(self) -> Any:
        if self._player is None:
            from facemood.audio import MusicPlayer

            self._player = MusicPlayer(self.config)
        return self._player

    def _yt_search_first_video_id(self, query: str) -> Optional[str]:
        try:
            import yt_dlp

            opts: Dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "default_search": "ytsearch1",
                "noplaylist": True,
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if not info or not isinstance(info, dict):
                return None
            for e in info.get("entries") or []:
                if isinstance(e, dict):
                    vid = e.get("id")
                    if vid and len(str(vid)) == 11:
                        return str(vid)
            iid = info.get("id")
            if iid and len(str(iid)) == 11:
                return str(iid)
            return None
        except Exception as e:
            logger.warning("yt-dlp ytsearch1 resolve failed: %s", e)
            return None

    def _soundcloud_first_track_url(self, query: str) -> Optional[str]:
        """First SoundCloud search hit (track/set page) via yt-dlp scsearch1."""
        try:
            import yt_dlp

            opts: Dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"scsearch1:{query}", download=False)
            if not info or not isinstance(info, dict):
                return None
            # scsearch1 top-level webpage_url is the pseudo-URL "scsearch1:…", not a browser link.
            for e in info.get("entries") or []:
                if isinstance(e, dict):
                    u = e.get("webpage_url")
                    if u and str(u).startswith("http"):
                        return str(u)
            u = info.get("webpage_url")
            if u and str(u).startswith("http"):
                return str(u)
            return None
        except Exception as e:
            logger.warning("yt-dlp scsearch1 resolve failed: %s", e)
            return None

    def _spotify_client_credentials_token(self) -> Optional[str]:
        cid = self.config.get("music.spotify_client_id")
        secret = self.config.get("music.spotify_client_secret")
        if not cid or not secret:
            return None
        try:
            auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
            r = requests.post(
                "https://accounts.spotify.com/api/token",
                data={"grant_type": "client_credentials"},
                headers={"Authorization": f"Basic {auth}"},
                timeout=15,
            )
            if not r.ok:
                logger.warning("Spotify token request failed: %s", r.status_code)
                return None
            return r.json().get("access_token")
        except Exception as e:
            logger.warning("Spotify auth: %s", e)
            return None

    def _spotify_api_track_url(self, query: str) -> Optional[str]:
        token = self._spotify_client_credentials_token()
        if not token:
            return None
        try:
            r = requests.get(
                "https://api.spotify.com/v1/search",
                params={"q": query, "type": "track", "limit": 1},
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            if not r.ok:
                return None
            items = (r.json().get("tracks") or {}).get("items") or []
            if not items:
                return None
            return items[0].get("external_urls", {}).get("spotify")
        except Exception as e:
            logger.warning("Spotify search API: %s", e)
            return None

    def _odesli_links_for_url(self, url: str) -> Dict[str, Any]:
        try:
            r = requests.get(
                "https://api.song.link/v1-alpha.1/links",
                params={"url": url},
                timeout=15,
            )
            if not r.ok:
                return {}
            return r.json().get("linksByPlatform") or {}
        except Exception as e:
            logger.debug("Songlink/Odesli: %s", e)
            return {}

    def _odesli_spotify_from_youtube(self, youtube_url: str) -> Optional[str]:
        p = self._odesli_links_for_url(youtube_url)
        sp = p.get("spotify") or p.get("spotifyMusic")
        if isinstance(sp, dict):
            return sp.get("url")
        return None

    def _spotify_resolved_track_url(self, query: str) -> Optional[str]:
        u = self._spotify_api_track_url(query)
        if u:
            return u
        vid = self._yt_search_first_video_id(query)
        if not vid:
            return None
        return self._odesli_spotify_from_youtube(f"https://www.youtube.com/watch?v={vid}")

    def _browser_new_arg(self) -> int:
        """webbrowser.open `new`: 0 = same window when possible; 2 = new tab."""
        if self.config.get("music.browser_open_new_tab", False):
            return 2
        return 0

    def _prepare_before_play(self, src: str) -> None:
        """Stop in-app audio so a new mood does not stack on yt-dlp|ffplay or stale pygame."""
        self._stop_youtube_audio_pipe()
        browser_sources = {"youtube_browser", "spotify_browser", "soundcloud_browser"}
        if src in browser_sources and self._player is not None:
            self._player.stop()

    def _stop_youtube_audio_pipe(self) -> None:
        if self._yt_audio_pipe is None:
            return
        p_out, p_in = self._yt_audio_pipe
        for p in (p_out, p_in):
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=2.0)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        self._yt_audio_pipe = None

    def _youtube_audio_pipe_play(self, emotion: str) -> bool:
        """Stream best audio: yt-dlp stdout → ffplay/mpv stdin (no browser)."""
        self._stop_youtube_audio_pipe()
        yargv = self._ytdlp_argv()
        if not yargv:
            logger.warning("yt-dlp not available — pip install yt-dlp")
            return False

        ffplay = shutil.which("ffplay")
        mpv = shutil.which("mpv")
        if not ffplay and not mpv:
            logger.warning("Neither ffplay nor mpv found (install ffmpeg and/or mpv)")
            return False

        q = self._pick_query(emotion)
        url = f"ytsearch1:{q}"
        ytdlp_cmd = yargv + ["-f", "bestaudio/best", "--no-playlist", "-o", "-", url]

        if ffplay:
            p_ytdlp = subprocess.Popen(
                ytdlp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            p_ff = subprocess.Popen(
                [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "-"],
                stdin=p_ytdlp.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if p_ytdlp.stdout:
                p_ytdlp.stdout.close()
        else:
            p_ytdlp = subprocess.Popen(
                ytdlp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            p_ff = subprocess.Popen(
                [mpv, "--no-video", "--really-quiet", "--no-terminal", "-"],
                stdin=p_ytdlp.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if p_ytdlp.stdout:
                p_ytdlp.stdout.close()

        self._yt_audio_pipe = (p_ytdlp, p_ff)
        self._last_track = f"YouTube audio · {q[:48]}"
        logger.info("Started in-app YouTube audio stream for query: %s", q[:60])
        return True

    def _open_youtube_search_fallback(self, emotion: str) -> bool:
        q = self._pick_query(emotion)
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q)
        return self._open_url(url, f"YouTube search · {emotion}")

    def _youtube_browser(self, emotion: str) -> bool:
        mode = str(self.config.get("music.youtube_browser_mode", "watch")).lower()
        q = self._pick_query(emotion)

        if mode == "audio":
            if self._youtube_audio_pipe_play(emotion):
                return True
            logger.info("Falling back from audio mode to watch or search")
            mode = "watch"

        if mode == "search":
            return self._open_youtube_search_fallback(emotion)

        # watch (default): first search result watch URL + autoplay
        if self._yt_dlp_available():
            vid = self._yt_search_first_video_id(q)
            if vid:
                watch = f"https://www.youtube.com/watch?v={vid}&autoplay=1"
                logger.info("Opening YouTube watch page (autoplay requested): %s", vid)
                return self._open_url(watch, f"YouTube watch · {vid}")

        logger.info("yt-dlp unavailable — opening search results instead (pip install yt-dlp for watch/autoplay)")
        return self._open_youtube_search_fallback(emotion)

    def _spotify_browser(self, emotion: str) -> bool:
        q = self._pick_query(emotion)
        url: Optional[str] = None
        label = f"Spotify search · {emotion}"
        # API works without yt-dlp; Odesli path needs yt-dlp for YouTube bridge.
        url = self._spotify_resolved_track_url(q)
        if url:
            label = f"Spotify track · {emotion}"
            logger.info("Opening Spotify track URL")
        else:
            url = "https://open.spotify.com/search/" + urllib.parse.quote(q, safe="")
            logger.info("Spotify: no track resolved — opening search (set spotify_client_id/secret for API)")
        return self._open_url(url, label)

    def _soundcloud_browser(self, emotion: str) -> bool:
        q = self._pick_query(emotion)
        label = f"SoundCloud search · {emotion}"
        url: Optional[str] = None
        if self._yt_dlp_available():
            url = self._soundcloud_first_track_url(q)
        if url:
            label = f"SoundCloud · {emotion}"
            logger.info("Opening SoundCloud page from search hit")
        else:
            url = "https://soundcloud.com/search?q=" + urllib.parse.quote_plus(q)
            logger.info("SoundCloud: scsearch1 failed — opening search page")
        return self._open_url(url, label)

    def play_emotion(self, emotion: str) -> bool:
        src = self.source_id()
        self._prepare_before_play(src)
        try:
            if src == "youtube_browser":
                return self._youtube_browser(emotion)
            if src == "spotify_browser":
                return self._spotify_browser(emotion)
            if src == "soundcloud_browser":
                return self._soundcloud_browser(emotion)
            if src == "youtube_stream":
                p = self._ensure_player()
                p.use_streaming = bool(self.config.get("music.prefer_streaming", True))
                ok = p.play_emotion(emotion)
                self._last_track = p.get_current_track()
                return ok
            if src == "local":
                p = self._ensure_player()
                p.use_streaming = False
                ok = p.play_emotion(emotion)
                self._last_track = p.get_current_track()
                return ok
            logger.warning("Unknown music.source=%r — YouTube search fallback", src)
            return self._open_youtube_search_fallback(emotion)
        except Exception as e:
            logger.error("play_emotion: %s", e)
            return False

    def _open_url(self, url: str, label: str, *, new: Optional[int] = None) -> bool:
        logger.info("Opening browser: %s", url[:100])
        self._last_track = label
        n = self._browser_new_arg() if new is None else new
        webbrowser.open(url, new=n)
        return True

    def get_current_track(self) -> Optional[str]:
        if self._player is not None:
            t = self._player.get_current_track()
            if t:
                return t
        return self._last_track

    def is_downloading(self) -> bool:
        if self._player is None:
            return False
        return bool(self._player.is_downloading())

    def get_download_progress(self) -> int:
        if self._player is None:
            return 0
        return int(self._player.get_download_progress())

    def get_volume(self) -> float:
        if self._player is None:
            return float(self.config.get("music.volume", 0.7))
        return float(self._player.get_volume())

    def set_volume(self, v: float) -> None:
        if self._player is not None:
            self._player.set_volume(v)

    def is_playing(self) -> bool:
        if self._yt_audio_pipe is not None:
            _, p2 = self._yt_audio_pipe
            return p2.poll() is None
        if self._player is not None:
            return bool(self._player.is_playing())
        return False

    def pause(self) -> None:
        if self._player is not None:
            self._player.pause()

    def unpause(self) -> None:
        if self._player is not None:
            self._player.unpause()

    def stop(self) -> None:
        self._stop_youtube_audio_pipe()
        if self._player is not None:
            self._player.stop()

    def toggle_streaming(self) -> bool:
        if self._player is None:
            return False
        return bool(self._player.toggle_streaming())

    def use_streaming_flag(self) -> bool:
        if self._player is None:
            return self.source_id() == "youtube_stream"
        return bool(self._player.use_streaming)

    def play_emotion_repeat(self, emotion: Optional[str]) -> bool:
        if not emotion:
            return False
        return self.play_emotion(emotion)

    @property
    def current_emotion(self) -> Optional[str]:
        if self._player is not None:
            return getattr(self._player, "current_emotion", None)
        return None

    def cleanup(self) -> None:
        self._stop_youtube_audio_pipe()
        if self._player is not None:
            self._player.cleanup()
            self._player = None
