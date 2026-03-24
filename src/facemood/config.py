"""Configuration with defaults for perception, affect, policy, and audio."""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    DEFAULT_CONFIG: Dict[str, Any] = {
        "camera": {"device_id": 0, "width": 1280, "height": 720},
        "face_detection": {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "emotion": {
            "update_interval": 0.12,
            "sad": {
                "mouth_threshold": 0.1,
                "brow_inner_threshold": 0.05,
                "brow_outer_threshold": 0.07,
                "eye_openness_threshold": 0.15,
            },
            "happy": {
                "mouth_distance_threshold": 0.45,
                "mouth_height_threshold": 0.3,
            },
            "surprised": {
                "mouth_height_threshold": 0.15,
                "mouth_width_threshold": 0.25,
            },
            "angry": {"brow_position_threshold": 0.6},
        },
        "temporal": {
            "window": 7,
            "ema_alpha": 0.42,
            "min_confidence": 0.38,
        },
        "policy": {
            "stable_frames": 12,
            "cooldown_seconds": 22.0,
            "min_confidence": 0.42,
            "neutral_reset_frames": 8,
        },
        "music": {
            "source": "youtube_browser",
            "base_path": "music",
            "volume": 0.7,
            "fade_duration": 1.0,
            "supported_formats": [".mp3", ".wav", ".ogg", ".flac"],
            "prefer_streaming": True,
            "emotion_queries": {},
            # youtube_browser: "watch" = open first search hit with autoplay=1 (needs yt-dlp to resolve);
            # "audio" = play audio in-app via yt-dlp | ffplay (needs yt-dlp + ffmpeg ffplay);
            # "search" = only open YouTube search results page
            "youtube_browser_mode": "watch",
            # If false, webbrowser prefers same window (often one tab); if true, always new tab.
            "browser_open_new_tab": False,
            # Optional: Spotify Web API (Client Credentials) for direct track URLs instead of search-only.
            "spotify_client_id": None,
            "spotify_client_secret": None,
        },
        "display": {
            "show_landmarks": True,
            "landmark_color": [0, 255, 0],
            "landmark_size": 2,
            "font_scale": 1.0,
            "font_color": [0, 255, 0],
            "font_thickness": 2,
            "futuristic_hud": True,
        },
        "logging": {"level": "INFO", "file": "facemood.log", "console": True},
    }

    def __init__(self, config_path: str = "config.json") -> None:
        self.config_path = Path(config_path)
        self.config = self._deep_copy(self.DEFAULT_CONFIG)
        self.load_config()

    @staticmethod
    def _deep_copy(d: Dict) -> Dict:
        import copy

        return copy.deepcopy(d)

    def load_config(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                    self._merge_config(self.config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config: {e}. Using defaults.")
        else:
            self.save_config()

    def _merge_config(self, base: Dict, override: Dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def save_config(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        keys = key_path.split(".")
        cfg = self.config
        for key in keys[:-1]:
            if key not in cfg:
                cfg[key] = {}
            cfg = cfg[key]
        cfg[keys[-1]] = value

    def get_music_path(self, emotion: str) -> Path:
        base_path = Path(self.get("music.base_path", "music"))
        return base_path / emotion.lower()
