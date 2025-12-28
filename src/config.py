"""
Configuration management module for Face Mood Music Player.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the application."""
    
    DEFAULT_CONFIG = {
        "camera": {
            "device_id": 0,
            "width": 640,
            "height": 480
        },
        "face_detection": {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        },
        "emotion": {
            "threshold_count": 40,
            "update_interval": 1.0,
            "sad": {
                "mouth_threshold": 0.1,
                "brow_inner_threshold": 0.05,
                "brow_outer_threshold": 0.07,
                "eye_openness_threshold": 0.15
            },
            "happy": {
                "mouth_distance_threshold": 0.45,
                "mouth_height_threshold": 0.3
            },
            "surprised": {
                "mouth_height_threshold": 0.15,
                "mouth_width_threshold": 0.25
            },
            "angry": {
                "brow_position_threshold": 0.6
            }
        },
        "music": {
            "base_path": "music",
            "volume": 0.7,
            "fade_duration": 1.0,
            "supported_formats": [".mp3", ".wav", ".ogg", ".flac"]
        },
        "display": {
            "show_landmarks": True,
            "landmark_color": [0, 255, 0],
            "landmark_size": 2,
            "font_scale": 1.0,
            "font_color": [0, 255, 0],
            "font_thickness": 2
        },
        "logging": {
            "level": "INFO",
            "file": "facemood.log",
            "console": True
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    self._merge_config(self.config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}. Using defaults.")
        else:
            self.save_config()
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'music.volume')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'music.volume')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def get_music_path(self, emotion: str) -> Path:
        """
        Get music directory path for an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Path to emotion's music directory
        """
        base_path = Path(self.get('music.base_path', 'music'))
        return base_path / emotion.lower()

