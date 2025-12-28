"""
Utility functions for Face Mood Music Player.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(config) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, config.get('logging.level', 'INFO').upper())
    log_file = config.get('logging.file', 'facemood.log')
    console_logging = config.get('logging.console', True)
    
    # Create logger
    logger = logging.getLogger('facemood')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_supported_audio_files(directory: Path, extensions: list) -> list:
    """
    Get list of supported audio files in a directory.
    
    Args:
        directory: Directory path
        extensions: List of supported file extensions
        
    Returns:
        List of audio file paths
    """
    if not directory.exists():
        return []
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory.glob(f'*{ext}'))
        audio_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(audio_files)


def normalize_emotion(emotion: str) -> str:
    """
    Normalize emotion string to lowercase.
    
    Args:
        emotion: Emotion string
        
    Returns:
        Normalized emotion string
    """
    return emotion.lower().strip()

