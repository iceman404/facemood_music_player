"""Logging and filesystem helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List


def setup_logging(config) -> logging.Logger:
    log_level = getattr(logging, config.get("logging.level", "INFO").upper())
    log_file = config.get("logging.file", "facemood.log")
    console_logging = config.get("logging.console", True)

    logger = logging.getLogger("facemood")
    logger.setLevel(log_level)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    if console_logging:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(ch)

    return logger


def get_supported_audio_files(directory: Path, extensions: List[str]) -> List[Path]:
    if not directory.exists():
        return []
    out: List[Path] = []
    for ext in extensions:
        out.extend(directory.glob(f"*{ext}"))
        out.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(out)


def normalize_emotion(emotion: str) -> str:
    return emotion.lower().strip()
