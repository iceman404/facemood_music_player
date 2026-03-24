"""Session statistics (optional UX — same behavior as v1)."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("facemood.telemetry")


class SessionStatistics:
    def __init__(self) -> None:
        self.emotion_counts: Dict[str, int] = defaultdict(int)
        self.total_detections = 0
        self.music_plays: Dict[str, int] = defaultdict(int)
        self.session_start = datetime.now()

    def record_emotion(self, emotion: str) -> None:
        self.emotion_counts[emotion] += 1
        self.total_detections += 1

    def record_music_play(self, emotion: str) -> None:
        self.music_plays[emotion] += 1
        logger.info("Music queued for %s (total plays: %s)", emotion, self.music_plays[emotion])

    def print_summary(self) -> None:
        summary = self._summary()
        print("\n" + "=" * 50)
        print("FACEMOOD SESSION")
        print("=" * 50)
        print(f"Frames labeled: {summary['total_detections']}")
        print(f"Duration (s): {summary['session_duration_seconds']:.1f}")
        print("\nAffect distribution:")
        for emo, count in summary["emotion_counts"].items():
            pct = summary["emotion_percentages"].get(emo, 0)
            print(f"  {emo}: {count} ({pct:.1f}%)")
        print("\nPlayback triggers:")
        for emo, count in summary["music_plays"].items():
            print(f"  {emo}: {count}")
        print("=" * 50 + "\n")

    def _summary(self) -> Dict:
        session_duration = (datetime.now() - self.session_start).total_seconds()
        pct: Dict[str, float] = {}
        if self.total_detections > 0:
            for emo, c in self.emotion_counts.items():
                pct[emo] = (c / self.total_detections) * 100
        return {
            "total_detections": self.total_detections,
            "emotion_counts": dict(self.emotion_counts),
            "emotion_percentages": pct,
            "music_plays": dict(self.music_plays),
            "session_duration_seconds": session_duration,
        }
