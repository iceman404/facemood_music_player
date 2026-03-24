"""
CLI: OpenCV window + camera loop. Core logic lives in `facemood.session.FaceMoodSession`.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time

# OpenCV's bundled Qt often lacks Wayland plugins; xcb avoids noisy warnings on Linux.
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from facemood.config import Config
from facemood.session import FaceMoodSession
from facemood.utils import setup_logging

logger = logging.getLogger("facemood.app")


class FaceMoodApplication:
    def __init__(self, config_path: str = "config.json") -> None:
        self.config = Config(config_path)
        self.log = setup_logging(self.config)
        self.running = False
        self._cleaned_up = False

        self.session = FaceMoodSession(self.config)

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        self.log.info("Signal %s — shutting down", signum)
        self.stop()
        sys.exit(0)

    def run(self) -> None:
        self.running = True
        dev = self.config.get("camera.device_id", 0)
        w = self.config.get("camera.width", 1280)
        h = self.config.get("camera.height", 720)

        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            self.log.error("Cannot open camera %s", dev)
            self.stop()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self.log.info("FaceMood CLI running (music.source=%s)", self.session.playback.source_id())
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = self.session.process_frame(frame)
                cv2.imshow("FaceMood // Affect pipeline", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    self.session.stats.print_summary()
                elif key == ord("p"):
                    if self.session.playback.is_playing():
                        self.session.playback.pause()
                    else:
                        self.session.playback.unpause()
                elif key == ord("t"):
                    self.session.playback.toggle_streaming()
                elif key in (ord("+"), ord("=")):
                    self.session.playback.set_volume(min(1.0, self.session.playback.get_volume() + 0.1))
                elif key == ord("-"):
                    self.session.playback.set_volume(max(0.0, self.session.playback.get_volume() - 0.1))
                elif key == ord("n"):
                    self.session.playback.stop()
                    self.session.policy.forget_last_played()
                    ls = self.session.smoother.last_smoothed
                    if ls and ls.label != "neutral":
                        self.session.playback.play_emotion(ls.label)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop()

    def stop(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self.running = False
        self.session.stats.print_summary()
        self.session.close()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="FaceMood — layered affect-aware music (CLI)")
    p.add_argument("--config", default="config.json", help="JSON config path")
    p.add_argument("--music-path", default=None, help="Override music.base_path")
    p.add_argument("--no-streaming", action="store_true", help="Disable yt-dlp streaming (local/youtube_stream)")
    args = p.parse_args()

    app = FaceMoodApplication(args.config)
    if args.music_path:
        app.config.set("music.base_path", args.music_path)
    if args.no_streaming:
        app.config.set("music.prefer_streaming", False)

    try:
        app.run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
