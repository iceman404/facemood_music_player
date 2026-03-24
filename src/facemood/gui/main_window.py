"""Main window: settings + live camera tab."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from facemood.config import Config
from facemood.session import FaceMoodSession

SOURCES = [
    ("youtube_browser", "YouTube (browser tab — no API)"),
    ("spotify_browser", "Spotify (track URL — API or Songlink)"),
    ("soundcloud_browser", "SoundCloud (first search hit)"),
    ("local", "Local folders (music/<emotion>/)"),
    ("youtube_stream", "YouTube audio (yt-dlp + pygame/ffplay)"),
]


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, session: FaceMoodSession, device_id: int, width: int, height: int) -> None:
        super().__init__()
        self.session = session
        self.device_id = device_id
        self.width = width
        self.height = height
        self._run = False
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            self.frame_ready.emit(None)
            return
        self._run = True
        while self._run and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                break
            frame = self.session.process_frame(frame)
            self.frame_ready.emit(frame)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def stop_capture(self) -> None:
        self._run = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.wait(4000)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FaceMood — Control")
        self.resize(1040, 720)

        self._config_path = Path("config.json")
        self.config = Config(str(self._config_path))
        self.config.set("display.footer_hint", "GUI — use Settings tab for playback source")
        self.session: FaceMoodSession | None = None

        self._thread: CameraThread | None = None

        self._build_ui()
        self._apply_dark_style()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "Live")
        tabs.addTab(self._build_settings_tab(), "Playback & camera")
        tabs.addTab(self._build_about_tab(), "About")

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)

    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #0a0e14; color: #5a6570;")
        self.video_label.setText("Camera stopped — press Start")
        lay.addWidget(self.video_label)

        row = QHBoxLayout()
        self.btn_start = QPushButton("Start camera")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._start_camera)
        self.btn_stop.clicked.connect(self._stop_camera)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        lay.addLayout(row)

        return w

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)

        g = QGroupBox("Music source")
        form = QFormLayout(g)
        self.combo_source = QComboBox()
        for sid, label in SOURCES:
            self.combo_source.addItem(label, sid)
        self._select_combo_source(self.config.get("music.source", "youtube_browser"))
        form.addRow("Playback:", self.combo_source)

        self.combo_yt_mode = QComboBox()
        self.combo_yt_mode.addItem("YouTube: watch page + autoplay (needs yt-dlp)", "watch")
        self.combo_yt_mode.addItem("YouTube: play audio in-app (yt-dlp → ffplay/mpv)", "audio")
        self.combo_yt_mode.addItem("YouTube: search results page only", "search")
        ym = str(self.config.get("music.youtube_browser_mode", "watch"))
        for i in range(self.combo_yt_mode.count()):
            if self.combo_yt_mode.itemData(i) == ym:
                self.combo_yt_mode.setCurrentIndex(i)
                break
        form.addRow("YouTube browser mode:", self.combo_yt_mode)

        self.edit_music_path = QLineEdit(str(self.config.get("music.base_path", "music")))
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_music)
        mp_row = QHBoxLayout()
        mp_row.addWidget(self.edit_music_path)
        mp_row.addWidget(btn_browse)
        form.addRow("Local library root:", mp_row)

        self.spin_device = QSpinBox()
        self.spin_device.setRange(0, 8)
        self.spin_device.setValue(int(self.config.get("camera.device_id", 0)))
        form.addRow("Camera index:", self.spin_device)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(320, 1920)
        self.spin_w.setValue(int(self.config.get("camera.width", 1280)))
        self.spin_h = QSpinBox()
        self.spin_h.setRange(240, 1080)
        self.spin_h.setValue(int(self.config.get("camera.height", 720)))
        res = QHBoxLayout()
        res.addWidget(QLabel("W"))
        res.addWidget(self.spin_w)
        res.addWidget(QLabel("H"))
        res.addWidget(self.spin_h)
        form.addRow("Resolution:", res)

        btn_save = QPushButton("Save to config.json")
        btn_save.clicked.connect(self._save_settings)
        form.addRow(btn_save)

        outer.addWidget(g)

        help_txt = QTextEdit()
        help_txt.setReadOnly(True)
        help_txt.setMaximumHeight(160)
        help_txt.setPlainText(
            "Local: subfolders happy, sad, surprised, angry, neutral under the library root.\n"
            "YouTube watch: pip install yt-dlp — opens the first search hit; autoplay may need one click in the browser.\n"
            "YouTube audio: yt-dlp + ffplay (ffmpeg) or mpv — plays sound without opening the browser.\n"
            "youtube_stream: pygame/ffplay download queue."
        )
        outer.addWidget(help_txt)

        return w

    def _select_combo_source(self, sid: str) -> None:
        for i in range(self.combo_source.count()):
            if self.combo_source.itemData(i) == sid:
                self.combo_source.setCurrentIndex(i)
                return

    def _browse_music(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Music library root", str(Path.home()))
        if d:
            self.edit_music_path.setText(d)

    def _save_settings(self) -> None:
        sid = self.combo_source.currentData()
        self.config.set("music.source", sid)
        self.config.set("music.youtube_browser_mode", self.combo_yt_mode.currentData())
        self.config.set("music.base_path", self.edit_music_path.text().strip())
        self.config.set("camera.device_id", int(self.spin_device.value()))
        self.config.set("camera.width", int(self.spin_w.value()))
        self.config.set("camera.height", int(self.spin_h.value()))
        self.config.save_config()
        QMessageBox.information(self, "Saved", "Settings written to config.json.\nRestart the camera to apply resolution/device.")

    def _build_about_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        t = QTextEdit()
        t.setReadOnly(True)
        t.setPlainText(
            "FaceMood — affect-aware playback\n\n"
            "• Perception: MediaPipe Face Landmarker + blendshapes\n"
            "• Temporal fusion + policy before any playback action\n"
            "• Playback: browser search (no API), local files, or yt-dlp stream\n\n"
            "CLI: python facemood_player.py\n"
            "GUI: python -m facemood.gui"
        )
        lay.addWidget(t)
        return w

    def _apply_dark_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: #121820; color: #e8eaed; font-size: 13px; }
            QGroupBox { font-weight: bold; border: 1px solid #2a3544; border-radius: 6px; margin-top: 12px; padding: 8px; }
            QPushButton { background-color: #1e3a5f; color: #a8d4ff; padding: 8px 16px; border-radius: 4px; border: 1px solid #2d5a8c; }
            QPushButton:hover { background-color: #2a4a70; }
            QPushButton:disabled { color: #555; background-color: #1a1f26; }
            QLineEdit, QSpinBox, QComboBox { background-color: #1a222d; border: 1px solid #2a3544; padding: 4px; border-radius: 3px; }
            QTextEdit { background-color: #0d1117; border: 1px solid #2a3544; }
            QTabWidget::pane { border: 1px solid #2a3544; border-radius: 4px; }
            QTabBar::tab { background: #1a222d; padding: 8px 16px; margin-right: 2px; }
            QTabBar::tab:selected { background: #1e3a5f; }
            """
        )

    def _start_camera(self) -> None:
        if self._thread is not None:
            return
        self.config = Config(str(self._config_path))
        self.config.set("display.footer_hint", "GUI — use Settings tab for playback source")
        if self.session is not None:
            self.session.close()
        self.session = FaceMoodSession(self.config)

        dev = int(self.spin_device.value())
        w = int(self.spin_w.value())
        h = int(self.spin_h.value())

        self._thread = CameraThread(self.session, dev, w, h)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop_camera(self) -> None:
        if self._thread is not None:
            self._thread.stop_capture()
            self._thread = None
        if self.session is not None:
            self.session.close()
            self.session = None
        self.video_label.clear()
        self.video_label.setText("Camera stopped")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_frame(self, frame: object) -> None:
        if frame is None:
            self.video_label.setText("Could not open camera")
            return
        arr = np.asarray(frame)
        if arr.ndim != 3:
            return
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def closeEvent(self, event) -> None:
        self._stop_camera()
        super().closeEvent(event)
