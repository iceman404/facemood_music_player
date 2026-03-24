"""GUI entry — `python -m facemood.gui`."""

from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def run_gui() -> None:
    from PyQt6.QtWidgets import QApplication

    from facemood.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
