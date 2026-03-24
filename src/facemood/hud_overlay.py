"""
Futuristic HUD overlay — dark glass panels, neon accents, confidence meter.
Uses OpenCV + NumPy only (no extra deps).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# BGR — neon / sci-fi palette
_C = (255, 214, 0)  # primary cyan
_M = (255, 128, 255)  # magenta accent
_DIM = (158, 158, 158)
_WHITE = (248, 248, 248)
_PANEL = np.array([28, 22, 18], dtype=np.float32)  # dark blue-gray BGR


def _blend_roi(dst: np.ndarray, y0: int, y1: int, tint: np.ndarray, alpha: float) -> None:
    roi = dst[y0:y1, :, :]
    t = np.broadcast_to(tint, roi.shape).astype(np.float32)
    r = roi.astype(np.float32)
    blended = r * (1.0 - alpha) + t * alpha
    np.clip(blended, 0, 255, out=blended)
    roi[:, :, :] = blended.astype(np.uint8)


def _corner_bracket(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    arm: int = 14,
    th: int = 2,
) -> None:
    """Draw four corner L-brackets around rectangle (x,y)-(x+w,y+h)."""
    cv2.line(img, (x, y + arm), (x, y), color, th)
    cv2.line(img, (x, y), (x + arm, y), color, th)
    cv2.line(img, (x + w - arm, y), (x + w, y), color, th)
    cv2.line(img, (x + w, y + arm), (x + w, y), color, th)
    cv2.line(img, (x, y + h - arm), (x, y + h), color, th)
    cv2.line(img, (x, y + h), (x + arm, y + h), color, th)
    cv2.line(img, (x + w - arm, y + h), (x + w, y + h), color, th)
    cv2.line(img, (x + w, y + h), (x + w, y + h - arm), color, th)


def render_hud(
    frame: np.ndarray,
    *,
    label: str,
    confidence: float,
    raw_label: str,
    policy_reason: str,
    track: Optional[str],
    downloading: bool,
    download_pct: int,
    volume: float,
    source_tag: str,
    footer_hint: Optional[str] = None,
) -> None:
    """Draw HUD in-place on BGR frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    small = cv2.FONT_HERSHEY_COMPLEX_SMALL

    header_h = 40
    footer_h = 118

    _blend_roi(frame, 0, header_h, _PANEL, 0.72)
    cv2.line(frame, (0, header_h - 1), (w, header_h - 1), _C, 1)

    title = "FACEMOOD"
    sub = "AFFECT PIPELINE  v2"
    cv2.putText(frame, title, (16, 27), small, 0.85, _C, 1, cv2.LINE_AA)
    cv2.putText(frame, sub, (16, 38), font, 0.38, _DIM, 1, cv2.LINE_AA)

    tag = (source_tag or "—")[:14]
    tw = cv2.getTextSize(tag, font, 0.5, 1)[0][0]
    tag_col = _C if tag not in ("LOCAL", "—") else _DIM
    cv2.putText(frame, tag, (w - tw - 18, 26), font, 0.5, tag_col, 1, cv2.LINE_AA)

    vol_txt = f"VOL {int(volume * 100):3d}%"
    cv2.putText(frame, vol_txt, (w - 120, 38), font, 0.42, _WHITE, 1, cv2.LINE_AA)

    y0 = h - footer_h
    _blend_roi(frame, y0, h, _PANEL, 0.78)
    cv2.line(frame, (0, y0), (w, y0), _C, 1)
    _corner_bracket(frame, 10, y0 + 8, w - 20, footer_h - 16, (60, 80, 100), arm=12, th=1)

    y = y0 + 28
    lab = label.upper() if label else "—"
    cv2.putText(frame, "PRIMARY AFFECT", (22, y), font, 0.45, _DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, lab, (22, y + 22), font, 0.95, _C, 2, cv2.LINE_AA)

    conf_txt = f"CONFIDENCE  {confidence:5.1%}"
    cv2.putText(frame, conf_txt, (240, y + 18), font, 0.55, _WHITE, 1, cv2.LINE_AA)

    bx, by, bw, bh = 240, y + 26, 200, 8
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 45, 55), -1)
    fill = int(np.clip(confidence, 0.0, 1.0) * bw)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), _C, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (100, 110, 130), 1)

    raw_t = f"RAW  {raw_label.upper()}"
    cv2.putText(frame, raw_t, (470, y + 18), font, 0.48, _M, 1, cv2.LINE_AA)

    pol = (policy_reason or "")[:52]
    cv2.putText(frame, f"POLICY  {pol}", (22, y0 + 98), font, 0.42, _DIM, 1, cv2.LINE_AA)

    if track:
        t = track if len(track) <= 48 else track[:45] + "..."
        cv2.putText(frame, f"NOW / LAST  {t}", (22, y0 + 78), font, 0.48, _WHITE, 1, cv2.LINE_AA)

    if downloading:
        cv2.putText(
            frame,
            f"BUFFERING  {download_pct}%",
            (w - 200, y0 + 78),
            font,
            0.45,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )

    hint = footer_hint or "Q quit   S stats   P pause   T stream   +/- vol   N skip"
    ht = cv2.getTextSize(hint, font, 0.38, 1)[0][0]
    cv2.putText(frame, hint, ((w - ht) // 2, h - 10), font, 0.38, (120, 125, 135), 1, cv2.LINE_AA)
