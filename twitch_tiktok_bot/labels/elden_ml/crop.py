"""Adaptive TikTok gameplay ROI crop.

Uses the active LayoutProfile (continuous gameplay band) so ML sees the same
panel OpenCV scores — classic middle-third, half/half zoom, or anything between.
"""

from __future__ import annotations

import cv2
import numpy as np

from twitch_tiktok_bot.labels.elden_layout import get_active_layout


def gameplay_roi_xywh(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the adaptive gameplay region."""
    h, w = frame.shape[:2]
    layout = get_active_layout()
    y1 = int(h * layout.game_y0)
    y2 = int(h * layout.game_y1)
    return 0, y1, w, max(1, y2 - y1)


def crop_gameplay(frame: np.ndarray) -> np.ndarray:
    """Crop to adaptive gameplay band. Falls back to full frame if tiny."""
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    x, y, w, h = gameplay_roi_xywh(frame)
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
        return frame
    return crop


def crop_and_resize(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Gameplay crop resized to square for the classifier."""
    crop = crop_gameplay(frame)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
