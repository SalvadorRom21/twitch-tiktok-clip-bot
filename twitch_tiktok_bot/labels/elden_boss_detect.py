"""Elden Ring boss fight detection via gameplay-third boss health bar."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from twitch_tiktok_bot.analyze.duration import get_video_duration
from twitch_tiktok_bot.labels.elden_layout import (
    LayoutProfile,
    estimate_layout_from_video,
    estimate_layout_profile,
    get_active_layout,
    refine_layout_with_bar,
    set_active_layout,
)


def _io_workers(cap: int = 8) -> int:
    """Thread workers for ffmpeg seeks / OpenCV reads (I/O bound)."""
    return max(2, min(cap, (os.cpu_count() or 4)))


def _scan_sample_times(duration: float, interval_sec: float) -> list[float]:
    times: list[float] = []
    t = 0.0
    while t < duration:
        times.append(t)
        t += interval_sec
    return times or [0.0]


@dataclass
class FrameSignals:
    time_sec: float
    boss_bar_score: float = 0.0
    boss_bar_fill: float = 0.0
    boss_name_score: float = 0.0
    death_screen_score: float = 0.0
    victory_screen_score: float = 0.0
    # Two stacked boss HP bars (Tree Sentinel pair, etc.).
    dual_boss_bar_score: float = 0.0

    def combined_boss_score(self) -> float:
        dual = self.dual_boss_bar_score * 0.15
        return min(1.0, self.boss_bar_score * 0.70 + self.boss_name_score * 0.25 + dual)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BossFightCandidate:
    id: str
    start_sec: float
    end_sec: float
    confidence: float
    reason: str
    signals: dict = field(default_factory=dict)
    peak_boss_bar_score: float = 0.0
    had_death_screen: bool = False
    had_victory_screen: bool = False
    source: str = "auto"

    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BossScanResult:
    vod_id: str
    duration_sec: float = 0.0
    candidates: list[BossFightCandidate] = field(default_factory=list)
    frame_samples: list[FrameSignals] = field(default_factory=list)
    scanned_at: str = ""

    def to_dict(self) -> dict:
        return {
            "vod_id": self.vod_id,
            "duration_sec": self.duration_sec,
            "candidates": [c.to_dict() for c in self.candidates],
            "frame_samples": [f.to_dict() for f in self.frame_samples],
            "scanned_at": self.scanned_at,
        }

    def to_client_dict(self) -> dict:
        """API payload for trainer UI — omits heavy frame signal cache."""
        return {
            "vod_id": self.vod_id,
            "duration_sec": self.duration_sec,
            "candidates": [c.to_dict() for c in self.candidates],
            "scanned_at": self.scanned_at,
        }


def _gameplay_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
    """TikTok gameplay window from the active adaptive layout profile."""
    h, w = frame.shape[:2]
    layout = get_active_layout()
    y1 = int(h * layout.game_y0)
    y2 = int(h * layout.game_y1)
    return 0, y1, w, max(1, y2 - y1)


def _boss_bar_search_band(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Find the strongest boss-bar strip across layout hypotheses (any zoom).

    Returns (band, y0, y1) in full-frame coordinates.
    """
    h, w = frame.shape[:2]
    layout = get_active_layout()
    x0, x1 = int(w * 0.12), int(w * 0.88)
    band_h = max(8, int(h * 0.032))
    best_score = -1.0
    best = (np.zeros((band_h, max(1, x1 - x0), 3), dtype=np.uint8), 0, band_h)
    step = max(1, band_h // 2)
    for sweep0, sweep1 in layout.bar_sweep_hypotheses():
        y_start = int(h * sweep0)
        y_end = max(y_start + band_h + 1, int(h * sweep1) - band_h)
        for y0 in range(y_start, y_end, step):
            y1 = y0 + band_h
            band = frame[y0:y1, x0:x1]
            score = _score_boss_bar(band)
            if score > best_score:
                best_score = score
                best = (band, y0, y1)
    return best


def _boss_bar_band(frame: np.ndarray) -> np.ndarray:
    """Horizontal strip containing the Elden boss HP bar (layout-adaptive)."""
    band, _y0, _y1 = _boss_bar_search_band(frame)
    return band


def _boss_name_band(frame: np.ndarray) -> np.ndarray:
    """Band above the located boss bar for white boss name text.

    Dual-boss stacks (two Tree Sentinels) put a second name between bars —
    use a taller crop so at least one label is visible.
    """
    h, w = frame.shape[:2]
    _band, bar_y0, _bar_y1 = _boss_bar_search_band(frame)
    name_h = max(8, int(h * 0.045))
    y_name = max(0, bar_y0 - name_h)
    cx1 = int(w * 0.12)
    cx2 = int(w * 0.88)
    return frame[y_name : y_name + name_h, cx1:cx2]


def _score_dual_boss_bars(frame: np.ndarray) -> float:
    """Two stacked wide red HP strips — Merica Tree Sentinel pair, etc.

    Search only the lower gameplay band (boss HUD), not player HP / scenery.
    Both peaks must be wide, similar, and tightly stacked (~14–55px).
    """
    h, w = frame.shape[:2]
    y_a_f, y_b_f = get_active_layout().dual_bar_band()
    y_a, y_b = int(h * y_a_f), int(h * y_b_f)
    x_a, x_b = int(w * 0.18), int(w * 0.82)
    roi = frame[y_a:y_b, x_a:x_b]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 90, 70), (12, 255, 255)),
        cv2.inRange(hsv, (168, 90, 70), (180, 255, 255)),
    )
    row = red.mean(axis=1) / 255.0
    if row.size < 20:
        return 0.0
    sm = np.convolve(row, np.ones(5) / 5.0, mode="same")
    peaks: list[tuple[int, float, float]] = []
    for i in range(2, len(sm) - 2):
        if (
            sm[i] >= 0.18
            and sm[i] >= sm[i - 1]
            and sm[i] >= sm[i + 1]
            and sm[i] >= sm[i - 2]
            and sm[i] >= sm[i + 2]
        ):
            width = float(np.count_nonzero(red[i])) / max(1, red.shape[1])
            if width >= 0.35:
                peaks.append((i, float(sm[i]), width))
    peaks.sort(key=lambda p: -p[1])
    kept: list[tuple[int, float, float]] = []
    for p in peaks:
        if all(abs(p[0] - k[0]) >= 14 for k in kept):
            kept.append(p)
        if len(kept) >= 3:
            break
    kept.sort(key=lambda p: p[0])
    if len(kept) < 2:
        return 0.0
    a, b = kept[0], kept[1]
    sep = b[0] - a[0]
    if sep < 14 or sep > 55:
        return 0.0
    # Real dual bars are nearly the same width; vine/FX strips usually aren't.
    width_ratio = min(a[2], b[2]) / max(a[2], b[2], 1e-6)
    if width_ratio < 0.55:
        return 0.0
    strength = 0.5 * (a[1] + b[1])
    width = 0.5 * (a[2] + b[2])
    if strength < 0.22 or width < 0.40:
        return 0.0
    return float(min(1.0, strength * 1.6 + width * 0.55 + width_ratio * 0.15))


def _score_boss_bar(band: np.ndarray) -> float:
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    # Elden boss bar: saturated red horizontal strip
    red_low = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red_high = cv2.inRange(hsv, (168, 80, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)
    red_ratio = float(np.count_nonzero(red_mask)) / red_mask.size
    if red_ratio < 0.012:
        return 0.0
    # Boss bar spans most of the width — reject small red UI dots
    row_coverage = []
    for row in red_mask:
        nz = np.count_nonzero(row)
        if nz:
            row_coverage.append(nz / row.shape[0])
    width_score = max(row_coverage) if row_coverage else 0.0
    if width_score < 0.18:
        return red_ratio * 4.0
    return min(1.0, red_ratio * 6.0 + width_score * 0.55)


def _score_boss_bar_fill(band: np.ndarray) -> float:
    """Horizontal HP fill — ~1.0 at full boss health, lower when damaged."""
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red_high = cv2.inRange(hsv, (168, 80, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)
    best_fill = 0.0
    for row in red_mask:
        nz = np.where(row > 0)[0]
        if nz.size == 0:
            continue
        span = (int(nz[-1]) - int(nz[0]) + 1) / row.shape[0]
        best_fill = max(best_fill, span)
    return round(min(1.0, best_fill), 3)


def _score_boss_name(band: np.ndarray) -> float:
    if band.size == 0:
        return 0.0
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bright = gray > 185
    ratio = float(np.count_nonzero(bright)) / bright.size
    if ratio < 0.008:
        return 0.0
    # Name text is a horizontal cluster, not full-width noise
    row_hits = [np.count_nonzero(row) / row.shape[0] for row in bright if np.any(row)]
    cluster = max(row_hits) if row_hits else 0.0
    if cluster < 0.04:
        return ratio * 3.0
    return min(1.0, ratio * 4.0 + cluster * 0.4)


def _center_text_roi(frame: np.ndarray) -> np.ndarray:
    """Center of adaptive gameplay band — YOU DIED / ENEMY FELLED live here."""
    h, w = frame.shape[:2]
    y0_f, y1_f = get_active_layout().center_text_band()
    cx1 = int(w * 0.12)
    cx2 = int(w * 0.88)
    cy1 = int(h * y0_f)
    cy2 = int(h * y1_f)
    return frame[cy1:cy2, cx1:cx2]


def _letterform_band_score(mask: np.ndarray) -> tuple[float, float]:
    """Return (max_row_coverage, horizontal_structure) for text-like masks."""
    if mask.size == 0:
        return 0.0, 0.0
    row_coverages = [
        float(np.count_nonzero(row)) / row.shape[0] for row in mask if np.any(row)
    ]
    if not row_coverages:
        return 0.0, 0.0
    band_width = max(row_coverages)
    # Text occupies a mid band of rows, not the whole crop (rejects gold bosses / fire).
    active = sum(1 for c in row_coverages if c >= 0.08)
    structure = active / max(1, mask.shape[0])
    return band_width, structure


def _score_loot_popup(frame: np.ndarray) -> float:
    """Dark loot / OK box that often follows ENEMY FELLED (0717 pattern)."""
    x, y, w, h = _gameplay_roi(frame)
    box = frame[y + int(h * 0.42) : y + int(h * 0.82), x + int(w * 0.15) : x + int(w * 0.85)]
    if box.size == 0:
        return 0.0
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    dark = float(np.count_nonzero(gray < 90)) / gray.size
    bright = float(np.count_nonzero(gray > 160)) / gray.size
    # Semi-transparent dark plate with a strip of bright item text.
    if dark >= 0.28 and 0.008 <= bright <= 0.28:
        return min(1.0, dark * 0.7 + bright * 2.2)
    return 0.0


def _score_player_hp_depleted(frame: np.ndarray) -> float:
    """How empty the player's top-left HP bar looks (1.0 ≈ dead / no red fill).

    Searches the top of the active gameplay band and a couple fallbacks so
    zoomed bottom-panel layouts still see the HP strip.
    """
    h, w = frame.shape[:2]
    layout = get_active_layout()
    bands = [
        (layout.game_y0, layout.game_y1),
        (0.48, 0.90),
        (0.30, 0.76),
    ]
    best = 0.0
    for gy0, gy1 in bands:
        gh = max(0.01, gy1 - gy0)
        y0 = int(h * (gy0 + gh * 0.04))
        y1 = int(h * (gy0 + gh * 0.22))
        band = frame[y0:y1, int(w * 0.02) : int(w * 0.34)]
        if band.size == 0:
            continue
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 60, 40), (14, 255, 255)),
            cv2.inRange(hsv, (166, 60, 40), (180, 255, 255)),
        )
        red_ratio = float(np.count_nonzero(red)) / red.size
        best = max(best, float(np.clip(1.0 - (red_ratio / 0.035), 0.0, 1.0)))
    return best


def _mist_door_score(frame: np.ndarray) -> float:
    """Yellow/black fog wall before many boss arenas (0717(1) Fire Giant)."""
    x, y, w, h = _gameplay_roi(frame)
    roi = frame[
        y + int(h * 0.15) : y + int(h * 0.85),
        x + int(w * 0.15) : x + int(w * 0.85),
    ]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 40, 60), (45, 255, 255))
    dark = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 0, 50)
    y_ratio = float(np.count_nonzero(yellow)) / yellow.size
    d_ratio = float(np.count_nonzero(dark)) / dark.size
    if y_ratio < 0.04 or d_ratio < 0.15:
        return 0.0
    return min(1.0, y_ratio * 4.0 + d_ratio * 0.4)


def _cinematic_letterbox_score(frame: np.ndarray) -> float:
    """Phase cinematics often letterbox the gameplay third (0717(1) ~174s)."""
    x, y, w, h = _gameplay_roi(frame)
    strip = max(4, int(h * 0.07))
    top = frame[y : y + strip, x : x + w]
    bot = frame[y + h - strip : y + h, x : x + w]
    if top.size == 0 or bot.size == 0:
        return 0.0
    top_dark = float(np.mean(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) < 28))
    bot_dark = float(np.mean(cv2.cvtColor(bot, cv2.COLOR_BGR2GRAY) < 28))
    return float(np.clip(0.5 * (top_dark + bot_dark), 0.0, 1.0))


def _death_banner_metrics(center: np.ndarray) -> tuple[float, float, float, float]:
    """Local YOU DIED plate metrics — works when fire washes the rest of the frame.

    Returns (plate_dark, mid_red, band_width, structure).
    """
    if center.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    h = center.shape[0]
    y0, y1 = int(h * 0.26), int(h * 0.66)
    mid = center[y0:y1, :]
    if mid.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 90, 50), (12, 255, 255)),
        cv2.inRange(hsv, (168, 90, 50), (180, 255, 255)),
    )
    gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    mid_red = float(np.count_nonzero(red)) / red.size
    band_width, structure = _letterform_band_score(red)

    # Contiguous dark+red rows = the semi-transparent YOU DIED plate.
    row_hits: list[float] = []
    for y in range(mid.shape[0]):
        dark_frac = float(np.count_nonzero(gray[y] < 70)) / max(1, gray.shape[1])
        red_frac = float(np.count_nonzero(red[y])) / max(1, red.shape[1])
        if dark_frac >= 0.42 and red_frac >= 0.015:
            row_hits.append(dark_frac)
        else:
            row_hits.append(0.0)
    best_len = 0
    best_dark = 0.0
    cur_len = 0
    cur_dark = 0.0
    for val in row_hits:
        if val > 0:
            cur_len += 1
            cur_dark += val
            if cur_len > best_len:
                best_len = cur_len
                best_dark = cur_dark / cur_len
        else:
            cur_len = 0
            cur_dark = 0.0
    plate_h = best_len / max(1, mid.shape[0])
    # Plate should be a banner strip, not the whole mid crop (fire wash).
    if plate_h < 0.06 or plate_h > 0.55:
        plate_dark = 0.0
    else:
        plate_dark = best_dark * min(1.0, plate_h / 0.12)
    return plate_dark, mid_red, band_width, structure


def _red_text_line_score(center: np.ndarray) -> float:
    """Score how much red looks like a horizontal YOU DIED letter line.

    Cloaks / petals / FX make dark+red plates; real YOU DIED is 5+ letter-sized
    blobs sharing one horizontal band.
    """
    if center.size == 0:
        return 0.0
    h = center.shape[0]
    mid = center[int(h * 0.22) : int(h * 0.70), :]
    if mid.size == 0:
        return 0.0
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 100, 60), (12, 255, 255)),
        cv2.inRange(hsv, (168, 100, 60), (180, 255, 255)),
    )
    # Clean speckles.
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(red, connectivity=8)
    if n <= 1:
        return 0.0
    mh, mw = mid.shape[:2]
    letters: list[tuple[float, float, float, float]] = []  # cy, cx, area, bh
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 40 or area > 0.04 * mid.size:
            continue
        if bh < 8 or bw < 4:
            continue
        # YOU DIED glyphs are medium height — not petal dots or cloak slabs.
        if bh < 0.045 * mh or bh > 0.28 * mh or bw > 0.22 * mw:
            continue
        aspect = bw / max(1.0, bh)
        if aspect > 2.8 or aspect < 0.15:
            continue
        letters.append((float(cents[i][1]), float(cents[i][0]), area, bh))
    if len(letters) < 6:
        return 0.0
    ys = np.array([p[0] for p in letters], dtype=np.float32)
    med = float(np.median(ys))
    band = [p for p in letters if abs(p[0] - med) <= 0.08 * mh]
    if len(band) < 6:
        return 0.0
    heights = np.array([p[3] for p in band], dtype=np.float32)
    if float(np.mean(heights)) <= 0:
        return 0.0
    if float(np.std(heights)) / float(np.mean(heights)) > 0.45:
        return 0.0  # mixed petals / FX sizes
    xs = sorted(p[1] for p in band)
    span = (xs[-1] - xs[0]) / max(1.0, mw)
    if span < 0.28 or span > 0.85:
        return 0.0
    # Text line should sit in the upper-mid of this ROI (not on the boss bar).
    if med > 0.72 * mh:
        return 0.0
    return float(min(1.0, len(band) / 12.0 + span * 0.4))


def _score_death_in_center(
    center: np.ndarray,
    *,
    boss_bar_score: float,
    hp_empty: float,
    letterbox: float,
) -> float:
    if center.size == 0:
        return 0.0
    plate_dark, mid_red, band_width, structure = _death_banner_metrics(center)
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    global_dark = float(np.count_nonzero(gray < 45)) / gray.size
    # Status banners (FROSTBITE) are white; stream TTS overlays are white too.
    # Only veto when red letterforms are weak.
    white = (gray > 195).astype(np.uint8) * 255
    white_bw, _white_st = _letterform_band_score(white)
    if (
        white_bw >= 0.22
        and band_width < 0.14
        and white_bw >= max(0.20, band_width * 1.35)
    ):
        return 0.0

    # Reject scarlet-rot / fire washes that fill the ROI with red (not YOU DIED).
    if band_width >= 0.60 and structure >= 0.45:
        return 0.0
    if mid_red >= 0.18 and structure >= 0.55:
        return 0.0

    # Letter-line check rejects Malenia cloak/petal FPs. Fire Giant YOU DIED is
    # often ember-merged (weak components) — allow fire_path without it.
    text_line = _red_text_line_score(center)

    fire_path = (
        plate_dark >= 0.45
        and mid_red >= 0.08
        and 0.28 <= band_width <= 0.70
        and boss_bar_score >= 0.28
        and structure <= 0.85
    )
    # Zoomed TikTok deaths often keep a bright facecam / arena; the dark plate
    # + red letter line is enough without a full-frame vignette. Plate can be
    # only ~0.5 on compressed scan frames (letter line still strong).
    dark_ok = global_dark >= 0.34 or (
        plate_dark >= 0.48 and text_line >= 0.55
    )
    classic_path = (
        dark_ok
        # Compact red letter strip — scarlet-rot / cloak washes are hotter/wider.
        and 0.022 <= mid_red <= 0.08
        and 0.18 <= band_width <= 0.40
        and 0.16 <= structure <= 0.40
        and text_line >= 0.35
        # Real YOU DIED sits on a dark plate; fire particles often don't.
        and plate_dark >= 0.40
    )
    if not fire_path and not classic_path:
        return 0.0
    # Cloak / fire-wash plates trip fire_path without glyphs. Ember-merged
    # YOU DIED may lack components, but washes are wide/hot — reject those.
    if fire_path and text_line < 0.35:
        if (
            global_dark < 0.42
            or band_width > 0.48
            or mid_red > 0.14
            or structure > 0.48
        ):
            return 0.0
    score = (
        mid_red * 3.6
        + max(global_dark, plate_dark) * 0.55
        + band_width * 0.6
        + hp_empty * 0.35
        + text_line * 0.45
    )
    if fire_path:
        score += 0.2
    if boss_bar_score >= 0.35:
        score += 0.1
    if letterbox >= 0.55:
        score *= 0.55
    return min(1.0, score)


def _score_death_screen(frame: np.ndarray, *, boss_bar_score: float = 0.0) -> float:
    """YOU DIED — dark plate + compact red banner. Boss bar may still be full/low.

    Tries several vertical center ROIs so zoomed bottom-panel deaths still score.
    """
    letterbox = _cinematic_letterbox_score(frame)
    if letterbox >= 0.72 and boss_bar_score < 0.35:
        return 0.0
    hp_empty = _score_player_hp_depleted(frame)
    if hp_empty < 0.55:
        return 0.0
    h, w = frame.shape[:2]
    best = 0.0
    for y0_f, y1_f in get_active_layout().center_text_hypotheses():
        center = frame[int(h * y0_f) : int(h * y1_f), int(w * 0.12) : int(w * 0.88)]
        best = max(
            best,
            _score_death_in_center(
                center,
                boss_bar_score=boss_bar_score,
                hp_empty=hp_empty,
                letterbox=letterbox,
            ),
        )
    return best


def _score_victory_screen(frame: np.ndarray, *, boss_bar_score: float = 0.0) -> float:
    """ENEMY FELLED / GREAT ENEMY FELLED — gold/cream banner after boss bar is gone.

    Clip 0717: desaturated gold banner. 0717(2): pale FELLED + loot popup with
    mid-gold below the usual threshold — allow loot confirm only when the bar is
    truly empty (not mid-fight residual red ~0.20).
    """
    # Residual bar (~0.20) means the fight is still on (0717(2) false FELLED).
    if boss_bar_score >= 0.12:
        return 0.0
    center = _center_text_roi(frame)
    if center.size == 0:
        return 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    # Loose gold/cream — Elden banners are often pale, not neon orange.
    gold = cv2.inRange(hsv, (8, 35, 90), (50, 255, 255))
    gold_ratio = float(np.count_nonzero(gold)) / gold.size
    if gold_ratio <= 0.015:
        return 0.0
    # Environment wash (doorway / fire / gold boss filling the crop).
    if gold_ratio > 0.32:
        return 0.0
    band_width, structure = _letterform_band_score(gold)
    if band_width < 0.12:
        return 0.0
    mid = center[int(center.shape[0] * 0.20) : int(center.shape[0] * 0.68), :]
    mid_gold = cv2.inRange(
        cv2.cvtColor(mid, cv2.COLOR_BGR2HSV), (8, 35, 90), (50, 255, 255)
    )
    mid_ratio = float(np.count_nonzero(mid_gold)) / max(1, mid_gold.size)
    loot = _score_loot_popup(frame)
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    dark = float(np.count_nonzero(gray < 55)) / gray.size
    # Leyndell / outdoor gold stone wash fills the crop with structure but little
    # dark plate (Merica stairs ~430–580). Real FELLED sits on a darker banner.
    if structure > 0.70 and dark < 0.32:
        return 0.0
    # Bright outdoor / fire-arena gold wash with no loot plate.
    if structure > 0.50 and loot < 0.35 and mid_ratio < 0.10:
        return 0.0
    if structure > 0.68 and loot < 0.35:
        return 0.0
    # Strong mid-band gold, OR pale wide banner + loot after bar emptied (0717(2)).
    # False mid-fight gold wash has loot-like dark boxes but a narrow band (bw~0.15);
    # real ENEMY FELLED is a wide letterform strip (bw~0.33) even when pale.
    pale_loot = (
        mid_ratio >= 0.02
        and loot >= 0.40
        and boss_bar_score < 0.08
        and band_width >= 0.22
        and 0.05 <= structure <= 0.28
        and dark >= 0.28
    )
    if mid_ratio < 0.07 and not pale_loot:
        return 0.0
    # Gate/arch "loot-like" dark boxes in bright yards (Merica ~628).
    if loot >= 0.40 and dark < 0.28 and structure > 0.40:
        return 0.0
    # Particles alone: high structure + low mid concentration → weak.
    if structure > 0.75 and mid_ratio < 0.10 and loot < 0.40:
        return min(0.35, mid_ratio * 3.0)
    score = mid_ratio * 4.2 + band_width * 1.0 + loot * 0.55
    if boss_bar_score < 0.08:
        score += 0.15
    if loot >= 0.40:
        score += 0.25
    if pale_loot:
        score += 0.2
    return min(1.0, score)


def analyze_frame(frame: np.ndarray, time_sec: float) -> FrameSignals:
    band, bar_y0, bar_y1 = _boss_bar_search_band(frame)
    name = _boss_name_band(frame)
    bar_score = round(_score_boss_bar(band), 3)
    dual = round(_score_dual_boss_bars(frame), 3)
    name_score = round(_score_boss_name(name), 3)
    # Dual stacks often bury one name label; treat dual as soft name evidence.
    if dual >= 0.55 and name_score < 0.20:
        name_score = max(name_score, 0.22)
    # Soft-refine layout when a confident named HUD bar appears (any zoom).
    # Require name text so LIKES/follower chrome cannot drag the band upward.
    if bar_score >= 0.75 and name_score >= 0.45 and frame is not None:
        h = frame.shape[0]
        bar_c = ((bar_y0 + bar_y1) * 0.5) / max(1, h)
        layout = get_active_layout()
        if (layout.game_y0 - 0.04) <= bar_c <= (layout.game_y1 + 0.04):
            set_active_layout(
                refine_layout_with_bar(layout, bar_c, alpha=0.18)
            )
    death = _score_death_screen(frame, boss_bar_score=bar_score)
    return FrameSignals(
        time_sec=round(time_sec, 2),
        boss_bar_score=bar_score,
        boss_bar_fill=round(_score_boss_bar_fill(band), 3),
        boss_name_score=name_score,
        death_screen_score=round(death, 3),
        victory_screen_score=round(
            _score_victory_screen(frame, boss_bar_score=bar_score), 3
        ),
        dual_boss_bar_score=dual,
    )


def _extract_sample_frame(
    video_path: Path,
    time_sec: float,
    out_path: Path,
    ffmpeg: str = "ffmpeg",
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def _count_prefix_frames(frames_dir: Path, expected: int) -> int:
    n = 0
    while n < expected and (frames_dir / f"frame_{n:05d}.jpg").exists():
        n += 1
    return n


def _extract_scan_grid_one_pass(
    video_path: Path,
    frames_dir: Path,
    *,
    interval_sec: float,
    expected: int,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
) -> bool:
    """Single ffmpeg pass at fps=1/interval → frame_00000.jpg… (huge vs per-seek)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    if expected <= 0:
        return False
    # Reuse a nearly-complete cache (rescans / interrupted runs).
    if _count_prefix_frames(frames_dir, expected) >= max(1, int(expected * 0.95)):
        return True

    pattern = str(frames_dir / "frame_%05d.jpg")
    fps = 1.0 / max(0.05, float(interval_sec))
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:.8f}",
        "-start_number",
        "0",
        "-q:v",
        "3",
        pattern,
    ]
    if on_progress:
        on_progress(
            {
                "phase": "sampling",
                "message": f"One-pass extract (0/{expected})",
                "current": 0,
                "total": expected,
                "phase_frac": 0.0,
            }
        )
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return False

    last_n = -1
    while proc.poll() is None:
        n = _count_prefix_frames(frames_dir, expected)
        if on_progress and n != last_n:
            last_n = n
            on_progress(
                {
                    "phase": "sampling",
                    "message": f"One-pass extract ({n}/{expected})",
                    "current": n,
                    "total": expected,
                    "phase_frac": n / max(1, expected),
                }
            )
        time.sleep(0.4)
    if proc.stderr is not None:
        try:
            proc.stderr.close()
        except OSError:
            pass
    if proc.returncode != 0:
        return False
    got = _count_prefix_frames(frames_dir, expected)
    if on_progress:
        on_progress(
            {
                "phase": "sampling",
                "message": f"One-pass extract ({got}/{expected})",
                "current": got,
                "total": expected,
                "phase_frac": 1.0,
            }
        )
    # Accept if we got a usable majority; gaps filled by parallel seeks.
    return got >= max(1, int(expected * 0.80)) or got >= expected - 2


def _fill_missing_scan_frames(
    video_path: Path,
    frames_dir: Path,
    times: list[float],
    *,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
) -> None:
    """Parallel single-frame seeks for any gaps left after one-pass extract."""
    jobs = [
        (idx, t, frames_dir / f"frame_{idx:05d}.jpg")
        for idx, t in enumerate(times)
        if not (frames_dir / f"frame_{idx:05d}.jpg").exists()
        or (frames_dir / f"frame_{idx:05d}.jpg").stat().st_size <= 0
    ]
    if not jobs:
        if on_progress:
            on_progress(
                {
                    "phase": "gap_fill",
                    "message": "Frame grid ready",
                    "current": 1,
                    "total": 1,
                    "phase_frac": 1.0,
                }
            )
        return
    total = len(jobs)
    workers = _io_workers()
    done = 0

    def _one(job: tuple[int, float, Path]) -> bool:
        _idx, t, path = job
        return _extract_sample_frame(video_path, t, path, ffmpeg=ffmpeg)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if on_progress and (done % 8 == 0 or done >= total):
                on_progress(
                    {
                        "phase": "gap_fill",
                        "message": f"Filling gaps ({done}/{total})",
                        "current": done,
                        "total": total,
                        "phase_frac": done / max(1, total),
                    }
                )


def _bootstrap_layout(
    video_path: Path,
    *,
    duration: float,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
) -> LayoutProfile:
    """Estimate continuous gameplay band before OpenCV/ML scoring."""
    if on_progress:
        on_progress(
            {
                "phase": "starting",
                "message": "Estimating gameplay layout / zoom…",
                "current": 0,
                "total": 1,
                "phase_frac": 0.8,
            }
        )
    sample_times = [
        t
        for t in (3.0, 12.0, 30.0, 60.0, 120.0, 180.0, max(5.0, duration * 0.35), max(8.0, duration * 0.65))
        if t < max(1.0, duration - 1.0)
    ]
    # Dedupe while preserving order.
    seen: set[float] = set()
    ordered: list[float] = []
    for t in sample_times:
        key = round(t, 1)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(float(t))
    profile = estimate_layout_from_video(
        video_path, ffmpeg=ffmpeg, sample_times=ordered or [5.0]
    )
    set_active_layout(profile)
    if on_progress:
        on_progress(
            {
                "phase": "starting",
                "message": (
                    f"Layout y={profile.game_y0:.2f}–{profile.game_y1:.2f} "
                    f"(adaptive zoom)"
                ),
                "current": 1,
                "total": 1,
                "phase_frac": 1.0,
            }
        )
    return profile


def extract_scan_frame_grid(
    video_path: Path,
    work_dir: Path,
    *,
    interval_sec: float = 2.0,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
) -> tuple[Path, float, list[float]]:
    """Populate ``boss_scan_frames/`` via one-pass ffmpeg (+ parallel gap fill)."""
    duration = get_video_duration(video_path, ffmpeg=ffmpeg)
    _bootstrap_layout(
        video_path, duration=duration, ffmpeg=ffmpeg, on_progress=on_progress
    )
    times = _scan_sample_times(duration, interval_sec)
    frames_dir = work_dir / "boss_scan_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    ok = _extract_scan_grid_one_pass(
        video_path,
        frames_dir,
        interval_sec=interval_sec,
        expected=len(times),
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    if not ok and on_progress:
        on_progress(
            {
                "phase": "sampling",
                "message": "Parallel frame extract…",
                "current": 0,
                "total": len(times),
                "phase_frac": 0.0,
            }
        )
    _fill_missing_scan_frames(
        video_path,
        frames_dir,
        times,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    return frames_dir, duration, times


def _analyze_frame_path(args: tuple[Path, float]) -> FrameSignals | None:
    path, t = args
    img = cv2.imread(str(path))
    if img is None:
        return None
    return analyze_frame(img, t)


def _analyze_paths_parallel(
    items: list[tuple[Path, float]],
    *,
    on_progress: object | None = None,
    phase: str = "scoring",
    message: str = "Scoring frames",
    frac_lo: float = 0.0,
    frac_hi: float = 1.0,
) -> list[FrameSignals]:
    if not items:
        return []
    total = len(items)
    workers = _io_workers(cap=max(4, min(12, (os.cpu_count() or 4))))
    out: list[FrameSignals | None] = [None] * total
    done = 0
    span = max(0.0, frac_hi - frac_lo)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_analyze_frame_path, items[i]): i for i in range(total)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            out[i] = fut.result()
            done += 1
            if on_progress and (done % 20 == 0 or done >= total):
                on_progress(
                    {
                        "phase": phase,
                        "message": f"{message} ({done}/{total})",
                        "current": done,
                        "total": total,
                        "phase_frac": frac_lo + span * (done / max(1, total)),
                    }
                )
    return [s for s in out if s is not None]


def _extract_probe_jobs_parallel(
    video_path: Path,
    jobs: list[tuple[float, Path]],
    *,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
    phase: str = "probes",
    message: str = "Probing",
    frac_lo: float = 0.0,
    frac_hi: float = 0.65,
) -> list[tuple[float, Path]]:
    """Extract many seek frames in parallel. Returns successful (time, path) pairs."""
    if not jobs:
        return []
    total = len(jobs)
    workers = _io_workers()
    done = 0
    ok: list[tuple[float, Path]] = []
    span = max(0.0, frac_hi - frac_lo)

    def _one(job: tuple[float, Path]) -> tuple[float, Path] | None:
        t, path = job
        if path.exists() and path.stat().st_size > 0:
            return t, path
        if _extract_sample_frame(video_path, t, path, ffmpeg=ffmpeg):
            return t, path
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result is not None:
                ok.append(result)
            if on_progress and (done % 4 == 0 or done >= total):
                on_progress(
                    {
                        "phase": phase,
                        "message": f"{message} ({done}/{total})",
                        "current": done,
                        "total": total,
                        "phase_frac": frac_lo + span * (done / max(1, total)),
                    }
                )
    ok.sort(key=lambda x: x[0])
    return ok


def sample_frame_signals(
    video_path: Path,
    work_dir: Path,
    *,
    interval_sec: float = 2.0,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
) -> tuple[list[FrameSignals], float]:
    frames_dir, duration, times = extract_scan_frame_grid(
        video_path,
        work_dir,
        interval_sec=interval_sec,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    items = [
        (frames_dir / f"frame_{idx:05d}.jpg", t)
        for idx, t in enumerate(times)
        if (frames_dir / f"frame_{idx:05d}.jpg").exists()
    ]
    signals = _analyze_paths_parallel(
        items,
        on_progress=on_progress,
        phase="scoring",
        message="Scoring frames",
        frac_lo=0.0,
        frac_hi=1.0,
    )
    return signals, duration


def supplement_start_probes(
    signals: list[FrameSignals],
    video_path: Path,
    tuning: EldenTuning,
    *,
    probe_dir: Path,
    lookback_sec: float | None = None,
    interval_sec: float = 1.0,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
) -> list[FrameSignals]:
    """1 Hz probes before each boss HUD cluster — catches bar pop between 2s scan samples."""
    if not video_path.exists():
        return signals
    lookback_sec = lookback_sec if lookback_sec is not None else tuning.start_bar_lookback_sec
    by_time = {round(s.time_sec, 3): s for s in signals}
    paired_times = sorted(s.time_sec for s in signals if _boss_hud_present(s, tuning))
    if not paired_times:
        return signals

    cluster_starts: list[float] = [paired_times[0]]
    last = paired_times[0]
    for pt in paired_times[1:]:
        if pt - last <= 120.0:
            last = pt
            continue
        cluster_starts.append(pt)
        last = pt

    probe_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[float, Path]] = []
    for first_paired in cluster_starts:
        fine_start = max(0.0, first_paired - 45.0)
        coarse_start = max(0.0, first_paired - lookback_sec)
        t = coarse_start
        while t < first_paired - 0.25:
            step = 0.5 if t >= fine_start else 1.0
            key = round(t, 3)
            if key not in by_time:
                jobs.append((t, probe_dir / f"start_{int(t * 10):06d}.jpg"))
            t += step
    extracted = _extract_probe_jobs_parallel(
        video_path,
        jobs,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
        phase="start_probes",
        message="Refining fight starts",
        frac_lo=0.0,
        frac_hi=0.65,
    )
    scored = _analyze_paths_parallel(
        [(path, t) for t, path in extracted],
        on_progress=on_progress,
        phase="start_probes",
        message="Scoring start probes",
        frac_lo=0.65,
        frac_hi=1.0,
    )
    for s in scored:
        by_time[round(s.time_sec, 3)] = s
    return sorted(by_time.values(), key=lambda s: s.time_sec)


def supplement_end_probes(
    signals: list[FrameSignals],
    video_path: Path,
    tuning: EldenTuning,
    *,
    probe_dir: Path,
    ffmpeg: str = "ffmpeg",
    on_progress: object | None = None,
) -> list[FrameSignals]:
    """Dense probes at the end of boss-bar runs — FELLED and YOU DIED are brief.

    Uses bar score (not name): Fire Giant / dark arenas often score name=0 while
    the red bar is clearly up. Also probes the run *tail* even when the bar never
    drops — YOU DIED can appear with the boss bar still visible (0717(1)).
    """
    if not video_path.exists():
        return signals
    by_time = {round(s.time_sec, 3): s for s in signals}
    bar_times = sorted(
        s.time_sec
        for s in signals
        if _is_hud_hit(s, tuning) or _boss_hud_present(s, tuning)
    )
    if not bar_times:
        return signals
    windows: list[tuple[float, float]] = []
    cluster_start = bar_times[0]
    last = bar_times[0]
    for pt in bar_times[1:]:
        # Short bridge only — a mid-fight bar dip (YOU DIED / phase) should
        # open a new window so we densify that tail, not just the VOD end.
        if pt - last <= 10.0:
            last = pt
            continue
        windows.append((cluster_start, last + 12.0))
        cluster_start = pt
        last = pt
    windows.append((cluster_start, last + 12.0))

    # Probe last 24s of each run at 0.5s (covers brief YOU DIED between 2s samples).
    probe_spans: list[tuple[float, float, float]] = []
    for win_start, win_end in windows:
        tail0 = max(win_start, win_end - 24.0)
        probe_spans.append((tail0, win_end, 0.5))
    # Also densify around mid-run HUD gaps (Fire Giant YOU DIED ~173 with bar still up).
    for i in range(len(bar_times) - 1):
        gap = bar_times[i + 1] - bar_times[i]
        if gap >= 6.0:
            probe_spans.append(
                (max(0.0, bar_times[i] - 1.0), bar_times[i + 1] + 2.0, 0.5)
            )

    probe_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[float, Path]] = []
    for t0, t1, step in probe_spans:
        t = t0
        while t <= t1:
            key = round(t, 3)
            if key not in by_time:
                jobs.append((t, probe_dir / f"probe_{int(t * 10):06d}.jpg"))
            t += step
    extracted = _extract_probe_jobs_parallel(
        video_path,
        jobs,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
        phase="end_probes",
        message="Refining fight ends",
        frac_lo=0.0,
        frac_hi=0.65,
    )
    scored = _analyze_paths_parallel(
        [(path, t) for t, path in extracted],
        on_progress=on_progress,
        phase="end_probes",
        message="Scoring end probes",
        frac_lo=0.65,
        frac_hi=1.0,
    )
    for s in scored:
        by_time[round(s.time_sec, 3)] = s
    return sorted(by_time.values(), key=lambda s: s.time_sec)


def supplement_terminal_probes(
    signals: list[FrameSignals],
    video_path: Path,
    *,
    probe_dir: Path,
    ffmpeg: str = "ffmpeg",
    radius_sec: float = 4.0,
    step_sec: float = 0.5,
    death_hint: float = 0.28,
    victory_hint: float = 0.30,
    on_progress: object | None = None,
) -> list[FrameSignals]:
    """Dense probes around likely YOU DIED / ENEMY FELLED hits (terminal-first)."""
    if not video_path.exists() or not signals:
        return signals
    hints = sorted(
        {
            round(s.time_sec, 1)
            for s in signals
            if s.death_screen_score >= death_hint or s.victory_screen_score >= victory_hint
        }
    )
    # Also hunt right after strong boss-bar stretches — death banners are often between 2s samples.
    ordered = sorted(signals, key=lambda s: s.time_sec)
    bar_run_end: float | None = None
    bar_run_start: float | None = None
    for s in ordered:
        if s.boss_bar_score >= 0.40 or s.boss_bar_fill >= 0.45:
            if bar_run_start is None:
                bar_run_start = s.time_sec
            bar_run_end = s.time_sec
        elif bar_run_end is not None and s.time_sec - bar_run_end <= 2.5:
            hints.append(round(bar_run_end + 2.0, 1))
            hints.append(round(bar_run_end + 4.0, 1))
            bar_run_start = None
            bar_run_end = None
        else:
            # Bar still up at run end (0717(1) YOU DIED) — seed the tail.
            if (
                bar_run_start is not None
                and bar_run_end is not None
                and bar_run_end - bar_run_start >= 8.0
            ):
                t = max(bar_run_start, bar_run_end - 18.0)
                while t <= bar_run_end + 1.0:
                    hints.append(round(t, 1))
                    t += 3.0
            bar_run_start = None
            bar_run_end = None
    if (
        bar_run_start is not None
        and bar_run_end is not None
        and bar_run_end - bar_run_start >= 8.0
    ):
        t = max(bar_run_start, bar_run_end - 18.0)
        while t <= bar_run_end + 1.0:
            hints.append(round(t, 1))
            t += 3.0
    hints = sorted(set(hints))
    if not hints:
        return signals

    by_time = {round(s.time_sec, 3): s for s in signals}
    centers: list[float] = []
    for t in hints:
        if centers and t - centers[-1] < 5.0:
            continue
        centers.append(float(t))

    probe_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[float, Path]] = []
    for center in centers:
        t = max(0.0, center - radius_sec)
        end = center + radius_sec
        while t <= end:
            key = round(t, 3)
            if key not in by_time:
                jobs.append((t, probe_dir / f"term_{int(t * 10):06d}.jpg"))
            t += step_sec
    extracted = _extract_probe_jobs_parallel(
        video_path,
        jobs,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
        phase="terminal_probes",
        message="Refining YOU DIED / FELLED",
        frac_lo=0.0,
        frac_hi=0.65,
    )
    scored = _analyze_paths_parallel(
        [(path, t) for t, path in extracted],
        on_progress=on_progress,
        phase="terminal_probes",
        message="Scoring terminal probes",
        frac_lo=0.65,
        frac_hi=1.0,
    )
    for s in scored:
        by_time[round(s.time_sec, 3)] = s
    return sorted(by_time.values(), key=lambda s: s.time_sec)


def collect_terminal_events(
    signals: list[FrameSignals],
    *,
    death_threshold: float = 0.50,
    victory_threshold: float = 0.55,
    collapse_sec: float = 8.0,
) -> list[tuple[float, str, float]]:
    """Peak YOU DIED / ENEMY FELLED times.

    Victory requires bar gone. Death does not — failed attempts keep HP on the boss.
    """
    raw: list[tuple[float, str, float]] = []
    for s in signals:
        if s.victory_screen_score >= victory_threshold and s.boss_bar_score < 0.12:
            raw.append((s.time_sec, "victory", s.victory_screen_score))
        elif s.death_screen_score >= death_threshold:
            raw.append((s.time_sec, "death", s.death_screen_score))
    raw.sort(key=lambda x: (x[0], -x[2]))
    collapsed: list[tuple[float, str, float]] = []
    for t, kind, score in raw:
        if collapsed and t - collapsed[-1][0] < collapse_sec:
            if score > collapsed[-1][2]:
                collapsed[-1] = (t, kind, score)
            continue
        collapsed.append((t, kind, score))
    return collapsed


def _dual_score(sig: FrameSignals) -> float:
    return float(getattr(sig, "dual_boss_bar_score", 0.0) or 0.0)


def _is_hud_hit(sig: FrameSignals, tuning: EldenTuning) -> bool:
    """Frame counts toward a boss-HUD run.

    Dual bars are strong evidence (Merica Tree Sentinels). Bar-only is kept for
    Fire Giant / dark arenas, but weak outdoor red blips need name or dual.
    Zoomed TikTok layouts fake a full red bar with LIKES / follower chrome.
    """
    dual = _dual_score(sig)
    if dual >= 0.50:
        return True
    if _boss_hud_present(sig, tuning):
        return True
    if sig.boss_bar_score >= tuning.start_bar_min_score and sig.boss_name_score >= 0.12:
        return True
    # Bar-only path (0717(1) Fire Giant): strong contiguous HP strip.
    # Reject near-full static chrome bars (LIKES) unless a name is readable.
    if sig.boss_bar_score >= 0.70 and 0.35 <= sig.boss_bar_fill <= 0.98:
        if sig.boss_name_score >= 0.12:
            return True
        if sig.boss_bar_fill <= 0.92 and get_active_layout().game_y0 < 0.38:
            return True
    return False


def _run_has_proven_hud(run: list[FrameSignals]) -> bool:
    """True once a run looks like a real fight (not a 1–2s red FX blip)."""
    if not run:
        return False
    peak_dual = max(_dual_score(s) for s in run)
    if peak_dual >= 0.50:
        return True
    # Need repeated bar+name pairs — a single map/legend flash is not enough.
    paired = [
        s
        for s in run
        if s.boss_name_score >= 0.35
        and s.boss_bar_score >= 0.55
        and 0.28 <= s.boss_bar_fill <= 0.98
    ]
    if len(paired) >= 3:
        return True
    dur = run[-1].time_sec - run[0].time_sec
    # Sustained bar-only fights (Fire Giant).
    return dur >= 25.0 and max(s.boss_bar_score for s in run) >= 0.70


def _run_is_boss_attempt(run: list[FrameSignals], *, duration: float) -> bool:
    """Reject overworld trash deaths / grace / map as 'boss clips'.

    Agent review of Merica: most FPs were ordinary combat or YOU DIED without a
    real boss name/dual HUD. Fire Giant stays via long bar-only fights.
    """
    if not run:
        return False
    if max(_dual_score(s) for s in run) >= 0.50:
        return True
    named = [
        s
        for s in run
        if s.boss_name_score >= 0.35
        and s.boss_bar_score >= 0.50
        and 0.25 <= s.boss_bar_fill <= 0.98
    ]
    if len(named) >= 3:
        return True
    # Fire Giant / dark arenas — no readable name, but long sustained bar.
    return duration >= 40.0 and max(s.boss_bar_score for s in run) >= 0.70


def _refine_onset_for_dual(run: list[FrameSignals], onset: float) -> float:
    """If dual bars appear later, don't keep elite/trash red as HUD onset.

    Requires a *sustained* dual stack (Merica Tree Sentinels). A brief
    boss+summon pair (Fire Giant + Alexander) must not jump onset forward.
    """
    dual_times = [s.time_sec for s in run if _dual_score(s) >= 0.50]
    if len(dual_times) < 6:
        return onset
    dual_onset = float(dual_times[0])
    # Dual should also dominate a short window (not a one-off summon flash).
    early = [t for t in dual_times if t <= dual_onset + 20.0]
    if len(early) < 4:
        return onset
    if dual_onset - onset >= 12.0:
        return dual_onset
    return onset


def _hud_onset_before(
    ordered: list[FrameSignals],
    terminal_t: float,
    look_from: float,
    tuning: EldenTuning,
    *,
    bridge_sec: float = 120.0,
) -> float | None:
    """Boss-bar onset for the attempt that ends at ``terminal_t``.

    Bridge long mid-fight HUD gaps (phase cinematics / Fire Giant tracking
    misses) unless a prior terminal sits in the gap.
    """
    bar_hits = [
        s
        for s in ordered
        if look_from <= s.time_sec < terminal_t - 1.0
        and (
            _is_hud_hit(s, tuning)
            or _boss_hud_present(s, tuning)
            or _can_open_segment(s, tuning)
        )
    ]
    if not bar_hits:
        return None
    # Only YOU DIED splits an attempt during walk-back. Gold FPs are common in
    # fire arenas and must not chop the onset back to a late HUD fragment.
    death_times = {
        t
        for t, kind, _score in collect_terminal_events(
            [s for s in ordered if look_from <= s.time_sec < terminal_t - 2.0]
        )
        if kind == "death"
    }

    def _can_bridge(
        run: list[FrameSignals], prev_t: float, next_s: FrameSignals
    ) -> bool:
        gap = next_s.time_sec - prev_t
        if gap <= 18.0:
            return True
        if gap > bridge_sec:
            return False
        if any(prev_t <= t < next_s.time_sec for t in death_times):
            return False
        # Merica: don't glue a 1s outdoor red blip (~425) across ~30s to dual HUD.
        if not _run_has_proven_hud(run):
            return False
        dual_peak = max(_dual_score(r) for r in run)
        if (
            gap > 15.0
            and dual_peak >= 0.55
            and _dual_score(next_s) < 0.45
            and next_s.boss_name_score < 0.20
        ):
            return False
        return True

    last_run: list[FrameSignals] = [bar_hits[0]]
    for s in bar_hits[1:]:
        if _can_bridge(last_run, last_run[-1].time_sec, s):
            last_run.append(s)
        else:
            last_run = [s]
    return float(last_run[0].time_sec)


def _find_hud_runs(
    ordered: list[FrameSignals],
    tuning: EldenTuning,
    *,
    gap_sec: float = 8.0,
    phase_bridge_sec: float = 120.0,
) -> list[tuple[float, float, list[FrameSignals]]]:
    """Contiguous stretches where the boss bar / HUD is present.

    Ends a run when the bar stays empty for a few seconds (0717: bar dies ~54s,
    ENEMY FELLED ~56s). Mid-fight gaps without YOU DIED (phase-2 cinematic /
    tracking misses on 0717(1)) stay bridged up to ``phase_bridge_sec``.
    """
    hits = [s for s in ordered if _is_hud_hit(s, tuning)]
    if not hits:
        return []
    death_times = [
        t for t, kind, _s in collect_terminal_events(ordered) if kind == "death"
    ]

    def _gap_has_death(t0: float, t1: float) -> bool:
        # Inclusive on t0: the death frame itself is often still a HUD hit, so
        # the gap after it must still split the run (Malenia phase-1 → phase-2).
        return any(t0 <= t < t1 for t in death_times)

    def _bar_looks_emptied(run: list[FrameSignals]) -> bool:
        tail = run[-min(6, len(run)) :]
        fills = [s.boss_bar_fill for s in tail if s.boss_bar_score >= 0.28]
        if not fills:
            return False
        # Only treat as emptied if the bar also stays gone afterward.
        t_end = tail[-1].time_sec
        if _boss_bar_returns_after(ordered, t_end, within_sec=12.0):
            return False
        return float(np.median(fills)) <= 0.18

    runs: list[list[FrameSignals]] = [[hits[0]]]
    for s in hits[1:]:
        prev = runs[-1][-1]
        gap = s.time_sec - prev.time_sec
        # YOU DIED often keeps the boss bar visible, so the post-death HUD hit
        # is only 2s later — still must start a new attempt run.
        if _gap_has_death(prev.time_sec, s.time_sec):
            runs.append([s])
            continue
        if gap <= gap_sec:
            runs[-1].append(s)
            continue
        run = runs[-1]
        # Dual-boss flee (Merica Tree Sentinels): after the stacked HUD leaves,
        # do not phase-bridge into later outdoor red FX / map gold "FELLED".
        # Short gaps still chain single-bar leftovers from the same fight.
        dual_peak = max(_dual_score(r) for r in run)
        if (
            gap > 15.0
            and dual_peak >= 0.55
            and _dual_score(s) < 0.45
            and s.boss_name_score < 0.20
        ):
            runs.append([s])
            continue
        if (
            gap <= phase_bridge_sec
            and not _bar_looks_emptied(run)
            and _run_has_proven_hud(run)
        ):
            runs[-1].append(s)
            continue
        runs.append([s])
    out: list[tuple[float, float, list[FrameSignals]]] = []
    for run in runs:
        if len(run) < max(2, tuning.min_hud_samples - 1):
            continue
        if not _run_has_proven_hud(run) and max(_dual_score(s) for s in run) < 0.50:
            # Drop brief bar-only FX clusters (Merica spell wash ~425–426).
            if run[-1].time_sec - run[0].time_sec < 12.0:
                continue
        peak_bar = max(s.boss_bar_score for s in run)
        if peak_bar < tuning.start_bar_min_score and max(_dual_score(s) for s in run) < 0.50:
            continue
        onset = run[0].time_sec
        # Skip leading LIKES/chrome / red FX before a real named or dual HUD.
        proven = [
            s.time_sec
            for s in run
            if _dual_score(s) >= 0.50
            or (
                s.boss_name_score >= 0.35
                and s.boss_bar_score >= 0.55
                and 0.25 <= s.boss_bar_fill <= 0.98
            )
        ]
        if proven and proven[0] - onset >= 6.0:
            onset = float(proven[0])
            run = [s for s in run if s.time_sec >= onset]
            if len(run) < 2:
                continue
        # Trim only on kill-like empties (bar was low). Mid-fight cinematic HUD
        # gaps (0717(1) phase 2) keep the run open even if the bar vanishes.
        run_end = run[-1].time_sec
        for s in run:
            if s.boss_bar_score < 0.22:
                continue
            recent = [
                r
                for r in run
                if s.time_sec - 6.0 <= r.time_sec <= s.time_sec and r.boss_bar_score >= 0.28
            ]
            recent_fill = (
                float(np.median([r.boss_bar_fill for r in recent])) if recent else 1.0
            )
            nxt = [
                p
                for p in ordered
                if s.time_sec < p.time_sec <= s.time_sec + 4.0
            ]
            if (
                recent_fill <= 0.22
                and len(nxt) >= 3
                and all(p.boss_bar_score < 0.22 for p in nxt)
            ):
                run_end = s.time_sec
                run = [r for r in run if r.time_sec <= run_end]
                break
            run_end = s.time_sec
        if run_end - onset < 4.0:
            continue
        out.append((onset, run_end, run))
    return out


def _first_score_peak(
    window: list[FrameSignals],
    score_fn,
    threshold: float,
) -> FrameSignals | None:
    hits = [s for s in window if score_fn(s) >= threshold]
    if not hits:
        return None
    first_t = hits[0].time_sec
    local = [s for s in hits if s.time_sec <= first_t + 3.0]
    return max(local, key=lambda s: (score_fn(s), s.time_sec))


def _boss_bar_returns_after(
    ordered: list[FrameSignals],
    t: float,
    *,
    within_sec: float = 12.0,
    bar_min: float = 0.34,
) -> bool:
    """True if the boss HUD comes back soon — tracking gap, not a real kill.

    Window is tight enough that post-kill loot UI flashes on 0717 (~69s after a
    56s FELLED) do not count, but Fire Giant mid-fight gaps (~11s) still do.
    """
    after = [
        s
        for s in ordered
        if t < s.time_sec <= t + within_sec and s.boss_bar_score >= bar_min
    ]
    return bool(after)


def _victory_after_bar_drop(
    ordered: list[FrameSignals],
    run_end: float,
    *,
    onset: float | None = None,
    hunt_sec: float = 8.0,
) -> tuple[float, str, float] | None:
    """Kill path only: boss bar emptied → ENEMY FELLED / GREAT ENEMY FELLED.

    Search from attempt onset (not only run_end): post-kill loot can fake a boss
    bar and push run_end past the real banner (0717). Mid-fight Fire Giant gaps
    that lose the bar are rejected when the HUD returns within ~12s. 0717(2):
    bar tracking drops while HP is low, then FELLED arrives ~15s later — extend
    the hunt when the bar was draining.
    """
    t0 = (onset + 5.0) if onset is not None else max(0.0, run_end - 0.5)
    hunt = hunt_sec
    # Use the last bar samples — earlier full-HP frames inflate a long median.
    tail = [
        s
        for s in ordered
        if run_end - 8.0 <= s.time_sec <= run_end and s.boss_bar_score >= 0.18
    ]
    if tail:
        fills = [s.boss_bar_fill for s in tail]
        last_fills = fills[-min(4, len(fills)) :]
        if float(np.median(last_fills)) <= 0.55 or min(fills) <= 0.40:
            # 0717(2): bar tracking drops ~34s, FELLED ~49s.
            hunt = max(hunt, 24.0)
    window = [
        s
        for s in ordered
        if t0 <= s.time_sec <= run_end + hunt and s.boss_bar_score < 0.12
    ]
    if not window:
        return None
    # Try victory peaks in time order; first confirmed kill wins.
    peaks = sorted(
        [s for s in window if s.victory_screen_score >= 0.45],
        key=lambda s: s.time_sec,
    )
    seen: set[float] = set()
    for s in peaks:
        bucket = round(s.time_sec / 3.0)
        if bucket in seen:
            continue
        seen.add(bucket)
        local = [p for p in peaks if abs(p.time_sec - s.time_sec) <= 3.0]
        best = max(local, key=lambda p: p.victory_screen_score)
        pre = [
            p
            for p in ordered
            if best.time_sec - 8.0 <= p.time_sec < best.time_sec
            and p.boss_bar_score >= 0.34
        ]
        if not pre:
            # Allow a slightly longer lookback when the bar drained slowly.
            pre = [
                p
                for p in ordered
                if best.time_sec - 20.0 <= p.time_sec < best.time_sec
                and p.boss_bar_score >= 0.34
            ]
        if not pre:
            continue
        if _boss_bar_returns_after(ordered, best.time_sec):
            continue
        # Dual-boss kills should show a drained bar; Merica outdoor gold often
        # fires FELLED while both Tree Sentinels were still healthy / fled.
        if onset is not None:
            dual_peak = max(
                (
                    _dual_score(p)
                    for p in ordered
                    if onset <= p.time_sec <= best.time_sec
                ),
                default=0.0,
            )
            if dual_peak >= 0.55:
                recent_fills = [p.boss_bar_fill for p in pre[-6:]]
                if recent_fills and float(np.median(recent_fills)) > 0.42:
                    continue
        return best.time_sec, "victory", best.victory_screen_score
    return None


def _death_during_attempt(
    ordered: list[FrameSignals],
    onset: float,
    run_end: float,
) -> tuple[float, str, float] | None:
    """Fail path: YOU DIED while the boss bar may still be full or partial.

    Only accept a *fresh* death peak (was low, then spikes) so dark-arena combat
    that dribbles a mid death score for minutes does not end the attempt early.
    """
    window = [
        s
        for s in ordered
        if onset + 3.0 <= s.time_sec <= run_end + 8.0
    ]
    fresh: list[FrameSignals] = []
    for s in window:
        if s.death_screen_score < 0.55:
            continue
        prev = [
            p
            for p in ordered
            if s.time_sec - 2.5 <= p.time_sec < s.time_sec
        ]
        if prev and max(p.death_screen_score for p in prev) >= 0.40:
            continue
        fresh.append(s)
    if not fresh:
        return None
    best_d = _first_score_peak(fresh, lambda s: s.death_screen_score, 0.55)
    if best_d is None:
        return None
    return best_d.time_sec, "death", best_d.death_screen_score


def _resolve_attempt_terminal(
    ordered: list[FrameSignals],
    onset: float,
    run_end: float,
) -> tuple[float, str, float] | None:
    """Two endings — do not require bar-empty for death.

    - YOU DIED: can fire mid-fight (bar full / low / anything)
    - ENEMY FELLED: only after the boss bar empties

    Prefer YOU DIED when present in the attempt. Mid-fight gold/fire often
    false-triggers ENEMY FELLED before a real death (0717(1) Fire Giant).
    """
    death = _death_during_attempt(ordered, onset, run_end)
    if death is not None:
        return death
    return _victory_after_bar_drop(ordered, run_end, onset=onset)


def _preroll_start(
    ordered: list[FrameSignals],
    onset: float,
    prev_close: float,
    tuning: EldenTuning,
) -> float:
    """Start 5–30s before HUD onset; prefer the closest quiet window.

    User target: begin shortly before the fight (closer is better), not minutes
    of exploration. Default ~8–12s matches mist-door / enter-room goldens.
    """
    max_preroll = min(max(float(tuning.cutscene_preroll_sec), 5.0), 30.0)
    # Try closer first, then expand if the short window is not quiet enough.
    candidates = [8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    start = onset
    for preroll in candidates:
        if preroll > max_preroll + 0.01:
            break
        pre = [s for s in ordered if onset - preroll <= s.time_sec < onset - 0.4]
        if not pre:
            if preroll <= 12.0:
                start = max(0.0, prev_close + 0.1, onset - preroll)
            continue
        quiet = sum(
            1
            for s in pre
            if s.boss_bar_score < 0.22 and _dual_score(s) < 0.40
        ) / len(pre)
        if quiet >= 0.55:
            start = max(0.0, prev_close + 0.1, onset - preroll)
            break
    return start


def cluster_attempts_terminal_first(
    signals: list[FrameSignals],
    *,
    tuning: EldenTuning | None = None,
    max_lookback_sec: float = 240.0,
) -> list[tuple[float, float, dict]]:
    """One clip per attempt using the 0717 lifecycle (+ failed attempts).

    enter preroll → boss HUD onset → (YOU DIED anytime | bar empty → FELLED) → hold.
    Primary path follows HUD runs (not every stray gold/red frame).
    """
    tuning = (tuning or EldenTuning.defaults()).apply_builtin_attempt_policy()
    if not signals:
        return []

    ordered = sorted(signals, key=lambda s: s.time_sec)
    runs = _find_hud_runs(ordered, tuning)
    out: list[tuple[float, float, dict]] = []
    prev_close = -1.0

    for onset, run_end, run in runs:
        if onset < prev_close:
            continue
        duration = run_end - onset
        if duration < max(5.0, tuning.min_fight_sec * 0.4):
            continue
        # Drop overworld deaths / map / trash combat that never look like a boss.
        if not _run_is_boss_attempt(run, duration=duration):
            continue
        onset = _refine_onset_for_dual(run, onset)
        duration = run_end - onset
        if duration < max(5.0, tuning.min_fight_sec * 0.4):
            continue

        terminal = _resolve_attempt_terminal(ordered, onset, run_end)
        if terminal is None:
            # Fallback: classic terminal-first inside the run window.
            local = [
                s
                for s in ordered
                if onset - 1.0 <= s.time_sec <= run_end + 12.0
            ]
            events = collect_terminal_events(local)
            events = [e for e in events if e[0] >= onset + 3.0]
            # Gold/fire FPs often look like ENEMY FELLED while the bar is only
            # briefly missing — drop victories where the HUD returns.
            events = [
                e
                for e in events
                if not (e[1] == "victory" and _boss_bar_returns_after(ordered, e[0]))
            ]
            deaths = [e for e in events if e[1] == "death"]
            if deaths:
                terminal = deaths[0]
            elif events:
                terminal = events[0]
            else:
                # Incomplete/no-banner runs are almost always hub/explore junk on
                # long VODs (Merica scan2). Keep only hard terminals.
                continue
        mt, kind, term_score = terminal
        if mt < onset + 3.0:
            continue

        paired = [s for s in run if _boss_hud_present(s, tuning)]
        evidence = paired if paired else run
        peak_bar = max(s.boss_bar_score for s in evidence)
        peak_fill = max(s.boss_bar_fill for s in run)
        peak_name = max((s.boss_name_score for s in paired), default=0.0)
        peak_dual = max(_dual_score(s) for s in run)

        start = _preroll_start(ordered, onset, prev_close, tuning)

        if kind == "death":
            hold = tuning.death_hold_sec
            end_kind = "you_died"
        elif kind == "victory":
            hold = tuning.victory_hold_sec
            end_kind = "enemy_felled"
        else:
            hold = 2.0
            end_kind = "incomplete"
        attempt_end = mt + hold
        if attempt_end - start < tuning.min_keep_sec:
            continue

        meta = {
            "peak_boss_bar_score": round(peak_bar, 3),
            "peak_boss_name_score": round(peak_name, 3),
            "peak_boss_bar_fill": round(peak_fill, 3),
            "peak_dual_boss_bar_score": round(peak_dual, 3),
            "had_death_screen": kind == "death",
            "had_victory_screen": kind == "victory",
            "frame_hits": len(evidence),
            "avg_boss_score": round(
                sum(s.combined_boss_score() for s in evidence) / max(1, len(evidence)),
                3,
            ),
            "terminal_score": round(term_score, 3),
            "ml_end_kind": end_kind,
            "detector": "lifecycle_0717",
            "hud_onset_sec": round(onset, 2),
        }
        out.append((round(start, 2), round(attempt_end, 2), meta))
        prev_close = attempt_end + 1.0

    if out:
        return _dedupe_segments(out, iou_threshold=0.35)

    # Last-resort: old terminal-first walk-back (rare VODs with weak bar tracking).
    terminals = collect_terminal_events(ordered)
    for mt, kind, term_score in terminals:
        if kind == "victory" and _boss_bar_returns_after(ordered, mt):
            continue
        look_from = max(0.0, mt - max_lookback_sec)
        onset = _hud_onset_before(ordered, mt, look_from, tuning)
        if onset is None or mt - onset < 5.0:
            continue
        hold = tuning.death_hold_sec if kind == "death" else tuning.victory_hold_sec
        meta = {
            "peak_boss_bar_score": 0.0,
            "peak_boss_name_score": 0.0,
            "had_death_screen": kind == "death",
            "had_victory_screen": kind == "victory",
            "frame_hits": 0,
            "avg_boss_score": 0.0,
            "terminal_score": round(term_score, 3),
            "ml_end_kind": "you_died" if kind == "death" else "enemy_felled",
            "detector": "terminal_fallback",
            "hud_onset_sec": round(onset, 2),
        }
        out.append((round(onset, 2), round(mt + hold, 2), meta))
    return _dedupe_segments(out, iou_threshold=0.35)


@dataclass
class EldenTuning:
    """Learned detection thresholds from supervised labels."""

    min_boss_name_score: float = 0.22
    min_boss_bar_score: float = 0.30
    require_name_with_bar: bool = True
    # Drop clips that never hit YOU DIED / ENEMY FELLED (unless fight still in progress).
    require_terminal: bool = True
    bar_without_name_reject: float = 0.42
    min_name_when_bar_high: float = 0.18
    min_bar_when_name_high: float = 0.28
    frame_hit_threshold: float = 0.28
    min_frame_hits: int = 4
    trim_at_victory: bool = True
    victory_hold_sec: float = 3.0
    max_post_victory_sec: float = 8.0
    pre_buffer_sec: float = 2.0
    start_name_threshold: float = 0.85
    retry_bridge_sec: float = 150.0
    merge_gap_sec: float = 90.0
    # One clip per attempt (bar appears → YOU DIED / ENEMY FELLED).
    split_per_attempt: bool = True
    merge_while_bar_visible: bool = True
    death_hold_sec: float = 4.0
    bar_gone_sec: float = 8.0
    # Bridge brief HUD flicker within one attempt (not across deaths).
    hud_bridge_sec: float = 14.0
    min_hud_samples: int = 4
    onset_quiet_sec: float = 10.0
    onset_quiet_name_max: float = 0.4
    start_bar_lookback_sec: float = 90.0
    start_bar_min_score: float = 0.36
    full_bar_min_score: float = 0.48
    full_bar_fill_min: float = 0.70
    cutscene_preroll_sec: float = 20.0
    end_on_death: bool = True
    min_fight_sec: float = 12.0
    min_clip_sec: float = 12.0
    # Minimum span to keep a candidate at all (short noise filter).
    min_keep_sec: float = 10.0
    # 0 = unlimited clip length (no cap on how long a clip can run).
    max_clip_sec: float = 0.0
    # Learned from human timing corrections: shift auto start/end by these seconds.
    start_offset_sec: float = 0.0
    end_offset_sec: float = 0.0
    learned_from: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EldenTuning:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def defaults(cls) -> EldenTuning:
        return cls()

    def apply_builtin_attempt_policy(self) -> EldenTuning:
        """Hard-coded Elden Ring rules from golden clips 0717 + 0717(1).

        Lifecycle of one keepable attempt clip:
          1. Optional mist-door / enter-room preroll (no boss bar)
          2. Boss HUD onset — red bar (+ name) appears in gameplay third
          3. Fight while HUD is up — phase-2 cinematics may hide HUD temporarily
          4. Terminal (two different endings):
             - YOU DIED — failed attempt; boss bar may still be full or partial
             - ENEMY FELLED / GREAT ENEMY FELLED — kill; only after bar empties
          5. End shortly after the banner (~death_hold / victory_hold)
        """
        self.split_per_attempt = True
        self.end_on_death = True
        self.trim_at_victory = True
        self.merge_while_bar_visible = True
        self.require_name_with_bar = True
        self.require_terminal = True
        self.max_clip_sec = 0.0
        # Precision floors (feedback may raise further; cannot drift below these).
        self.min_boss_name_score = max(self.min_boss_name_score, 0.20)
        self.min_boss_bar_score = max(self.min_boss_bar_score, 0.28)
        self.start_bar_min_score = max(self.start_bar_min_score, 0.34)
        self.min_name_when_bar_high = max(self.min_name_when_bar_high, 0.16)
        self.min_bar_when_name_high = max(self.min_bar_when_name_high, 0.26)
        self.bar_without_name_reject = min(max(self.bar_without_name_reject, 0.35), 0.45)
        self.min_hud_samples = max(self.min_hud_samples, 3)
        self.min_keep_sec = max(self.min_keep_sec, 8.0)
        self.min_fight_sec = max(min(self.min_fight_sec, 20.0), 8.0)
        self.min_clip_sec = max(min(self.min_clip_sec, 20.0), 8.0)
        self.hud_bridge_sec = min(max(self.hud_bridge_sec, 8.0), 16.0)
        # Quiet lead-in 5–30s before HUD (prefer ~8–12s; closer is better).
        self.cutscene_preroll_sec = min(max(self.cutscene_preroll_sec, 8.0), 30.0)
        self.death_hold_sec = min(max(self.death_hold_sec, 3.0), 5.0)
        self.victory_hold_sec = min(max(self.victory_hold_sec, 3.0), 5.0)
        return self


def frame_passes_tuning(sig: FrameSignals, tuning: EldenTuning) -> bool:
    """Per-frame gate learned from human labels."""
    bar = sig.boss_bar_score
    name = sig.boss_name_score
    if bar >= tuning.bar_without_name_reject and name < tuning.min_name_when_bar_high:
        return False
    if bar >= 0.35 and name < tuning.min_boss_name_score:
        return False
    if name >= 0.85 and bar < tuning.min_bar_when_name_high:
        return False
    if name >= tuning.min_boss_name_score:
        return True
    if bar >= tuning.min_boss_bar_score and name >= tuning.min_name_when_bar_high:
        return True
    return sig.combined_boss_score() >= tuning.frame_hit_threshold and name > 0.05


def _find_attempt_onset(
    all_signals: list[FrameSignals],
    seg_start: float,
    seg_end: float,
    tuning: EldenTuning,
) -> float:
    """Find when a boss attempt actually begins — after quiet period or death."""
    window = sorted(
        [s for s in all_signals if seg_start - 30.0 <= s.time_sec <= seg_end + 2.0],
        key=lambda s: s.time_sec,
    )
    if not window:
        return max(0.0, seg_start - tuning.pre_buffer_sec)

    quiet_gap = tuning.onset_quiet_sec
    quiet_name = tuning.onset_quiet_name_max
    onset_candidates: list[float] = []

    for s in window:
        if s.time_sec < seg_start - 1.0:
            continue
        bar_hit = s.boss_bar_score >= max(tuning.start_bar_min_score, 0.35)
        name_hit = s.boss_name_score >= tuning.start_name_threshold
        if not bar_hit and not name_hit:
            continue
        lookback = [p for p in window if s.time_sec - quiet_gap - 6 <= p.time_sec <= s.time_sec - 4]
        if not lookback:
            continue
        quiet_hits = sum(
            1 for p in lookback if p.boss_name_score < quiet_name and p.boss_bar_score < 0.35
        )
        if quiet_hits >= max(2, int(len(lookback) * 0.55)):
            onset_candidates.append(s.time_sec)

    if onset_candidates:
        markers = [
            s.time_sec
            for s in window
            if seg_start <= s.time_sec <= seg_end + 5
            and (s.death_screen_score >= 0.35 or s.victory_screen_score >= 0.35)
        ]
        for onset in reversed(onset_candidates):
            tail = [s for s in window if onset <= s.time_sec <= onset + 180]
            if not tail:
                continue
            if any(
                s.death_screen_score >= 0.35 or s.victory_screen_score >= 0.35 for s in tail
            ):
                return max(0.0, onset - tuning.pre_buffer_sec)
        return max(0.0, onset_candidates[-1] - tuning.pre_buffer_sec)

    deaths = [s.time_sec for s in window if s.death_screen_score >= 0.35]
    for dt in deaths:
        after = [
            s
            for s in window
            if dt + 5.0 <= s.time_sec <= dt + 120.0
            and s.boss_name_score >= tuning.min_boss_name_score
        ]
        if after:
            return max(0.0, after[0].time_sec - tuning.pre_buffer_sec)

    for s in window:
        if s.time_sec < seg_start:
            continue
        if (
            s.boss_name_score >= tuning.min_boss_name_score
            and s.boss_bar_score >= tuning.min_boss_bar_score
        ):
            return max(0.0, s.time_sec - tuning.pre_buffer_sec)

    for s in window:
        if s.time_sec >= seg_start and s.boss_name_score >= tuning.min_boss_name_score:
            return max(0.0, s.time_sec - tuning.pre_buffer_sec)
    return max(0.0, seg_start - tuning.pre_buffer_sec)


def _boss_hud_present(sig: FrameSignals, tuning: EldenTuning) -> bool:
    """Boss name + red health bar both visible (Elden Ring HUD)."""
    if _dual_score(sig) >= 0.55 and sig.boss_bar_score >= tuning.min_boss_bar_score:
        return True
    min_name = min(tuning.min_boss_name_score, 0.65)
    return (
        sig.boss_name_score >= min_name
        and sig.boss_bar_score >= tuning.min_boss_bar_score
    )


def _can_open_segment(sig: FrameSignals, tuning: EldenTuning) -> bool:
    """Start a fight window only when bar+name look like a real boss HUD."""
    if _boss_hud_present(sig, tuning):
        return True
    if _dual_score(sig) >= 0.50 and sig.boss_bar_score >= tuning.start_bar_min_score:
        return True
    if not tuning.require_name_with_bar:
        return sig.boss_bar_score >= tuning.start_bar_min_score
    # Reject red scenery / fire — need some name-band text with the bar.
    name_floor = max(0.10, tuning.min_boss_name_score * 0.55)
    return (
        sig.boss_bar_score >= tuning.start_bar_min_score
        and sig.boss_name_score >= name_floor
    )


def _has_strong_terminal(
    signals: list[FrameSignals],
    start: float,
    end: float,
    *,
    threshold: float = 0.50,
) -> tuple[bool, bool]:
    """Death / victory with a stricter threshold (cuts cinematic gold/red noise)."""
    had_death = False
    had_victory = False
    for s in signals:
        if not (start <= s.time_sec <= end + 1.0):
            continue
        if s.death_screen_score >= threshold:
            had_death = True
        if s.victory_screen_score >= threshold:
            had_victory = True
    return had_death, had_victory


def _is_credible_boss_clip(
    *,
    peak_bar: float,
    peak_name: float,
    frame_hits: int,
    had_death: bool,
    had_victory: bool,
    duration_sec: float,
    tuning: EldenTuning,
) -> bool:
    """Final gate: real named HUD, and usually a death/victory terminal."""
    if duration_sec < tuning.min_keep_sec:
        return False
    if tuning.require_name_with_bar and peak_name < tuning.min_boss_name_score:
        return False
    if peak_bar < tuning.min_boss_bar_score and peak_name < tuning.start_name_threshold:
        return False
    if peak_bar >= tuning.bar_without_name_reject and peak_name < tuning.min_name_when_bar_high:
        return False
    if frame_hits < tuning.min_hud_samples:
        # Allow a short but very clear named bar + terminal.
        if not (
            (had_death or had_victory)
            and peak_name >= max(0.35, tuning.min_boss_name_score)
            and peak_bar >= tuning.min_boss_bar_score
            and frame_hits >= 2
        ):
            return False
    if tuning.require_terminal and not (had_death or had_victory):
        # In-progress fight only if HUD is strong and sustained.
        if peak_name < 0.35 or frame_hits < max(5, tuning.min_hud_samples + 1):
            return False
    return True


def _boss_hud_engaged(
    sig: FrameSignals,
    tuning: EldenTuning,
    *,
    saw_paired: bool,
    last_paired_time: float,
) -> bool:
    if _boss_hud_present(sig, tuning):
        return True
    if not saw_paired:
        return False
    if sig.time_sec - last_paired_time > tuning.hud_bridge_sec:
        return False
    if sig.boss_name_score >= tuning.min_boss_name_score * 0.75:
        return True
    if sig.boss_bar_score >= 0.35:
        return sig.boss_name_score >= tuning.min_name_when_bar_high
    return False


def _boss_hud_gone(sig: FrameSignals) -> bool:
    return sig.boss_name_score < 0.35 and sig.boss_bar_score < 0.12


def _strong_death(sig: FrameSignals, *, threshold: float = 0.5) -> bool:
    return sig.death_screen_score >= threshold


def _attempt_end_victory(
    sig: FrameSignals,
    *,
    last_paired: float,
    frames: list[FrameSignals],
    tuning: EldenTuning,
) -> bool:
    if not _strong_victory(sig, threshold=0.45):
        return False
    if sig.time_sec < last_paired - 2.0:
        return False
    if _recent_paired(frames, sig.time_sec, tuning, within_sec=10.0):
        return False
    return sig.time_sec >= last_paired - 1.0


def _attempt_end_death(
    sig: FrameSignals,
    *,
    last_paired: float,
    frames: list[FrameSignals],
    tuning: EldenTuning,
) -> bool:
    if not _strong_death(sig, threshold=0.45):
        return False
    if sig.time_sec < last_paired - 5.0:
        return False
    if _recent_paired(frames, sig.time_sec, tuning, within_sec=4.0):
        return False
    return True


def _strong_victory(sig: FrameSignals, *, threshold: float = 0.5) -> bool:
    return sig.victory_screen_score >= threshold


def _recent_paired(
    frames: list[FrameSignals],
    since: float,
    tuning: EldenTuning,
    *,
    within_sec: float = 6.0,
) -> bool:
    return any(
        f.time_sec >= since - within_sec
        and f.time_sec <= since + within_sec
        and _boss_hud_present(f, tuning)
        for f in frames
    )


def _cluster_by_boss_hud(
    signals: list[FrameSignals],
    tuning: EldenTuning,
) -> list[tuple[float, float, list[FrameSignals]]]:
    """Cluster boss HUD into fight windows.

    When ``split_per_attempt`` is on, each YOU DIED / ENEMY FELLED closes the
    current window so every attempt becomes its own segment.
    """
    if not signals:
        return []

    segments: list[tuple[float, float, list[FrameSignals]]] = []
    i = 0
    n = len(signals)

    while i < n:
        while i < n and not _can_open_segment(signals[i], tuning):
            i += 1
        if i >= n:
            break

        seg_frames: list[FrameSignals] = [signals[i]]
        seg_open = signals[i].time_sec
        saw_paired = _boss_hud_present(signals[i], tuning)
        last_paired_time = signals[i].time_sec if saw_paired else signals[i].time_sec - 1.0
        last_victory_time: float | None = None
        i += 1
        gone_start: float | None = None

        while i < n:
            sig = signals[i]

            # Per-attempt mode: death or kill ends this clip after a short hold.
            if tuning.split_per_attempt and (
                _strong_death(sig, threshold=0.50)
                or (
                    _strong_victory(sig, threshold=0.55)
                    and saw_paired
                    and not _recent_paired(seg_frames, sig.time_sec, tuning, within_sec=8.0)
                )
            ):
                if sig.time_sec - seg_open >= max(5.0, tuning.min_fight_sec * 0.5):
                    kind = "death" if _strong_death(sig, threshold=0.50) else "victory"
                    hold = (
                        tuning.death_hold_sec if kind == "death" else tuning.victory_hold_sec
                    )
                    terminal_t = sig.time_sec
                    seg_frames.append(sig)
                    i += 1
                    while i < n and signals[i].time_sec <= terminal_t + hold + 1.0:
                        seg_frames.append(signals[i])
                        i += 1
                    break

            if _strong_victory(sig, threshold=0.55):
                if saw_paired and not _recent_paired(seg_frames, sig.time_sec, tuning, within_sec=12.0):
                    last_victory_time = sig.time_sec

            if _boss_hud_engaged(
                sig,
                tuning,
                saw_paired=saw_paired,
                last_paired_time=last_paired_time,
            ):
                if _boss_hud_present(sig, tuning):
                    saw_paired = True
                    last_paired_time = sig.time_sec
                    last_victory_time = None
                seg_frames.append(sig)
                gone_start = None
                i += 1
                continue

            if sig.victory_screen_score >= 0.45 and saw_paired:
                seg_frames.append(sig)
                if _strong_victory(sig, threshold=0.55) and not _recent_paired(
                    seg_frames, sig.time_sec, tuning, within_sec=12.0
                ):
                    last_victory_time = sig.time_sec
                gone_start = None
                i += 1
                continue

            if last_victory_time is not None:
                after_win = sig.time_sec - last_victory_time
                still_fighting = _recent_paired(seg_frames, sig.time_sec, tuning)
                if (
                    after_win >= tuning.victory_hold_sec
                    and not still_fighting
                    and (_boss_hud_gone(sig) or after_win >= tuning.max_post_victory_sec)
                ):
                    break
                if after_win >= tuning.max_post_victory_sec and not still_fighting:
                    break

            if gone_start is None:
                gone_start = sig.time_sec
            if _boss_hud_gone(sig):
                if sig.time_sec - last_paired_time <= tuning.hud_bridge_sec:
                    gone_start = None
                elif sig.time_sec - gone_start >= tuning.bar_gone_sec:
                    break
            else:
                gone_start = None
            seg_frames.append(sig)
            i += 1

        paired = [f for f in seg_frames if _boss_hud_present(f, tuning)]
        if len(paired) < tuning.min_hud_samples:
            strong_name = any(
                f.boss_name_score >= 0.85 and f.boss_bar_score >= tuning.min_boss_bar_score
                for f in paired
            )
            if len(paired) < 2 or not strong_name:
                continue
        peak_name = max((f.boss_name_score for f in paired), default=0.0)
        peak_bar = max((f.boss_bar_score for f in paired), default=0.0)
        if tuning.require_name_with_bar and peak_name < tuning.min_boss_name_score:
            continue
        if peak_name < tuning.min_boss_name_score * 0.9 and peak_bar < 0.45:
            continue
        if peak_bar >= tuning.bar_without_name_reject and peak_name < tuning.min_name_when_bar_high:
            continue
        if len(paired) >= 2:
            paired_span = paired[-1].time_sec - paired[0].time_sec
            if paired_span < 4.0 and peak_name < 0.45:
                continue
        span = seg_frames[-1].time_sec - seg_frames[0].time_sec
        if span < tuning.min_keep_sec and len(paired) < max(4, tuning.min_hud_samples):
            continue
        segments.append((seg_frames[0].time_sec, seg_frames[-1].time_sec, seg_frames))

    return segments


def _merge_hud_segments(
    segments: list[tuple[float, float, list[FrameSignals]]],
    *,
    max_gap_sec: float = 50.0,
    split_per_attempt: bool = False,
) -> list[tuple[float, float, list[FrameSignals]]]:
    """Merge nearby HUD segments — same attempt with brief bar/name flicker.

    When splitting per attempt, never glue a death/victory ending into the next try.
    """
    if len(segments) < 2:
        return segments
    segments = sorted(segments, key=lambda s: s[0])
    merged: list[tuple[float, float, list[FrameSignals]]] = [segments[0]]
    for start, end, frames in segments[1:]:
        ps, pe, pframes = merged[-1]
        gap = start - pe
        prev_had_victory = any(_strong_victory(f, threshold=0.55) for f in pframes)
        prev_had_death = any(_strong_death(f, threshold=0.50) for f in pframes)
        if split_per_attempt and (prev_had_death or prev_had_victory):
            merged.append((start, end, frames))
            continue
        if gap <= max_gap_sec and not (prev_had_victory and gap > 12.0):
            merged[-1] = (ps, max(pe, end), pframes + frames)
        else:
            merged.append((start, end, frames))
    return merged


def _compute_clip_end(
    start: float,
    frames: list[FrameSignals],
    all_signals: list[FrameSignals],
    tuning: EldenTuning,
) -> float:
    """End on first YOU DIED or boss kill — short reaction tail only."""
    paired = [f for f in frames if _boss_hud_present(f, tuning)]
    last_paired = max((f.time_sec for f in paired), default=frames[-1].time_sec)
    scan_until = max(frames[-1].time_sec, last_paired) + 45.0
    window = [s for s in all_signals if start <= s.time_sec <= scan_until]

    deaths = [
        s.time_sec
        for s in window
        if _attempt_end_death(s, last_paired=last_paired, frames=frames, tuning=tuning)
        and s.time_sec >= start + max(4.0, tuning.min_fight_sec * 0.4)
    ]
    victories = [
        s.time_sec
        for s in window
        if _attempt_end_victory(s, last_paired=last_paired, frames=frames, tuning=tuning)
        and s.time_sec >= start + max(4.0, tuning.min_fight_sec * 0.4)
    ]
    tail_victories = [
        s.time_sec
        for s in all_signals
        if last_paired - 2.0 <= s.time_sec <= last_paired + 45.0
        and _strong_victory(s)
        and not _recent_paired(frames, s.time_sec, tuning, within_sec=10.0)
    ]
    if tail_victories:
        victories = sorted(set(victories + tail_victories))

    end_events: list[tuple[float, str]] = []
    if tuning.end_on_death and deaths:
        end_events.append((deaths[0], "death"))
    if tuning.trim_at_victory and victories:
        end_events.append((victories[0], "victory"))
    if end_events:
        end_events.sort(key=lambda e: e[0])
        t, kind = end_events[0]
        hold = tuning.death_hold_sec if kind == "death" else tuning.victory_hold_sec
        return t + hold

    hold = tuning.victory_hold_sec
    max_tail = tuning.max_post_victory_sec
    return min(last_paired + hold, last_paired + max_tail + 2.0)


def _dedupe_segments(
    segments: list[tuple[float, float, dict]],
    *,
    iou_threshold: float = 0.5,
) -> list[tuple[float, float, dict]]:
    if len(segments) < 2:
        return segments
    segments = sorted(segments, key=lambda s: (s[0], s[1]))
    kept: list[tuple[float, float, dict]] = [segments[0]]
    for start, end, meta in segments[1:]:
        ps, pe, _ = kept[-1]
        overlap = max(0.0, min(pe, end) - max(ps, start))
        union = max(pe, end) - min(ps, start)
        iou = overlap / union if union > 0 else 0.0
        if iou >= iou_threshold:
            if (end - start) < (pe - ps):
                kept[-1] = (start, end, meta)
            continue
        kept.append((start, end, meta))
    return kept


def _find_fight_start_anchor(
    paired: list[FrameSignals],
    all_signals: list[FrameSignals],
    tuning: EldenTuning,
) -> float:
    """First real boss attempt — skip isolated HUD blips before the actual fight."""
    if not paired:
        return 0.0
    ordered = sorted(paired, key=lambda f: f.time_sec)
    clusters: list[list[FrameSignals]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame.time_sec - clusters[-1][-1].time_sec <= 40.0:
            clusters[-1].append(frame)
        else:
            clusters.append([frame])

    def _cluster_strength(cluster: list[FrameSignals]) -> bool:
        if len(cluster) >= 2:
            return True
        return max(f.boss_bar_score for f in cluster) >= 0.7

    anchor_frame: FrameSignals | None = None
    for cluster in clusters:
        if not _cluster_strength(cluster):
            continue
        bar_hits = [f for f in cluster if f.boss_bar_score >= 0.45]
        anchor_frame = min(bar_hits or cluster, key=lambda f: f.time_sec)
        break
    if anchor_frame is None:
        anchor_frame = ordered[0]

    lookback = [
        s
        for s in all_signals
        if anchor_frame.time_sec - tuning.pre_buffer_sec - 0.5
        <= s.time_sec
        <= anchor_frame.time_sec
        and s.boss_bar_score >= 0.45
    ]
    if lookback:
        return max(0.0, min(s.time_sec for s in lookback))
    return max(0.0, anchor_frame.time_sec - tuning.pre_buffer_sec)


def _first_sustained_bar_onset(
    all_signals: list[FrameSignals],
    before: float,
    tuning: EldenTuning,
    *,
    lookback_sec: float | None = None,
) -> float | None:
    """Boss bar often appears seconds before the name is readable — find that rise."""
    min_bar = tuning.start_bar_min_score
    lookback = lookback_sec if lookback_sec is not None else tuning.start_bar_lookback_sec
    window = sorted(
        [
            s
            for s in all_signals
            if before - lookback <= s.time_sec <= before + 1.0
        ],
        key=lambda s: s.time_sec,
    )
    for i, s in enumerate(window):
        if s.time_sec > before or s.boss_bar_score < min_bar:
            continue
        if s.boss_bar_score >= 0.45:
            return s.time_sec
        for j in range(i + 1, len(window)):
            if window[j].time_sec - s.time_sec > 14.0:
                break
            if window[j].boss_bar_score >= min_bar:
                return s.time_sec
    return None


def _find_quiet_bar_rise(
    all_signals: list[FrameSignals],
    seg_start: float,
    before: float,
    tuning: EldenTuning,
) -> float | None:
    """First boss bar after a quiet stretch — attempt actually beginning."""
    quiet_gap = tuning.onset_quiet_sec
    quiet_name = tuning.onset_quiet_name_max
    window = sorted(
        [s for s in all_signals if seg_start - 5.0 <= s.time_sec <= before + 2.0],
        key=lambda s: s.time_sec,
    )
    for s in window:
        if s.time_sec < seg_start or s.boss_bar_score < 0.35:
            continue
        lookback = [p for p in window if s.time_sec - quiet_gap - 6 <= p.time_sec <= s.time_sec - 4]
        if not lookback:
            continue
        quiet_hits = sum(
            1 for p in lookback if p.boss_name_score < quiet_name and p.boss_bar_score < 0.35
        )
        if quiet_hits >= max(2, int(len(lookback) * 0.55)):
            return s.time_sec
    return None


def _find_full_hp_onset(
    all_signals: list[FrameSignals],
    fight_center: float,
    tuning: EldenTuning,
) -> float | None:
    """Boss bar at full HP — wide red strip after quiet (start of attempt)."""
    lookback = tuning.start_bar_lookback_sec
    window = sorted(
        [s for s in all_signals if fight_center - lookback <= s.time_sec <= fight_center + 20.0],
        key=lambda s: s.time_sec,
    )
    fill_min = tuning.full_bar_fill_min
    for i, s in enumerate(window):
        full = s.boss_bar_fill >= fill_min or (
            s.boss_bar_score >= tuning.full_bar_min_score and s.boss_bar_fill >= fill_min * 0.85
        )
        if not full:
            continue
        prev = [p for p in window if s.time_sec - 14.0 <= p.time_sec < s.time_sec - 0.5]
        if prev:
            quiet = sum(
                1 for p in prev if p.boss_bar_fill < 0.18 and p.boss_bar_score < 0.22
            ) / len(prev)
            if quiet < 0.4:
                continue
        nxt = [p for p in window if s.time_sec < p.time_sec <= s.time_sec + 8.0]
        if nxt and not any(
            p.boss_bar_fill >= fill_min * 0.5 or p.boss_bar_score >= tuning.start_bar_min_score
            for p in nxt
        ):
            continue
        return s.time_sec
    return None


def _find_full_bar_onset(
    all_signals: list[FrameSignals],
    seg_start: float,
    seg_end: float,
    tuning: EldenTuning,
) -> float | None:
    """First fresh boss bar pop (full HP) — low bar before, sustained bar after."""
    lookback = tuning.start_bar_lookback_sec
    window = sorted(
        [s for s in all_signals if seg_start - lookback <= s.time_sec <= seg_end + 2.0],
        key=lambda s: s.time_sec,
    )
    min_bar = tuning.start_bar_min_score
    full_bar = tuning.full_bar_min_score
    for i, s in enumerate(window):
        if s.time_sec < seg_start - 2.0:
            continue
        bar = s.boss_bar_score
        if bar < min_bar:
            continue
        prev = [p for p in window if s.time_sec - 14.0 <= p.time_sec < s.time_sec - 1.0]
        if prev:
            quiet_ratio = sum(
                1 for p in prev if p.boss_bar_score < 0.2 and p.boss_name_score < 0.35
            ) / len(prev)
            if quiet_ratio < 0.45:
                continue
        if bar >= full_bar:
            onset = s.time_sec
        else:
            nxt = [p for p in window if s.time_sec < p.time_sec <= s.time_sec + 6.0]
            if not (nxt and any(p.boss_bar_score >= min_bar for p in nxt)):
                continue
            onset = s.time_sec
        after = [p for p in window if onset <= p.time_sec <= onset + 90.0]
        paired_after = sum(1 for p in after if _boss_hud_present(p, tuning))
        if paired_after < 2:
            continue
        return onset
    return None


def _extend_start_for_cutscene(
    bar_onset: float,
    all_signals: list[FrameSignals],
    tuning: EldenTuning,
) -> float:
    """Include pre-fight cutscene (no boss HUD) before the bar appears."""
    earliest = max(0.0, bar_onset - tuning.cutscene_preroll_sec)
    start = max(0.0, bar_onset - tuning.pre_buffer_sec)
    window = sorted(
        [s for s in all_signals if earliest <= s.time_sec <= bar_onset],
        key=lambda s: s.time_sec,
    )
    for s in reversed(window):
        if s.boss_bar_score >= 0.35 or s.boss_bar_fill >= 0.25:
            break
        if _boss_hud_present(s, tuning):
            break
        if s.death_screen_score >= 0.35 or _strong_victory(s):
            break
        start = s.time_sec
    return max(0.0, max(start, bar_onset - tuning.cutscene_preroll_sec))


def _resolve_attempt_start(
    seg_start: float,
    seg_end: float,
    frames: list[FrameSignals],
    all_signals: list[FrameSignals],
    tuning: EldenTuning,
) -> float:
    """Start at full boss HP bar (+ cutscene pre-roll only for confirmed fights)."""
    paired = [f for f in frames if _boss_hud_present(f, tuning)]
    bar_hits = [f for f in frames if f.boss_bar_score >= tuning.start_bar_min_score]
    fight_center = (
        min(p.time_sec for p in paired)
        if paired
        else min(f.time_sec for f in bar_hits)
        if bar_hits
        else (seg_start + seg_end) / 2.0
    )
    had_death, had_victory = _has_strong_terminal(frames, seg_start, seg_end, threshold=0.50)
    allow_preroll = (had_death or had_victory) and len(paired) >= max(2, tuning.min_hud_samples - 1)

    def _with_optional_preroll(onset: float) -> float:
        if allow_preroll:
            return _extend_start_for_cutscene(onset, all_signals, tuning)
        return max(0.0, onset - tuning.pre_buffer_sec)

    for finder in (_find_full_hp_onset, _find_full_bar_onset):
        if finder is _find_full_hp_onset:
            onset = finder(all_signals, fight_center, tuning)
        else:
            onset = finder(all_signals, seg_start, seg_end, tuning)
        if onset is not None:
            return _with_optional_preroll(onset)

    if paired:
        anchor = _find_fight_start_anchor(paired, all_signals, tuning)
        return _with_optional_preroll(anchor)
    return _find_attempt_onset(all_signals, seg_start, seg_end, tuning)


def _refine_segment_bounds(
    start: float,
    end: float,
    frames: list[FrameSignals],
    all_signals: list[FrameSignals],
    tuning: EldenTuning,
) -> tuple[float, float]:
    """Tighten clip start/end using boss name + victory signals."""
    if not frames:
        return start, end

    paired = [f for f in frames if _boss_hud_present(f, tuning)]
    start = _resolve_attempt_start(start, end, frames, all_signals, tuning)
    end = _compute_clip_end(start, frames, all_signals, tuning)

    # Apply offsets learned from your manual timing corrections.
    start = max(0.0, start + tuning.start_offset_sec)
    end = end + tuning.end_offset_sec

    end = max(start + 2.0, end)
    # Only cap length if a positive max is configured (0 = unlimited).
    if tuning.max_clip_sec and (end - start) > tuning.max_clip_sec:
        end = start + tuning.max_clip_sec
    return start, end


def _split_on_victory_cooldown(
    start: float,
    end: float,
    frames: list[FrameSignals],
    all_signals: list[FrameSignals],
    tuning: EldenTuning,
    *,
    cooldown_sec: float = 75.0,
) -> list[tuple[float, float, list[FrameSignals]]]:
    """Split when a victory ends one fight and boss activity resumes much later."""
    victories = sorted(
        s.time_sec
        for s in all_signals
        if start <= s.time_sec <= end and s.victory_screen_score >= 0.35
    )
    if len(victories) < 1:
        return [(start, end, frames)]

    pieces: list[tuple[float, float, list[FrameSignals]]] = []
    seg_start = start
    seg_frames = list(frames)

    for vt in victories:
        clip_end = vt + tuning.victory_hold_sec
        later = [f for f in seg_frames if f.time_sec > clip_end + 8.0]
        if not later:
            continue
        gap = later[0].time_sec - clip_end
        if gap >= cooldown_sec:
            chunk = [f for f in seg_frames if seg_start <= f.time_sec <= clip_end + 2.0]
            if chunk:
                pieces.append((seg_start, clip_end + 2.0, chunk))
            seg_start = later[0].time_sec
            seg_frames = [f for f in seg_frames if f.time_sec >= seg_start]

    if seg_frames:
        pieces.append((seg_start, end, seg_frames))
    return pieces if pieces else [(start, end, frames)]


def _split_segment_per_attempt(
    start: float,
    end: float,
    frames: list[FrameSignals],
    all_signals: list[FrameSignals],
    tuning: EldenTuning,
) -> list[tuple[float, float, list[FrameSignals]]]:
    """Split a multi-attempt window into one clip per try.

    Each attempt = fight onset (full HP / HUD after quiet) → YOU DIED or ENEMY FELLED.
    """
    if not frames:
        return []

    window = sorted(
        [
            s
            for s in all_signals
            if start - tuning.start_bar_lookback_sec <= s.time_sec <= end + 30.0
        ],
        key=lambda s: s.time_sec,
    )
    if not window:
        return [(start, end, frames)]

    terminals: list[tuple[float, str]] = []
    for s in window:
        if not (start <= s.time_sec <= end + 8.0):
            continue
        if _strong_death(s, threshold=0.50):
            terminals.append((s.time_sec, "death"))
        elif _strong_victory(s, threshold=0.55) and not _recent_paired(
            frames, s.time_sec, tuning, within_sec=8.0
        ):
            terminals.append((s.time_sec, "victory"))

    collapsed: list[tuple[float, str]] = []
    for mt, kind in terminals:
        if collapsed and mt - collapsed[-1][0] < 8.0:
            continue
        collapsed.append((mt, kind))

    if len(collapsed) <= 1:
        return [(start, end, frames)]

    pieces: list[tuple[float, float, list[FrameSignals]]] = []
    prev_close = start - 1.0

    for mt, kind in collapsed:
        look_from = max(prev_close, start - 5.0)
        fight_center = (look_from + mt) / 2.0
        onset = _find_full_hp_onset(all_signals, fight_center, tuning)
        if onset is None or onset < look_from - 2.0 or onset > mt - 3.0:
            onset = _find_full_bar_onset(all_signals, look_from, mt, tuning)
        if onset is None or onset < look_from - 2.0 or onset > mt - 3.0:
            # First sustained bar after previous attempt closed.
            bar_hits = [
                s.time_sec
                for s in window
                if look_from <= s.time_sec < mt - 2.0
                and _can_open_segment(s, tuning)
            ]
            onset = bar_hits[0] if bar_hits else max(look_from, start)

        onset = max(0.0, onset)
        if mt - onset < max(5.0, tuning.min_fight_sec * 0.5):
            continue

        hold = tuning.death_hold_sec if kind == "death" else tuning.victory_hold_sec
        attempt_end = min(mt + hold, end + hold)
        # First attempt on a boss: allow cutscene pre-roll only with named HUD.
        chunk_probe = [
            f for f in all_signals if onset - 1.0 <= f.time_sec <= attempt_end + 1.0
        ]
        paired_probe = [f for f in chunk_probe if _boss_hud_present(f, tuning)]
        if (
            not pieces
            and onset - start < tuning.cutscene_preroll_sec
            and len(paired_probe) >= max(2, tuning.min_hud_samples - 1)
        ):
            onset = _extend_start_for_cutscene(onset, all_signals, tuning)

        chunk = [
            f
            for f in all_signals
            if onset - 1.0 <= f.time_sec <= attempt_end + 1.0
        ]
        if not chunk:
            chunk = [f for f in frames if onset - 1.0 <= f.time_sec <= attempt_end + 1.0]
        if chunk and attempt_end - onset >= tuning.min_keep_sec:
            pieces.append((onset, attempt_end, chunk))
        prev_close = attempt_end + 2.0

    # Trailing attempt still in progress (no death/victory yet) — only keep strong HUD.
    if prev_close < end - 5.0:
        trailing = [f for f in frames if f.time_sec >= prev_close]
        if trailing:
            paired = [f for f in trailing if _boss_hud_present(f, tuning)]
            peak_bar = max(f.boss_bar_score for f in trailing)
            peak_name = max((f.boss_name_score for f in paired), default=0.0)
            if (
                len(paired) >= tuning.min_hud_samples
                and peak_bar >= tuning.start_bar_min_score
                and peak_name >= tuning.min_boss_name_score
                and end - prev_close >= tuning.min_keep_sec
            ):
                pieces.append((prev_close, end, trailing))

    return pieces if pieces else [(start, end, frames)]


def _cluster_boss_segments(
    signals: list[FrameSignals],
    *,
    boss_threshold: float = 0.22,
    gap_sec: float = 90.0,
    pre_buffer_sec: float = 5.0,
    min_span_sec: float = 8.0,
    tuning: EldenTuning | None = None,
    refine_signals: list[FrameSignals] | None = None,
) -> list[tuple[float, float, dict]]:
    """Merge frame hits into fight windows — one clip per attempt when enabled."""
    if not signals:
        return []

    tuning = tuning or EldenTuning.defaults()
    timing_signals = refine_signals if refine_signals is not None else signals

    if tuning.merge_while_bar_visible:
        out: list[tuple[float, float, dict]] = []
        raw = _merge_hud_segments(
            _cluster_by_boss_hud(signals, tuning),
            max_gap_sec=18.0 if tuning.split_per_attempt else 50.0,
            split_per_attempt=tuning.split_per_attempt,
        )
        for _start, _end, frames in raw:
            subsegments = (
                _split_segment_per_attempt(_start, _end, frames, timing_signals, tuning)
                if tuning.split_per_attempt
                else [(_start, _end, frames)]
            )
            for sub_start, sub_end, sub_frames in subsegments:
                start, end = _refine_segment_bounds(
                    sub_start, sub_end, sub_frames, timing_signals, tuning
                )
                paired = [f for f in sub_frames if _boss_hud_present(f, tuning)] or sub_frames
                peak_bar = max(f.boss_bar_score for f in paired)
                peak_name = max(f.boss_name_score for f in paired)
                had_death, had_victory = _has_strong_terminal(
                    timing_signals, start, end, threshold=0.50
                )
                if not _is_credible_boss_clip(
                    peak_bar=peak_bar,
                    peak_name=peak_name,
                    frame_hits=len([f for f in sub_frames if _boss_hud_present(f, tuning)]),
                    had_death=had_death,
                    had_victory=had_victory,
                    duration_sec=end - start,
                    tuning=tuning,
                ):
                    continue
                meta = {
                    "peak_boss_bar_score": round(peak_bar, 3),
                    "peak_boss_name_score": round(peak_name, 3),
                    "had_death_screen": had_death,
                    "had_victory_screen": had_victory,
                    "frame_hits": len([f for f in sub_frames if _boss_hud_present(f, tuning)]),
                    "avg_boss_score": round(
                        sum(f.combined_boss_score() for f in paired) / len(paired), 3
                    ),
                }
                out.append((start, end, meta))
        return _dedupe_segments(out, iou_threshold=0.35)

    gap_sec = tuning.merge_gap_sec
    pre_buffer_sec = tuning.pre_buffer_sec

    def _is_hit(s: FrameSignals) -> bool:
        if not frame_passes_tuning(s, tuning):
            return False
        return s.combined_boss_score() >= boss_threshold or s.boss_name_score >= tuning.min_boss_name_score

    hits = [s for s in signals if _is_hit(s)]
    if not hits:
        return []

    if len(hits) < tuning.min_frame_hits:
        strong = [s for s in hits if s.boss_name_score >= 0.35]
        if len(strong) < max(2, tuning.min_frame_hits - 1):
            return []

    def _should_bridge(cur_end: float, sig: FrameSignals, cur_frames: list[FrameSignals]) -> bool:
        gap = sig.time_sec - cur_end
        between = [s for s in signals if cur_end < s.time_sec < sig.time_sec]
        had_death = any(s.death_screen_score >= 0.35 for s in between)
        had_victory = any(s.victory_screen_score >= 0.35 for s in between)

        if tuning.split_per_attempt:
            if had_death or had_victory:
                return False
            return gap <= gap_sec

        if gap <= gap_sec:
            return True
        if gap > tuning.retry_bridge_sec:
            return False
        if not had_death:
            return False
        return any(
            s.boss_name_score >= tuning.min_boss_name_score
            or s.boss_bar_score >= tuning.min_boss_bar_score
            for s in between
        ) or sig.boss_name_score >= tuning.min_boss_name_score

    segments: list[tuple[float, float, list[FrameSignals]]] = []
    cur_start = hits[0].time_sec
    cur_end = hits[0].time_sec
    cur_frames = [hits[0]]

    for sig in hits[1:]:
        if _should_bridge(cur_end, sig, cur_frames):
            cur_end = sig.time_sec
            cur_frames.append(sig)
        else:
            segments.append((cur_start, cur_end, cur_frames))
            cur_start = sig.time_sec
            cur_end = sig.time_sec
            cur_frames = [sig]
    segments.append((cur_start, cur_end, cur_frames))

    out: list[tuple[float, float, dict]] = []
    for _start, _end, frames in segments:
        span = _end - _start + 2.0
        if span < min_span_sec and max(f.combined_boss_score() for f in frames) < 0.45:
            continue

        subsegments = (
            _split_segment_per_attempt(_start, _end, frames, timing_signals, tuning)
            if tuning.split_per_attempt
            else _split_on_victory_cooldown(_start, _end, frames, timing_signals, tuning)
        )
        for sub_start, sub_end, sub_frames in subsegments:
            start, end = _refine_segment_bounds(
                sub_start, sub_end, sub_frames, timing_signals, tuning
            )
            peak_bar = max(f.boss_bar_score for f in sub_frames)
            peak_name = max(f.boss_name_score for f in sub_frames)
            had_death, had_victory = _has_strong_terminal(
                timing_signals, start, end, threshold=0.50
            )
            frame_hits = len([f for f in sub_frames if _boss_hud_present(f, tuning)]) or len(
                sub_frames
            )
            if not _is_credible_boss_clip(
                peak_bar=peak_bar,
                peak_name=peak_name,
                frame_hits=frame_hits,
                had_death=had_death,
                had_victory=had_victory,
                duration_sec=end - start,
                tuning=tuning,
            ):
                continue
            meta = {
                "peak_boss_bar_score": round(peak_bar, 3),
                "peak_boss_name_score": round(peak_name, 3),
                "had_death_screen": had_death,
                "had_victory_screen": had_victory,
                "frame_hits": frame_hits,
                "avg_boss_score": round(
                    sum(f.combined_boss_score() for f in sub_frames) / len(sub_frames), 3
                ),
            }
            out.append((start, end, meta))
    return _dedupe_segments(out)


def _candidate_id(vod_id: str, start: float, end: float) -> str:
    key = f"{vod_id}:{start:.1f}:{end:.1f}"
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def _build_reason(meta: dict) -> str:
    parts = [
        f"Boss bar score {meta.get('peak_boss_bar_score', 0):.2f}",
        f"name text {meta.get('peak_boss_name_score', 0):.2f}",
    ]
    dual = float(meta.get("peak_dual_boss_bar_score") or 0.0)
    if dual >= 0.50:
        parts.append(f"dual boss bars {dual:.2f}")
    if meta.get("had_death_screen"):
        parts.append("YOU DIED screen detected")
    if meta.get("had_victory_screen"):
        parts.append("victory banner detected")
    if meta.get("ml_end_kind") == "incomplete":
        parts.append("HUD ended without banner (flee/cut)")
    parts.append(f"{meta.get('frame_hits', 0)} frame hits")
    return "; ".join(parts)


def _segment_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return overlap / union if union > 0 else 0.0


def load_fight_anchors(data_dir: Path | None, vod_id: str) -> list[tuple[float, float]]:
    """Human-trusted fight windows (correct, manually-timed, or missed) to snap clips to."""
    if data_dir is None:
        return []
    path = data_dir / "reference" / "elden_ring_supervised.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    anchors: list[tuple[float, float]] = []
    for e in data.get("examples", []):
        if e.get("vod_id") != vod_id:
            continue
        verdict = e.get("verdict")
        trusted = (
            verdict == "correct"
            or verdict == "missed"
            or bool(e.get("adjusted"))
        )
        if not trusted:
            continue
        try:
            anchors.append((float(e["start_sec"]), float(e["end_sec"])))
        except (KeyError, TypeError, ValueError):
            continue
    return anchors


def _merge_anchors(
    anchors: list[tuple[float, float]],
    *,
    overlap_iou: float = 0.6,
) -> list[tuple[float, float]]:
    """Collapse near-duplicate labels of the SAME attempt (relabeled across sessions).

    Only heavily-overlapping windows merge. Adjacent windows are kept separate so
    back-to-back attempts on the same boss stay as distinct clips.
    """
    if not anchors:
        return []
    ordered = sorted(anchors, key=lambda a: a[0])
    clusters: list[list[tuple[float, float]]] = [[ordered[0]]]
    for rs, re in ordered[1:]:
        # Compare against the whole current cluster's span, not just the last entry.
        cs = min(c[0] for c in clusters[-1])
        ce = max(c[1] for c in clusters[-1])
        if _segment_iou(rs, re, cs, ce) >= overlap_iou:
            clusters[-1].append((rs, re))
        else:
            clusters.append([(rs, re)])

    def _median(vals: list[float]) -> float:
        vals = sorted(vals)
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    merged: list[tuple[float, float]] = []
    for cluster in clusters:
        starts = [c[0] for c in cluster]
        ends = [c[1] for c in cluster]
        merged.append((_median(starts), _median(ends)))
    return merged


def apply_fight_anchors(
    candidates: list[BossFightCandidate],
    anchors: list[tuple[float, float]],
    *,
    vod_id: str = "",
) -> list[BossFightCandidate]:
    """Snap auto-detections to the human-labeled fight window they overlap.

    Snapping only — a labeled window with no overlapping detection is NOT injected
    (use the timing editor to move a clip onto a missed fight).
    """
    if not anchors:
        return candidates

    merged = _merge_anchors(anchors)
    result = list(candidates)
    used_anchors: set[int] = set()

    # Each detection snaps to its single best-overlapping (unclaimed) labeled window.
    for i, cand in enumerate(result):
        best_j = -1
        best_iou = 0.0
        for j, (rs, re) in enumerate(merged):
            if j in used_anchors or re <= rs + 2.0:
                continue
            iou = _segment_iou(cand.start_sec, cand.end_sec, rs, re)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= 0.1:
            rs, re = merged[best_j]
            result[i] = BossFightCandidate(
                id=cand.id,
                start_sec=round(max(0.0, rs), 2),
                end_sec=round(re, 2),
                confidence=cand.confidence,
                reason=cand.reason + "; aligned to labeled fight",
                signals=cand.signals,
                peak_boss_bar_score=cand.peak_boss_bar_score,
                had_death_screen=cand.had_death_screen,
                had_victory_screen=cand.had_victory_screen,
                source=cand.source,
            )
            used_anchors.add(best_j)

    # Drop near-duplicate clips created by snapping (keep higher confidence).
    result.sort(key=lambda c: c.start_sec)
    deduped: list[BossFightCandidate] = []
    for cand in result:
        dup = next(
            (
                d
                for d in deduped
                if _segment_iou(cand.start_sec, cand.end_sec, d.start_sec, d.end_sec)
                >= 0.85
            ),
            None,
        )
        if dup is None:
            deduped.append(cand)
        elif cand.confidence > dup.confidence:
            deduped[deduped.index(dup)] = cand
    return deduped


def scan_boss_fights(
    video_path: Path,
    work_dir: Path,
    vod_id: str,
    *,
    interval_sec: float = 2.0,
    ffmpeg: str = "ffmpeg",
    tuning: EldenTuning | None = None,
    on_progress: object | None = None,
) -> BossScanResult:
    """Sample the VOD, probe attempt edges, then emit one clip per attempt."""
    tuning = (tuning or EldenTuning.defaults()).apply_builtin_attempt_policy()

    def _progress(fields: dict) -> None:
        if callable(on_progress):
            on_progress(fields)

    _progress(
        {
            "phase": "starting",
            "message": "Reading video duration…",
            "pct": 1.0,
            "current": 0,
            "total": 0,
        }
    )
    signals, duration = sample_frame_signals(
        video_path,
        work_dir,
        interval_sec=interval_sec,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    probe_dir = work_dir / "edge_probes"
    refined = supplement_start_probes(
        signals,
        video_path,
        tuning,
        probe_dir=probe_dir,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    refined = supplement_end_probes(
        refined,
        video_path,
        tuning,
        probe_dir=probe_dir,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    _progress(
        {
            "phase": "clustering",
            "message": "Building attempt clips…",
            "pct": 97.0,
            "current": 1,
            "total": 1,
        }
    )
    return rescan_from_signals(
        signals,
        duration,
        vod_id,
        tuning=tuning,
        refine_signals=refined,
    )


def rescan_from_signals(
    signals: list[FrameSignals],
    duration: float,
    vod_id: str,
    *,
    tuning: EldenTuning | None = None,
    refine_signals: list[FrameSignals] | None = None,
    data_dir: Path | None = None,
) -> BossScanResult:
    """Re-cluster boss fights from cached frame signals (fast — no ffmpeg)."""
    tuning = (tuning or load_elden_tuning()).apply_builtin_attempt_policy()
    timing = refine_signals if refine_signals is not None else signals
    segments = _cluster_boss_segments(
        signals, tuning=tuning, refine_signals=timing
    )
    candidates: list[BossFightCandidate] = []
    for start, end, meta in segments:
        confidence = min(
            0.95,
            meta["avg_boss_score"] * 0.25
            + meta["peak_boss_bar_score"] * 0.2
            + meta["peak_boss_name_score"] * 0.45
            + (0.1 if meta["frame_hits"] >= 5 else 0.0),
        )
        candidates.append(
            BossFightCandidate(
                id=_candidate_id(vod_id, start, end),
                start_sec=round(start, 2),
                end_sec=round(end, 2),
                confidence=round(confidence, 3),
                reason=_build_reason(meta),
                signals=meta,
                peak_boss_bar_score=meta["peak_boss_bar_score"],
                had_death_screen=bool(meta["had_death_screen"]),
                had_victory_screen=bool(meta["had_victory_screen"]),
            )
        )
    anchors = load_fight_anchors(data_dir, vod_id)
    candidates = apply_fight_anchors(candidates, anchors, vod_id=vod_id)
    return BossScanResult(
        vod_id=vod_id,
        duration_sec=duration,
        candidates=candidates,
        frame_samples=timing,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )


def tuning_path(data_dir: Path) -> Path:
    return data_dir / "reference" / "elden_ring_tuning.json"


def load_elden_tuning(data_dir: Path | None = None) -> EldenTuning:
    if data_dir is None:
        return EldenTuning.defaults().apply_builtin_attempt_policy()
    path = tuning_path(data_dir)
    if not path.exists():
        return EldenTuning.defaults().apply_builtin_attempt_policy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return EldenTuning.from_dict(data.get("tuning", data)).apply_builtin_attempt_policy()
    except (json.JSONDecodeError, TypeError, ValueError):
        return EldenTuning.defaults().apply_builtin_attempt_policy()


def save_elden_tuning(data_dir: Path, tuning: EldenTuning, meta: dict | None = None) -> Path:
    path = tuning_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tuning": tuning.to_dict(),
        **(meta or {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _coarse_frame_samples(
    samples: list[FrameSignals],
    *,
    interval_sec: float = 2.0,
) -> list[FrameSignals]:
    """Keep 2s scan grid only — drop 1 Hz / 0.5 Hz probe frames from disk cache."""
    out: list[FrameSignals] = []
    for sig in samples:
        step = round(sig.time_sec / interval_sec)
        if abs(sig.time_sec - step * interval_sec) <= 0.15:
            out.append(sig)
    return out


def save_scan_result(work_dir: Path, result: BossScanResult) -> Path:
    path = work_dir / "boss_scan.json"
    payload = result.to_dict()
    payload["frame_samples"] = [
        f.to_dict() for f in _coarse_frame_samples(result.frame_samples)
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_scan_result(work_dir: Path) -> BossScanResult | None:
    path = work_dir / "boss_scan.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError):
        return None
    candidates = [BossFightCandidate(**c) for c in data.get("candidates", [])]
    frames = [FrameSignals(**f) for f in data.get("frame_samples", [])]
    return BossScanResult(
        vod_id=str(data.get("vod_id", "")),
        duration_sec=float(data.get("duration_sec", 0)),
        candidates=candidates,
        frame_samples=frames,
        scanned_at=str(data.get("scanned_at", "")),
    )
