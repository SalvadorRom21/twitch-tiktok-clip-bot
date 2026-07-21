"""Adaptive TikTok gameplay-band estimation for any face/game zoom ratio."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock

import cv2
import numpy as np


@dataclass(frozen=True)
class LayoutProfile:
    """Continuous gameplay panel in normalized frame coordinates [0, 1]."""

    game_y0: float = 0.30
    game_y1: float = 0.76

    def clamp(self) -> LayoutProfile:
        y0 = float(np.clip(self.game_y0, 0.08, 0.70))
        # Keep enough vertical room for YOU DIED / FELLED (short bands miss them).
        y1 = float(np.clip(self.game_y1, y0 + 0.38, 0.95))
        if y1 - y0 < 0.38:
            y1 = min(0.95, y0 + 0.38)
        # Classic mid-stack: don't let soft-refine collapse the bottom edge.
        if y0 <= 0.36:
            y1 = max(y1, min(0.90, 0.74))
        return LayoutProfile(game_y0=round(y0, 4), game_y1=round(y1, 4))

    @property
    def height(self) -> float:
        return max(0.01, self.game_y1 - self.game_y0)

    def bar_sweep(self) -> tuple[float, float]:
        """Boss HP bar lives in the lower part of the gameplay panel."""
        span = self.height
        return (
            self.game_y0 + span * 0.30,
            min(0.93, self.game_y1 - span * 0.02),
        )

    def dual_bar_band(self) -> tuple[float, float]:
        span = self.height
        return (
            self.game_y0 + span * 0.48,
            min(0.93, self.game_y1 - span * 0.02),
        )

    def center_text_band(self) -> tuple[float, float]:
        """YOU DIED / ENEMY FELLED — mid band of gameplay (full-frame y)."""
        span = self.height
        return (
            self.game_y0 + span * 0.16,
            self.game_y0 + span * 0.80,
        )

    def bar_sweep_hypotheses(self) -> list[tuple[float, float]]:
        """Overlapping sweeps so dark arenas / odd zooms still find the bar."""
        primary = self.bar_sweep()
        hyps = [
            primary,
            (0.46, 0.74),
            (0.52, 0.82),
            (0.58, 0.88),
            (max(0.40, self.game_y0 + 0.05), min(0.92, self.game_y1)),
        ]
        # Dedupe near-identical ranges.
        out: list[tuple[float, float]] = []
        for a, b in hyps:
            if b - a < 0.12:
                continue
            if any(abs(a - x) < 0.03 and abs(b - y) < 0.03 for x, y in out):
                continue
            out.append((float(a), float(b)))
        return out

    def center_text_hypotheses(self) -> list[tuple[float, float]]:
        primary = self.center_text_band()
        hyps = [
            primary,
            (0.40, 0.66),
            (0.48, 0.78),
            (0.52, 0.84),
            (
                self.game_y0 + self.height * 0.10,
                self.game_y0 + self.height * 0.85,
            ),
        ]
        out: list[tuple[float, float]] = []
        for a, b in hyps:
            if b - a < 0.15:
                continue
            if any(abs(a - x) < 0.03 and abs(b - y) < 0.03 for x, y in out):
                continue
            out.append((float(a), float(b)))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def classic(cls) -> LayoutProfile:
        return cls(0.30, 0.76)


_lock = Lock()
_active: LayoutProfile = LayoutProfile.classic()


def get_active_layout() -> LayoutProfile:
    with _lock:
        return _active


def set_active_layout(profile: LayoutProfile | None) -> LayoutProfile:
    """Set process-wide layout used by OpenCV + ML crops during a scan."""
    global _active
    with _lock:
        _active = (profile or LayoutProfile.classic()).clamp()
        return _active


def _skin_row_ratio(frame: np.ndarray) -> np.ndarray:
    """Per-row fraction of skin-like pixels (facecam cue)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broad skin ranges under mixed LED lighting.
    skin = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 25, 50), (25, 180, 255)),
        cv2.inRange(hsv, (160, 20, 50), (180, 180, 255)),
    )
    return skin.mean(axis=1) / 255.0


def _row_energy(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var = gray.astype(np.float32).var(axis=1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = np.mean(np.abs(gx), axis=1)
    energy = 0.55 * var + 0.45 * grad * 8.0
    k = max(9, (len(energy) // 40) | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(energy, kernel, mode="same")


def _face_game_split(frames: list[np.ndarray]) -> float | None:
    """Estimate y where facecam ends / gameplay begins using skin drop-off."""
    ratios = []
    for frame in frames[:10]:
        # Center columns only — ignore side leaderboards / stickers.
        h, w = frame.shape[:2]
        mid = frame[:, int(w * 0.22) : int(w * 0.78)]
        r = _skin_row_ratio(mid)
        k = max(9, (len(r) // 35) | 1)
        sm = np.convolve(r, np.ones(k) / k, mode="same")
        ratios.append(sm)
    if not ratios:
        return None
    m = min(len(r) for r in ratios)
    skin = np.median(np.stack([r[:m] for r in ratios], axis=0), axis=0)
    h = len(skin)
    upper = skin[: int(h * 0.45)]
    peak = float(np.max(upper)) if upper.size else 0.0
    if peak < 0.055:
        return None
    thr = max(0.03, peak * 0.40)
    start = int(h * 0.20)
    end = int(h * 0.70)
    split_i = None
    low_run = 0
    for i in range(start, end):
        if skin[i] < thr:
            low_run += 1
            if low_run >= max(8, h // 35):
                split_i = i - low_run // 2
                break
        else:
            low_run = 0
    if split_i is None:
        return None
    split = split_i / h
    # Confirm: upper band has more skin than the band below the split.
    upper_mean = float(np.mean(skin[:split_i]))
    lower_mean = float(np.mean(skin[split_i : int(h * 0.88)]))
    if upper_mean < 0.04 or upper_mean < lower_mean * 1.35:
        return None
    return float(np.clip(split, 0.24, 0.60))


def _band_red_bonus(frames: list[np.ndarray], y0: float, y1: float) -> float:
    """Bonus when a wide saturated red strip appears in the lower band (boss bar)."""
    bonus = 0.0
    hits = 0
    for frame in frames[:8]:
        h, w = frame.shape[:2]
        ya = int(h * (y0 + (y1 - y0) * 0.35))
        yb = int(h * y1)
        xa, xb = int(w * 0.12), int(w * 0.88)
        roi = frame[ya:yb, xa:xb]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 90, 70), (12, 255, 255)),
            cv2.inRange(hsv, (168, 90, 70), (180, 255, 255)),
        )
        row = red.mean(axis=1) / 255.0
        if row.size < 4:
            continue
        peak = float(row.max())
        peak_i = int(np.argmax(row))
        width = float(np.count_nonzero(red[peak_i])) / max(1, red.shape[1])
        if peak >= 0.18 and width >= 0.35:
            hits += 1
            bonus += peak * 0.6 + width * 0.5
    if hits <= 0:
        return 0.0
    return bonus / max(1, min(len(frames), 8))


def estimate_layout_profile(frames: list[np.ndarray]) -> LayoutProfile:
    """Estimate continuous gameplay y-range (any zoom / stack ratio).

    Prefer a facecam→game split from skin cues (works when the arena is dark).
    Fall back to red-bar-weighted band search.
    """
    usable = [f for f in frames if f is not None and getattr(f, "size", 0) > 0]
    if not usable:
        return LayoutProfile.classic()

    # Zoomed TikTok stacks: strong lower-panel red bar with little mid-stack red
    # (Night's Cavalry) — prefer bottom gameplay even if skin split is weak.
    lower_red = _band_red_bonus(usable, 0.55, 0.92)
    mid_red = _band_red_bonus(usable, 0.28, 0.55)
    if lower_red >= 0.24 and lower_red >= mid_red * 1.35:
        return LayoutProfile(game_y0=0.42, game_y1=0.90).clamp()

    split = _face_game_split(usable)
    if split is not None:
        # Gameplay starts near the facecam/game cut; end above follower chrome.
        y0 = float(np.clip(split - 0.02, 0.24, 0.58))
        # Classic stacks end ~0.76; zoomed bottom panels need ~0.88–0.90.
        y1 = 0.90 if y0 >= 0.42 else 0.76
        y1 = max(y1, y0 + 0.40)
        y1 = min(0.90, y1)
        if y1 - y0 >= 0.38:
            # Zoomed bottom panels: boss bar / red HUD sits well below mid-stack.
            if _band_red_bonus(usable, max(y0, 0.48), 0.92) >= 0.18:
                y1 = 0.90
            return LayoutProfile(game_y0=y0, game_y1=y1).clamp()

    # Grid search fallback — penalize face-top, reward lower panels + red bars.
    energies = [_row_energy(f) for f in usable[:12]]
    best_score = -1e18
    best = LayoutProfile.classic()
    for y0 in np.linspace(0.24, 0.58, 18):
        for height in (0.40, 0.44, 0.48, 0.52, 0.56):
            y1 = float(y0) + float(height)
            if y1 > 0.93:
                continue
            # Detail score inside band.
            detail = 0.0
            n = 0
            for energy in energies:
                h = len(energy)
                a, b = int(h * y0), int(h * y1)
                if b <= a + 8:
                    continue
                detail += float(np.mean(energy[a:b]))
                n += 1
            if n <= 0:
                continue
            score = detail / n
            score += 0.8 * _band_red_bonus(usable, float(y0), y1)
            score += float(y0) * 0.45  # prefer lower gameplay panels
            if float(y0) < 0.26:
                score -= 0.8
            if 0.28 <= float(y0) <= 0.36 and 0.70 <= y1 <= 0.80:
                score += 0.35  # classic prior
            if score > best_score:
                best_score = score
                best = LayoutProfile(game_y0=float(y0), game_y1=y1)
    return best.clamp()


def refine_layout_with_bar(
    profile: LayoutProfile,
    bar_y_center: float,
    *,
    alpha: float = 0.35,
) -> LayoutProfile:
    """Nudge gameplay band so a detected boss bar sits in its lower portion."""
    if not (0.25 <= bar_y_center <= 0.92):
        return profile
    # Never shrink — short bands drop ENEMY FELLED / YOU DIED out of ROI.
    target_height = float(np.clip(max(profile.height, 0.42), 0.42, 0.58))
    target_y0 = bar_y_center - 0.78 * target_height
    target_y1 = target_y0 + target_height
    if target_y0 <= 0.36:
        target_y1 = max(target_y1, 0.76)
        target_y0 = target_y1 - target_height
    blended = LayoutProfile(
        game_y0=(1.0 - alpha) * profile.game_y0 + alpha * target_y0,
        game_y1=(1.0 - alpha) * profile.game_y1 + alpha * target_y1,
    )
    return blended.clamp()


def estimate_layout_from_video(
    video_path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_times: list[float] | None = None,
) -> LayoutProfile:
    """Pull a few frames and estimate layout (used before full scan grid)."""
    from pathlib import Path
    import subprocess
    import tempfile

    video_path = Path(video_path)
    if not video_path.exists():
        return LayoutProfile.classic()
    times = sample_times or [5.0, 20.0, 45.0, 90.0, 150.0, 210.0]
    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="elden_layout_") as tmp:
        tmp_path = Path(tmp)
        for i, t in enumerate(times):
            out = tmp_path / f"s_{i}.jpg"
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{float(t):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "4",
                str(out),
            ]
            subprocess.run(cmd, capture_output=True, check=False)
            img = cv2.imread(str(out))
            if img is not None:
                frames.append(img)
    return estimate_layout_profile(frames)
