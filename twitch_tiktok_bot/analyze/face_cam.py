"""Detect streamer face-cam overlay region for stacked TikTok layout."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from twitch_tiktok_bot.config import AppConfig
    from twitch_tiktok_bot.models import ClipAnalysis

VALID_FACE_CAM_CORNERS = frozenset(
    {"auto", "top-left", "top-right", "bottom-left", "bottom-right"}
)

ANALYSIS_WIDTH = 960
MIN_OVERLAY_AREA = 0.025
MAX_OVERLAY_AREA = 0.42
WEBCAM_ASPECT = 16 / 9


@dataclass
class FaceCamRegion:
    """Normalized 0-1 bounding box of the face-cam overlay on the Twitch layout."""

    x: float
    y: float
    w: float
    h: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> FaceCamRegion | None:
        if not data:
            return None
        try:
            return cls(
                x=float(data["x"]),
                y=float(data["y"]),
                w=float(data["w"]),
                h=float(data["h"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def normalize_face_cam_corner(corner: str | None) -> str:
    value = (corner or "auto").strip().lower()
    return value if value in VALID_FACE_CAM_CORNERS else "auto"


def default_face_cam_region(corner: str = "auto") -> FaceCamRegion:
    """Last-resort fallback only — detection should run automatically first."""
    _ = normalize_face_cam_corner(corner)
    return FaceCamRegion(x=0.02, y=0.66, w=0.24, h=0.30)


def _clamp_region(region: FaceCamRegion) -> FaceCamRegion:
    x = max(0.0, min(region.x, 1.0))
    y = max(0.0, min(region.y, 1.0))
    w = max(0.02, min(region.w, 1.0 - x))
    h = max(0.02, min(region.h, 1.0 - y))
    return FaceCamRegion(x=x, y=y, w=w, h=h)


def _region_area(region: FaceCamRegion) -> float:
    return region.w * region.h


def _contains_point(region: FaceCamRegion, cx: float, cy: float) -> bool:
    return (
        region.x <= cx <= region.x + region.w
        and region.y <= cy <= region.y + region.h
    )


def _sample_frame_paths(
    video_path: Path,
    output_dir: Path,
    sample_count: int = 8,
    ffmpeg: str = "ffmpeg",
) -> list[Path]:
    """Spread frame samples across the full video (clips and long VODs)."""
    from twitch_tiktok_bot.analyze.duration import get_video_duration

    output_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path, ffmpeg=ffmpeg)
    if duration <= 0:
        return []

    paths: list[Path] = []
    for index in range(sample_count):
        # Evenly spaced timestamps — avoids only sampling the first second of a VOD.
        timestamp = duration * (index + 1) / (sample_count + 1)
        out_path = output_dir / f"face_{index + 1:04d}.jpg"
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        if out_path.exists() and out_path.stat().st_size > 0:
            paths.append(out_path)
    return paths


def _load_frames(frame_paths: list[Path], width: int = ANALYSIS_WIDTH) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        height = max(1, int(width * h / w))
        frames.append(cv2.resize(image, (width, height)))
    return frames


def _corner_for_point(cx: float, cy: float) -> str:
    if cx < 0.5 and cy < 0.5:
        return "top-left"
    if cx >= 0.5 and cy < 0.5:
        return "top-right"
    if cx < 0.5:
        return "bottom-left"
    return "bottom-right"


def _corner_zone(
    corner: str, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    zones = {
        "top-left": (0, 0, int(frame_w * 0.58), int(frame_h * 0.58)),
        "top-right": (int(frame_w * 0.42), 0, frame_w, int(frame_h * 0.58)),
        "bottom-left": (0, int(frame_h * 0.42), int(frame_w * 0.58), frame_h),
        "bottom-right": (
            int(frame_w * 0.42),
            int(frame_h * 0.42),
            frame_w,
            frame_h,
        ),
    }
    return zones[corner]


def _is_overlay_face(
    x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
) -> bool:
    rel_area = (w * h) / (frame_w * frame_h)
    if rel_area > 0.30 or rel_area < 0.001:
        return False
    if w > frame_w * 0.48 or h > frame_h * 0.52:
        return False
    in_horizontal_edge = x < frame_w * 0.45 or x + w > frame_w * 0.55
    in_vertical_edge = y < frame_h * 0.45 or y + h > frame_h * 0.55
    return in_horizontal_edge and in_vertical_edge


def _face_cascades() -> list[tuple[cv2.CascadeClassifier, bool]]:
    names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    cascades: list[tuple[cv2.CascadeClassifier, bool]] = []
    for name in names:
        path = cv2.data.haarcascades + name
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            continue
        cascades.append((cascade, False))
        if "profile" in name:
            cascades.append((cascade, True))
    return cascades


def _detect_faces_in_gray(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    faces: list[tuple[int, int, int, int]] = []
    for cascade, flip in _face_cascades():
        work = cv2.flip(gray, 1) if flip else gray
        detected = cascade.detectMultiScale(
            work,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(24, 24),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in detected:
            if flip:
                x = gray.shape[1] - x - w
            faces.append((int(x), int(y), int(w), int(h)))
    return faces


def _skin_fraction(patch: np.ndarray) -> float:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 18, 55), (28, 255, 255)) | cv2.inRange(
        hsv, (160, 18, 55), (180, 255, 255)
    )
    return float(mask.mean()) / 255.0


def _faces_in_patch(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    return _detect_faces_in_gray(gray)


def _valid_streamer_face(frame: np.ndarray, face: tuple[int, int, int, int]) -> bool:
    """Haar cascades false-positive on doorways/UI — require skin in the face patch."""
    x, y, w, h = face
    fh, fw = frame.shape[:2]
    if w * h < 500:
        return False
    patch = frame[max(0, y) : min(fh, y + h), max(0, x) : min(fw, x + w)]
    if patch.size == 0:
        return False
    return _skin_fraction(patch) >= 0.10


def _person_anchor_in_bounds(
    frame: np.ndarray, bounds: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Find the streamer via skin-tone clustering when profile face detect fails."""
    x1, y1, x2, y2 = bounds
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 18, 55), (28, 255, 255)) | cv2.inRange(
        hsv, (160, 18, 55), (180, 255, 255)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    _num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for label in range(1, _num_labels):
        bx, by, bw, bh, area = stats[label]
        if area < 220:
            continue
        if area > best_area:
            best_area = area
            best = (x1 + bx, y1 + by, bw, bh)

    return best


def _score_overlay_bounds(
    frames: list[np.ndarray],
    var_map: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> float:
    """Prefer stable regions that look like a person webcam, not static UI menus."""
    x1, y1, x2, y2 = bounds
    if x2 <= x1 or y2 <= y1:
        return -1.0

    roi_var = var_map[y1:y2, x1:x2]
    stability = 1.0 / (float(roi_var.mean()) + 0.5)
    patch = frames[0][y1:y2, x1:x2]
    skin = _skin_fraction(patch)

    gray = cv2.equalizeHist(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
    faces = [
        face
        for face in _faces_in_patch(gray)
        if _valid_streamer_face(frames[0], (x1 + face[0], y1 + face[1], face[2], face[3]))
    ]
    has_face = len(faces) > 0

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    text_like = float(edges.mean()) / 255.0

    if skin < 0.045 and not has_face:
        return -1.0

    area = (x2 - x1) * (y2 - y1)
    frame_area = frames[0].shape[0] * frames[0].shape[1]
    rel_area = area / frame_area
    if rel_area < MIN_OVERLAY_AREA or rel_area > MAX_OVERLAY_AREA:
        return -1.0

    return stability + skin * 18.0 + (5.0 if has_face else 0.0) - text_like * 3.0


def _best_face_rect(
    frames: list[np.ndarray],
    var_map: np.ndarray,
    corner_hint: str,
) -> tuple[int, int, int, int] | None:
    """Pick the face inside the most stable, skin-toned corner region."""
    hint = normalize_face_cam_corner(corner_hint)
    corners = [hint] if hint != "auto" else list(
        ("top-left", "top-right", "bottom-left", "bottom-right")
    )

    best_face: tuple[int, int, int, int] | None = None
    best_score = -1.0
    frame_h, frame_w = frames[0].shape[:2]

    for corner in corners:
        zone = _corner_zone(corner, frame_w, frame_h)
        static = _static_overlay_bounds_in_zone(frames, var_map, zone)
        if static is None:
            continue
        sx1, sy1, sx2, sy2 = static
        patch = frames[0][sy1:sy2, sx1:sx2]
        gray = cv2.equalizeHist(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
        region_score = _score_overlay_bounds(frames, var_map, static)
        if region_score < 0:
            continue

        gray = cv2.equalizeHist(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
        for x, y, w, h in _faces_in_patch(gray):
            abs_face = (sx1 + x, sy1 + y, w, h)
            if not _valid_streamer_face(frames[0], abs_face):
                continue
            face_score = region_score + (w * h) / 10_000.0
            if face_score > best_score:
                best_score = face_score
                best_face = abs_face

        anchor = _person_anchor_in_bounds(frames[0], static)
        if anchor is not None:
            ax, ay, aw, ah = anchor
            anchor_score = region_score + (aw * ah) / 8000.0
            if anchor_score > best_score:
                best_score = anchor_score
                best_face = anchor

    if best_face is not None:
        return best_face

    # Fallback: score validated faces by local stability + skin around them.
    for frame in frames:
        fh, fw = frame.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for x, y, w, h in _detect_faces_in_gray(gray):
            if not _is_overlay_face(x, y, w, h, fw, fh):
                continue
            if not _valid_streamer_face(frame, (x, y, w, h)):
                continue
            pad_x, pad_y = int(w * 1.2), int(h * 1.4)
            bx1 = max(0, x - pad_x)
            by1 = max(0, y - pad_y)
            bx2 = min(fw, x + w + pad_x)
            by2 = min(fh, y + h + pad_y)
            score = _score_overlay_bounds(frames, var_map, (bx1, by1, bx2, by2))
            if score > best_score:
                best_score = score
                best_face = (x, y, w, h)

    return best_face


def _temporal_variance_map(frames: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.var(stack, axis=0).mean(axis=2)


def _gameplay_strip_score(patch: np.ndarray) -> float:
    """Higher = more likely game menu/UI, not webcam interior."""
    if patch.size == 0:
        return 1.0
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    menu = cv2.inRange(hsv, (85, 12, 12), (140, 200, 150))
    dark = cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 0, 68)
    return max(float(menu.mean()), float(dark.mean()) * 0.9) / 255.0


def _minimap_or_hud_strip_score(patch: np.ndarray) -> float:
    """Higher = Apex minimap / HUD stack above the webcam (not the cam itself)."""
    if patch.size == 0:
        return 0.0
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    if _skin_fraction(patch) >= 0.055:
        return 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    terrain = cv2.inRange(hsv, (18, 20, 30), (100, 255, 255))
    chrome = cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 5, 62)
    ui = cv2.inRange(hsv, (0, 0, 140), (180, 100, 255))
    return min(
        1.0,
        float(terrain.mean()) / 255.0 * 0.75
        + float(chrome.mean()) / 255.0 * 0.45
        + float(ui.mean()) / 255.0 * 0.25,
    )


def _zone_for_webcam_scan(
    face: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    corner: str,
) -> tuple[int, int, int, int]:
    """
    Search zone centered on the face — not the whole top-left corner.

    Apex often stacks minimap above webcam on the left edge; scanning the full
    corner rectangle pulls the minimap into the crop.
    """
    fx, fy, fw, fh = face
    if corner.endswith("left"):
        zx1, zx2 = 0, int(frame_w * 0.44)
    else:
        zx1, zx2 = int(frame_w * 0.56), frame_w

    zy1 = max(0, fy - int(fh * 3.2))
    zy2 = min(frame_h, fy + fh + int(fh * 2.8))
    return zx1, zy1, zx2, zy2


def _trim_hud_above_webcam(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    face: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Drop minimap / match UI sitting above the webcam box."""
    x1, y1, x2, y2 = bounds
    fx, fy, fw, fh = face
    face_mid = fy + fh / 2
    box_h = max(y2 - y1, 1)
    face_rel = (face_mid - y1) / box_h

    if y1 <= fy - int(fh * 0.35):
        return bounds

    new_y1 = max(y1, fy - int(fh * 0.38))
    if face_rel > 0.52 or box_h > fh * 1.8:
        pass
    elif box_h <= fh * 2.4 and face_rel <= 0.62:
        return bounds
    for row in range(int(fy - fh * 0.45), y1, -1):
        if row < 0:
            break
        strip_h = max(2, fh // 5)
        strip = _face_centered_strip(frame, row, face, x1, x2)
        if strip.shape[0] < strip_h:
            strip = frame[row : row + strip_h, x1:x2]
        if _minimap_or_hud_strip_score(strip) > 0.13:
            new_y1 = max(new_y1, row + strip_h)
            continue
        if _gameplay_strip_score(strip) > 0.30:
            new_y1 = max(new_y1, row + 2)
            break
        gray = float(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).mean())
        if gray > 82 and _minimap_or_hud_strip_score(strip) < 0.07:
            new_y1 = row

    y1 = max(y1, min(new_y1, fy - int(fh * 0.12)))
    if y2 - y1 < int(fh * 1.4):
        return bounds
    return x1, int(y1), x2, y2


def _clamp_bounds_to_webcam(
    bounds: tuple[int, int, int, int],
    face: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Cap bounds so we do not swallow the whole corner — never shrink below face box."""
    x1, y1, x2, y2 = bounds
    _fx, _fy, fw, fh = face
    max_w = int(frame_w * 0.30)
    # Left-edge OBS cams need extra height for signs above and chin below the face.
    max_h = int(frame_h * (0.38 if x1 <= 4 else 0.30))
    min_w = int(fw * 1.05)
    min_h = int(fh * 1.05)

    if x2 - x1 > max_w:
        x2 = x1 + max_w
    if y2 - y1 > max_h:
        y2 = y1 + max_h

    if x2 - x1 < min_w:
        cx = (x1 + x2) // 2
        half = min_w // 2
        x1, x2 = max(0, cx - half), min(frame_w, cx + half)
    if y2 - y1 < min_h:
        cy = (y1 + y2) // 2
        half = min_h // 2
        y1, y2 = max(0, cy - half), min(frame_h, cy + half)

    return (
        max(0, x1),
        max(0, y1),
        min(frame_w, x2),
        min(frame_h, y2),
    )


def _strip_is_webcam_interior(
    patch: np.ndarray,
    ref_brightness: float,
    *,
    strict_up: bool = False,
) -> bool:
    """True when a scan strip still looks like the OBS webcam, not gameplay."""
    if patch.size == 0 or patch.shape[0] < 1 or patch.shape[1] < 1:
        return False
    if _gameplay_strip_score(patch) > 0.40:
        return False

    skin = _skin_fraction(patch)
    bright = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean())
    if skin >= 0.045:
        return True

    if strict_up:
        if bright < 78:
            return False
        if abs(bright - ref_brightness) > 58:
            return False

    return bright >= 52


def _stable_threshold_for_zone(
    var_map: np.ndarray, zone: tuple[int, int, int, int]
) -> float:
    zx1, zy1, zx2, zy2 = zone
    roi_var = var_map[zy1:zy2, zx1:zx2]
    if roi_var.size == 0:
        return 1.0
    return float(np.percentile(roi_var, 28))


def _face_centered_strip(
    frame: np.ndarray,
    row: int,
    face: tuple[int, int, int, int],
    x1: int,
    x2: int,
) -> np.ndarray:
    """Row slice around the face — avoids webcam chrome borders on wide strips."""
    fx, _fy, fw, _fh = face
    cx1 = max(x1, fx - int(fw * 0.12))
    cx2 = min(x2, fx + fw + int(fw * 0.18))
    row = max(0, min(row, frame.shape[0] - 1))
    return frame[row : row + 1, cx1:cx2]


def _stability_band_rows(
    face: tuple[int, int, int, int], frame_h: int
) -> tuple[int, int]:
    """Rows used to test column stability — centered on the face."""
    _fx, fy, _fw, fh = face
    return max(0, fy - int(fh * 0.45)), min(frame_h, fy + int(fh * 2.0))


def _snap_left_webcam_edge(
    var_map: np.ndarray | None,
    face: tuple[int, int, int, int],
    corner: str,
    frame_w: int,
    frame_h: int,
    stable_limit: float,
) -> int | None:
    """Left-edge overlays (Apex) often touch x=0 even past a thin HUD gap."""
    if var_map is None or not corner.endswith("left"):
        return None
    fx, _fy, _fw, _fh = face
    if fx > frame_w * 0.22:
        return None
    band_y1, band_y2 = _stability_band_rows(face, frame_h)
    edge_var = float(var_map[band_y1:band_y2, 0].mean())
    if edge_var <= stable_limit:
        return 0
    return None


def _grow_webcam_bounds(
    frame: np.ndarray,
    var_map: np.ndarray | None,
    face: tuple[int, int, int, int],
    corner: str,
) -> tuple[int, int, int, int]:
    """
    Grow from the face to the OBS webcam rectangle.

    Horizontal and downward growth follow temporal stability — the webcam
    overlay is static while gameplay around it changes. Upward growth also
    rejects minimap/HUD strips stacked above the cam (common in Apex).
    """
    frame_h, frame_w = frame.shape[:2]
    fx, fy, fw, fh = face
    zone = _zone_for_webcam_scan(face, frame_w, frame_h, corner)
    zx1, zy1, zx2, zy2 = zone
    stable_thresh = (
        _stable_threshold_for_zone(var_map, zone) if var_map is not None else 1e9
    )
    stable_limit = stable_thresh * 1.38
    spike_limit = stable_limit * 2.6

    x1, x2 = fx, fx + fw
    y1, y2 = fy, fy + fh

    def _column_stable(col: int, row_a: int, row_b: int) -> bool:
        if var_map is None or col < zx1 or col >= zx2:
            return var_map is None
        return float(var_map[row_a:row_b, col].mean()) <= stable_limit

    def _row_stable(row: int, col_a: int, col_b: int) -> bool:
        if var_map is None or row < zy1 or row >= zy2:
            return var_map is None
        return float(var_map[row, col_a:col_b].mean()) <= stable_limit

    down_limit = min(zy2, fy + fh + int(fh * 2.15))
    min_y2 = fy + fh + max(4, int(fh * 0.08))
    while y2 < down_limit:
        if var_map is not None and not _row_stable(y2, x1, x2):
            break
        if y2 >= min_y2 and y2 > fy + int(fh * 0.85):
            strip = _face_centered_strip(frame, y2, face, x1, x2)
            if _gameplay_strip_score(strip) > 0.32:
                break
        y2 += 1
    y2 = max(y2, min_y2)

    band_y1, band_y2 = _stability_band_rows(face, frame_h)
    while x1 > zx1:
        if var_map is not None and not _column_stable(x1 - 1, band_y1, band_y2):
            break
        x1 -= 1

    snapped_left = _snap_left_webcam_edge(
        var_map, face, corner, frame_w, frame_h, stable_limit
    )
    if snapped_left is not None:
        x1 = snapped_left

    up_limit = max(zy1, fy - int(fh * 2.4))
    unstable_gap = 0
    max_unstable_gap = max(14, int(fh * 0.32))
    while y1 > up_limit:
        row = y1 - 1
        if var_map is not None and not _row_stable(row, x1, x2):
            unstable_gap += 1
            if unstable_gap > max_unstable_gap:
                break
        else:
            unstable_gap = 0
        strip = _face_centered_strip(frame, row, face, x1, x2)
        if _minimap_or_hud_strip_score(strip) > 0.30:
            break
        y1 -= 1

    right_limit = min(zx2, int(frame_w * 0.30))
    min_right = fx + int(fw * 1.05)
    col_stable_limit = stable_limit * 1.75
    while x2 < right_limit:
        if var_map is None:
            x2 += 1
            continue
        col_var = float(var_map[y1:y2, x2].mean())
        if x2 >= min_right and col_var > spike_limit:
            break
        if col_var > col_stable_limit:
            break
        x2 += 1

    bounds = _trim_hud_above_webcam(frame, (x1, y1, x2, y2), face)
    bounds = _trim_gameplay_below_webcam(frame, bounds, face)
    return _clamp_bounds_to_webcam(bounds, face, frame_w, frame_h)


def _scan_webcam_bounds_from_face(
    frames: list[np.ndarray],
    face: tuple[int, int, int, int],
    corner_hint: str,
    *,
    var_map: np.ndarray | None = None,
    reference_frame: np.ndarray | None = None,
) -> tuple[int, int, int, int] | None:
    """
    From the detected face, scan up/down/left/right until we hit webcam edges.

    Uses temporal stability (webcam is static) plus minimap rejection above
    the face so gameplay and HUD stacks are not included.
    """
    if not frames:
        return None

    frame = reference_frame if reference_frame is not None else frames[0]
    frame_h, frame_w = frame.shape[:2]
    if var_map is None:
        var_map = _temporal_variance_map(frames) if len(frames) >= 2 else None
    fx, fy, fw, fh = face
    fcx = (fx + fw / 2) / frame_w
    fcy = (fy + fh / 2) / frame_h
    hint = normalize_face_cam_corner(corner_hint)
    corner = hint if hint != "auto" else _corner_for_point(fcx, fcy)

    bounds = _grow_webcam_bounds(frame, var_map, face, corner)

    if bounds[2] - bounds[0] < 36 or bounds[3] - bounds[1] < 36:
        return None
    return bounds


def _trim_gameplay_below_webcam(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    face: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Cut gameplay/HUD rows that leaked below the OBS webcam border."""
    x1, y1, x2, y2 = bounds
    fx, fy, fw, fh = face
    scan_from = max(y1 + int(fh * 1.1), fy + int(fh * 0.55))
    new_y2 = y2
    right_x = int(x1 + (x2 - x1) * 0.55)

    for row in range(y2 - 1, scan_from - 1, -1):
        strip = _face_centered_strip(frame, row, face, x1, x2)
        right_strip = frame[row : row + 2, right_x:x2]
        if (
            _gameplay_strip_score(strip) > 0.34
            or _gameplay_strip_score(right_strip) > 0.28
        ):
            new_y2 = min(new_y2, row)
            break

    if new_y2 <= scan_from:
        if new_y2 - y1 < int(fh * 1.2):
            return bounds
        return x1, y1, x2, int(new_y2)

    gray = cv2.cvtColor(frame[scan_from:new_y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if gray.size > 0:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 110)
        row_density = edges.mean(axis=1)
        best_i = 0
        best_score = -1.0
        for index, score in enumerate(row_density):
            if float(score) > best_score:
                best_score = float(score)
                best_i = index
        if best_score > 18:
            edge_y2 = scan_from + best_i + 2
            if edge_y2 > y1 + fh:
                new_y2 = min(new_y2, edge_y2)

    new_y2 = min(y2, max(new_y2, fy + fh))

    if new_y2 - y1 < int(fh * 1.2):
        return bounds
    return x1, y1, x2, int(new_y2)


def _fit_webcam_overlay_frame(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    face: tuple[int, int, int, int],
    corner: str,
) -> tuple[int, int, int, int]:
    """Snap the left edge to the OBS overlay border — never grow into gameplay."""
    x1, y1, x2, y2 = bounds
    frame_h, frame_w = frame.shape[:2]
    _fx, _fy, fw, fh = face
    pad = max(8, int(fw * 0.15))
    sx1 = max(0, x1 - pad)
    sy1 = max(0, y1 - pad)
    sx2 = min(frame_w, x2 + pad)
    sy2 = min(frame_h, y2 + pad)
    patch = frame[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return bounds

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    col_density = edges.mean(axis=0)
    row_density = edges.mean(axis=1)

    def _peak_index(
        density: np.ndarray, start: int, end: int, *, min_score: float = 8.0
    ) -> int | None:
        if density.size == 0:
            return None
        lo = max(0, min(start, end))
        hi = min(density.shape[0] - 1, max(start, end))
        step = 1 if end >= start else -1
        best_i: int | None = None
        best_score = min_score
        for index in range(lo, hi + step, step):
            score = float(density[index])
            if score > best_score:
                best_score = score
                best_i = index
        return best_i

    _fx, fy, _fw, fh = face
    nx1, nx2, ny1, ny2 = x1, x2, y1, y2

    left_local = x1 - sx1
    left_peak = _peak_index(col_density, 0, left_local)
    if left_peak is not None:
        nx1 = max(0, sx1 + left_peak)

    bottom_local = y2 - sy1
    search_start = max(0, bottom_local - int((y2 - y1) * 0.55))
    bottom_peak = _peak_index(row_density, search_start, bottom_local, min_score=16.0)
    if bottom_peak is not None:
        edge_y2 = sy1 + bottom_peak + 2
        right_x = int(x1 + (x2 - x1) * 0.55)
        right_strip = frame[edge_y2 : edge_y2 + 2, right_x:x2]
        # Trim to OBS border only when gameplay leaks past it — keep chin otherwise.
        if (
            _gameplay_strip_score(right_strip) > 0.22
            and fy + fh <= edge_y2 <= y2
        ):
            ny2 = edge_y2

    if corner.endswith("left") and nx1 <= 3:
        nx1 = 0

    chrome = max(2, int(fw * 0.04))
    nx1 = max(0, nx1 - chrome)
    nx2 = min(x2, nx2)
    ny1 = max(0, min(ny1, fy - int(fh * 0.42)))
    ny2 = min(frame_h, max(ny2, fy + int(fh * 1.12)))

    if nx2 - nx1 < 30 or ny2 - ny1 < 30:
        return bounds
    return int(nx1), int(ny1), int(nx2), int(ny2)


def _expand_webcam_room_for_decor(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    face: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Grow vertically for signs above and chin below — never into gameplay columns."""
    x1, y1, x2, y2 = bounds
    fx, fy, fw, fh = face
    frame_h, frame_w = frame.shape[:2]
    ref_patch = frame[fy : fy + fh, fx : fx + fw]
    ref_bright = float(cv2.cvtColor(ref_patch, cv2.COLOR_BGR2GRAY).mean())

    new_y1 = y1
    up_cap = max(0, fy - int(fh * 0.55))
    for row in range(y1 - 1, up_cap - 1, -1):
        strip = frame[row : row + 2, x1:x2]
        if _minimap_or_hud_strip_score(strip) > 0.16:
            break
        if _gameplay_strip_score(strip) > 0.22:
            break
        if _strip_is_webcam_interior(strip, ref_bright, strict_up=True):
            new_y1 = row
        else:
            break

    new_y2 = y2
    cx1 = max(x1, fx - int(fw * 0.08))
    cx2 = min(x2, fx + fw + int(fw * 0.08))
    down_cap = min(frame_h, fy + int(fh * 1.45))
    for row in range(y2, down_cap):
        strip = frame[row : row + 2, cx1:cx2]
        if _gameplay_strip_score(strip) > 0.26:
            break
        if _strip_is_webcam_interior(strip, ref_bright):
            new_y2 = row + 2
        else:
            break

    if new_y2 - new_y1 < int(fh * 1.15):
        return bounds
    return x1, int(new_y1), x2, int(min(frame_h, new_y2))


def _sanitize_camera_bounds(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    face: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Trim bottom/right edges where gameplay leaks past the OBS webcam border."""
    x1, y1, x2, y2 = bounds
    _fx, fy, _fw, fh = face
    min_h = int(fh * 1.2)
    min_w = int(fh * 1.8)

    right_x = int(x1 + (x2 - x1) * 0.55)
    edge_band = max(10, int((y2 - y1) * 0.14))
    scan_from_y = max(y1 + min_h, y2 - edge_band)
    new_y2 = y2
    for row in range(y2 - 1, scan_from_y - 1, -1):
        right_strip = frame[row : row + 2, right_x:x2]
        if _gameplay_strip_score(right_strip) > 0.30:
            new_y2 = row
            break

    new_y2 = max(new_y2, y1 + min_h, fy + fh)

    corner_h = max(8, int((new_y2 - y1) * 0.12))
    bot_y = new_y2 - corner_h
    edge_cols = max(10, int((x2 - x1) * 0.08))
    scan_from_x = max(x1 + min_w, x2 - edge_cols)
    new_x2 = x2
    for col in range(x2 - 1, scan_from_x - 1, -1):
        corner_patch = frame[bot_y:new_y2, col : col + 3]
        if _gameplay_strip_score(corner_patch) > 0.46:
            new_x2 = col
        else:
            break

    new_x2 = max(new_x2, x1 + min_w)
    if new_x2 - x1 < min_w or new_y2 - y1 < min_h:
        return bounds
    return x1, y1, int(new_x2), int(new_y2)


def _expand_full_obs_webcam(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    face: tuple[int, int, int, int],
    corner: str,
    var_map: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """
    Anchor the crop to the OBS overlay edge (full-width webcam on the left).

    Aspect ratio is handled at render time — detection stays camera-only and never
    grows downward into gameplay to chase 16:9.
    """
    _ = frame, var_map
    x1, y1, x2, y2 = bounds
    frame_w = frame.shape[1]
    _fx, _fy, _fw, fh = face

    if corner.endswith("left"):
        x1 = 0
    elif corner.endswith("right"):
        x2 = frame_w

    if x2 - x1 < int(fh * 1.2) or y2 - y1 < int(fh * 1.2):
        return bounds
    return int(x1), int(y1), int(x2), int(y2)


def _pixels_to_region(
    bounds: tuple[int, int, int, int], frame_w: int, frame_h: int
) -> FaceCamRegion:
    x1, y1, x2, y2 = bounds
    return _clamp_region(
        FaceCamRegion(
            x=x1 / frame_w,
            y=y1 / frame_h,
            w=(x2 - x1) / frame_w,
            h=(y2 - y1) / frame_h,
        )
    )


def _validate_overlay(
    region: FaceCamRegion,
    face: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> bool:
    area = _region_area(region)
    if area < MIN_OVERLAY_AREA or area > MAX_OVERLAY_AREA:
        return False
    aspect = region.w / max(region.h, 1e-6)
    if aspect < 0.55 or aspect > 2.8:
        return False
    fx, fy, fw, fh = face
    fcx = (fx + fw / 2) / frame_w
    fcy = (fy + fh / 2) / frame_h
    return _contains_point(region, fcx, fcy)


def _overlay_from_face(
    frames: list[np.ndarray],
    face: tuple[int, int, int, int],
    corner_hint: str,
    *,
    var_map: np.ndarray | None = None,
    reference_frame: np.ndarray | None = None,
) -> FaceCamRegion | None:
    """Scan from the face outward until we hit the webcam overlay edges."""
    frame = reference_frame if reference_frame is not None else frames[0]
    bounds = _scan_webcam_bounds_from_face(
        frames,
        face,
        corner_hint,
        var_map=var_map,
        reference_frame=frame,
    )
    if bounds is None:
        return None

    frame_h, frame_w = frames[0].shape[:2]
    fcx = (face[0] + face[2] / 2) / frame_w
    fcy = (face[1] + face[3] / 2) / frame_h
    hint = normalize_face_cam_corner(corner_hint)
    corner = hint if hint != "auto" else _corner_for_point(fcx, fcy)

    bounds = _trim_hud_above_webcam(frame, bounds, face)
    bounds = _trim_gameplay_below_webcam(frame, bounds, face)
    bounds = _expand_webcam_room_for_decor(frame, bounds, face)
    bounds = _fit_webcam_overlay_frame(frame, bounds, face, corner)
    bounds = _sanitize_camera_bounds(frame, bounds, face)
    bounds = _expand_full_obs_webcam(frame, bounds, face, corner, var_map=var_map)
    bounds = _sanitize_camera_bounds(frame, bounds, face)
    bounds = _clamp_bounds_to_webcam(bounds, face, frame_w, frame_h)
    region = _pixels_to_region(bounds, frame_w, frame_h)
    if _validate_overlay(region, face, frame_w, frame_h):
        return region
    return None


def _static_overlay_bounds_in_zone(
    frames: list[np.ndarray],
    var_map: np.ndarray,
    zone: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Largest stable rectangle inside a corner search zone."""
    if len(frames) < 3:
        return None
    zx1, zy1, zx2, zy2 = zone
    roi_var = var_map[zy1:zy2, zx1:zx2]
    if roi_var.size == 0:
        return None

    threshold = float(np.percentile(roi_var, 25))
    mask = (roi_var <= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    _num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    best_bounds: tuple[int, int, int, int] | None = None
    best_score = -1.0
    for label in range(1, _num_labels):
        bx, by, bw, bh, area = stats[label]
        if area < 400:
            continue
        rel = (bw * bh) / max(1, (zx2 - zx1) * (zy2 - zy1))
        if rel < 0.06 or rel > 0.95:
            continue
        bounds = (zx1 + bx, zy1 + by, zx1 + bx + bw, zy1 + by + bh)
        patch = frames[0][bounds[1] : bounds[3], bounds[0] : bounds[2]]
        skin = _skin_fraction(patch)
        score = skin * 20.0 + min(rel, 0.5)
        if score > best_score:
            best_score = score
            best_bounds = bounds

    return best_bounds


def _static_overlay_in_corner(
    frames: list[np.ndarray],
    corner: str,
    var_map: np.ndarray | None = None,
) -> FaceCamRegion | None:
    """Stable rectangle in one corner."""
    if len(frames) < 3:
        return None
    frame_h, frame_w = frames[0].shape[:2]
    if var_map is None:
        var_map = _temporal_variance_map(frames)
    zone = _corner_zone(corner, frame_w, frame_h)
    best_bounds = _static_overlay_bounds_in_zone(frames, var_map, zone)
    if best_bounds is None:
        return None
    region = _pixels_to_region(best_bounds, frame_w, frame_h)
    if _region_area(region) < MIN_OVERLAY_AREA:
        return None
    return region


def _best_overlay_candidate(
    frames: list[np.ndarray],
    face: tuple[int, int, int, int] | None,
    corner_hint: str,
) -> tuple[FaceCamRegion | None, str]:
    """Face-first edge scan; static corner regions are fallback only."""
    if face is not None:
        scanned = _overlay_from_face(frames, face, corner_hint)
        if scanned is not None:
            return scanned, "face+overlay"

    var_map = _temporal_variance_map(frames)
    frame_h, frame_w = frames[0].shape[:2]
    hint = normalize_face_cam_corner(corner_hint)
    corners = [hint] if hint != "auto" else list(
        ("top-left", "top-right", "bottom-left", "bottom-right")
    )

    best: tuple[FaceCamRegion, float, str] | None = None
    for corner in corners:
        zone = _corner_zone(corner, frame_w, frame_h)
        bounds = _static_overlay_bounds_in_zone(frames, var_map, zone)
        if bounds is None:
            continue
        score = _score_overlay_bounds(frames, var_map, bounds)
        if score < 0:
            continue
        region = _pixels_to_region(bounds, frame_w, frame_h)
        if best is None or score > best[1]:
            best = (region, score, "static+person")

    if best is None:
        return None, "none"
    return best[0], best[2]


def _save_debug_overlay(
    frame: np.ndarray,
    face: tuple[int, int, int, int] | None,
    region: FaceCamRegion | None,
    output_path: Path,
    *,
    panel_bounds: tuple[int, int, int, int] | None = None,
) -> None:
    debug = frame.copy()
    if face is not None:
        fx, fy, fw, fh = face
        cv2.rectangle(debug, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    if region is not None:
        h, w = debug.shape[:2]
        x1 = int(region.x * w)
        y1 = int(region.y * h)
        x2 = int((region.x + region.w) * w)
        y2 = int((region.y + region.h) * h)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 128, 255), 2)
    cv2.imwrite(str(output_path), debug)


def _frames_near(
    frames: list[np.ndarray],
    target: np.ndarray | None,
    *,
    max_count: int = 5,
) -> list[np.ndarray]:
    """Pick frames closest to the reference layout for stable edge scanning."""
    if not frames:
        return []
    if target is None:
        return frames[:max_count]

    best_idx = 0
    best_dist = float("inf")
    target_f = target.astype(np.float32)
    for index, frame in enumerate(frames):
        if frame.shape != target.shape:
            continue
        dist = float(np.mean((frame.astype(np.float32) - target_f) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_idx = index

    start = max(0, best_idx - 2)
    end = min(len(frames), best_idx + 3)
    picked = list(frames[start:end])
    while len(picked) < 3:
        picked.append(frames[best_idx])
    return picked[:max_count]


def _best_in_game_face_frame(
    frames: list[np.ndarray],
    var_map: np.ndarray,
    corner_hint: str,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None]:
    """Prefer a left-edge in-game webcam frame (avoids menu-only layouts)."""
    best_face: tuple[int, int, int, int] | None = None
    best_frame: np.ndarray | None = None
    best_score = -1.0

    for frame in frames:
        fh, fw = frame.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for x, y, w, h in _detect_faces_in_gray(gray):
            if not _valid_streamer_face(frame, (x, y, w, h)):
                continue
            if x > fw * 0.38:
                continue
            cy = (y + h / 2) / fh
            if cy < 0.30:
                continue
            size_score = (w * h) / 10_000.0
            pos_score = 2.0 if 0.36 <= cy <= 0.58 else 1.0
            score = size_score + pos_score
            if score > best_score:
                best_score = score
                best_face = (x, y, w, h)
                best_frame = frame

    if best_face is not None:
        return best_face, best_frame

    face = _best_face_rect(frames, var_map, corner_hint=corner_hint)
    return face, frames[0] if frames else None


def detect_face_cam_region(
    video_path: Path,
    work_dir: Path,
    sample_count: int = 10,
    ffmpeg: str = "ffmpeg",
    corner_hint: str = "auto",
) -> tuple[FaceCamRegion | None, str]:
    """
    1. Detect the streamer's face to learn which corner the webcam is in.
    2. Scan from the face outward until we hit the webcam overlay edges.
    3. Exclude minimap/HUD stacked above the cam (common in Apex VODs).
    """
    from twitch_tiktok_bot.analyze.duration import get_video_duration

    frame_dir = work_dir / "face_samples"
    frame_paths = _sample_frame_paths(
        video_path, frame_dir, sample_count=sample_count, ffmpeg=ffmpeg
    )
    if not frame_paths:
        return None, "none"

    frames = _load_frames(frame_paths)
    if not frames:
        return None, "none"

    duration = get_video_duration(video_path, ffmpeg=ffmpeg)
    var_map = _temporal_variance_map(frames)

    if duration > 120:
        face, debug_frame = _best_in_game_face_frame(frames, var_map, corner_hint)
    else:
        face = _best_face_rect(frames, var_map, corner_hint=corner_hint)
        debug_frame = frames[0]

    if face is None:
        return _best_overlay_candidate(frames, None, corner_hint)

    scan_frames = (
        _frames_near(frames, debug_frame)
        if duration > 120
        else frames
    )
    grow_frame = debug_frame if debug_frame is not None else scan_frames[0]
    region = _overlay_from_face(
        scan_frames,
        face,
        corner_hint,
        var_map=var_map,
        reference_frame=grow_frame,
    )
    if region is not None:
        debug_path = frame_dir / "overlay_debug.jpg"
        _save_debug_overlay(
            debug_frame if debug_frame is not None else frames[0],
            face,
            region,
            debug_path,
        )
        return region, "face+overlay"

    return _best_overlay_candidate(frames, face, corner_hint)


def detect_face_crop_center(
    video_path: Path,
    work_dir: Path,
    sample_count: int = 8,
    ffmpeg: str = "ffmpeg",
    corner_hint: str = "auto",
) -> float | None:
    """Legacy horizontal center for single-crop layout."""
    region, _method = detect_face_cam_region(
        video_path,
        work_dir,
        sample_count=sample_count,
        ffmpeg=ffmpeg,
        corner_hint=corner_hint,
    )
    if region:
        return region.x + region.w / 2
    return None


@dataclass
class FaceCamLayout:
    face_crop_center_x: float | None
    face_cam_region: dict[str, float] | None
    method: str = ""


def should_detect_face_cam(config: AppConfig) -> bool:
    """Stacked TikTok layout always needs a face-cam overlay box."""
    render = config.render
    return render.face_crop_enabled or render.layout.lower() == "stacked"


def resolve_face_cam_layout(
    video_path: Path,
    work_dir: Path,
    config: AppConfig,
    ffmpeg: str = "ffmpeg",
) -> FaceCamLayout:
    """
    Single face-cam algorithm for clips, VODs, and all future editors.

    Detects the overlay box, falls back to config override, then corner preset.
    """
    render = config.render
    if not should_detect_face_cam(config):
        return FaceCamLayout(None, None, "disabled")

    override = render.face_cam_override
    region: FaceCamRegion | None = None
    method = ""

    if override:
        region = FaceCamRegion.from_dict(override)
        if region:
            method = "override"
    else:
        region, method = detect_face_cam_region(
            video_path,
            work_dir,
            sample_count=render.face_sample_count,
            ffmpeg=ffmpeg,
            corner_hint=render.face_cam_corner,
        )

    if region:
        return FaceCamLayout(
            face_crop_center_x=region.x + region.w / 2,
            face_cam_region=region.to_dict(),
            method=method,
        )

    if render.layout.lower() == "stacked":
        fallback = default_face_cam_region(render.face_cam_corner)
        return FaceCamLayout(
            face_crop_center_x=fallback.x + fallback.w / 2,
            face_cam_region=fallback.to_dict(),
            method="default",
        )

    center = detect_face_crop_center(
        video_path,
        work_dir,
        sample_count=render.face_sample_count,
        ffmpeg=ffmpeg,
        corner_hint=render.face_cam_corner,
    )
    return FaceCamLayout(face_crop_center_x=center, face_cam_region=None, method=method)


def resolve_face_cam_for_render(
    analysis: ClipAnalysis,
    config: AppConfig,
) -> tuple[float | None, FaceCamRegion | None]:
    """
    Resolve face cam for FFmpeg render — works with old analysis files too.
    """
    region = FaceCamRegion.from_dict(analysis.face_cam_region)
    if region:
        center = analysis.face_crop_center_x or (region.x + region.w / 2)
        return center, region

    override = config.render.face_cam_override
    if override:
        region = FaceCamRegion.from_dict(override)
        if region:
            return region.x + region.w / 2, region

    if config.render.layout.lower() == "stacked":
        fallback = default_face_cam_region(config.render.face_cam_corner)
        return fallback.x + fallback.w / 2, fallback

    if analysis.face_crop_center_x is not None:
        return analysis.face_crop_center_x, None

    return None, None
