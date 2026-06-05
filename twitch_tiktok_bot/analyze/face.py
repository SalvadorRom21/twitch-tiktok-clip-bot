"""Face detection for smart vertical crop positioning."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def _sample_frame_paths(
    video_path: Path,
    output_dir: Path,
    sample_count: int = 8,
    ffmpeg: str = "ffmpeg",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "face_%04d.jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_count}",
        "-frames:v",
        str(sample_count),
        "-q:v",
        "3",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    return sorted(output_dir.glob("face_*.jpg"))


def detect_face_crop_center(
    video_path: Path,
    work_dir: Path,
    sample_count: int = 8,
    ffmpeg: str = "ffmpeg",
) -> float | None:
    """
    Return normalized horizontal face center (0.0-1.0) for crop positioning.
    None if no faces detected.
    """
    frame_dir = work_dir / "face_samples"
    frame_paths = _sample_frame_paths(
        video_path, frame_dir, sample_count=sample_count, ffmpeg=ffmpeg
    )
    if not frame_paths:
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None

    centers: list[float] = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )
        if len(faces) == 0:
            continue
        # Use largest face in frame
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        center_x = (x + w / 2) / image.shape[1]
        centers.append(float(center_x))

    if not centers:
        return None
    return float(np.median(centers))
