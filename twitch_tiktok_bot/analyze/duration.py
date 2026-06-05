"""Video duration helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_video_duration(video_path: Path, ffmpeg: str = "ffmpeg") -> float:
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if ffprobe == ffmpeg:
        ffprobe = "ffprobe"
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())

    # Fallback: parse ffmpeg stderr
    cmd = [ffmpeg, "-i", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            part = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = part.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"Could not determine duration for {video_path}")
