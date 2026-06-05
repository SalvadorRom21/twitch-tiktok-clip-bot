"""Download Twitch clips via yt-dlp."""

from __future__ import annotations

import subprocess
from pathlib import Path


def download_clip(url: str, output_dir: Path, clip_id: str | None = None) -> Path:
    """Download a Twitch clip URL to output_dir and return the local file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = clip_id or "clip"
    output_template = str(output_dir / f"{stem}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "-f",
        "best[ext=mp4]/best",
        "-o",
        output_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\n{result.stderr or result.stdout}"
        )

    candidates = sorted(output_dir.glob(f"{stem}.*"))
    video_files = [p for p in candidates if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
    if not video_files:
        raise FileNotFoundError(f"No video file found after download in {output_dir}")
    return video_files[0]
