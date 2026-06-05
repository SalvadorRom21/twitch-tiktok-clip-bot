"""Download Twitch clips and VODs via yt-dlp."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _format_section(start_sec: float | None, end_sec: float | None) -> str | None:
    if start_sec is None and end_sec is None:
        return None

    def _fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    start = _fmt(start_sec or 0.0)
    end = _fmt(end_sec) if end_sec is not None else ""
    return f"*{start}-{end}" if end else f"*{start}-"


def download_video(
    url: str,
    output_dir: Path,
    video_id: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> Path:
    """Download a Twitch clip or VOD URL to output_dir and return the local file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_id or "video"
    output_template = str(output_dir / f"{stem}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "-f",
        "best[height<=1080][ext=mp4]/best[height<=1080]/best[ext=mp4]/best",
        "-o",
        output_template,
    ]
    section = _format_section(start_sec, end_sec)
    if section:
        cmd += ["--download-sections", section]
    cmd.append(url)

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


def download_clip(url: str, output_dir: Path, clip_id: str | None = None) -> Path:
    """Backward-compatible clip download wrapper."""
    return download_video(url, output_dir, video_id=clip_id)
