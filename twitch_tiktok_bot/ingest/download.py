"""Download Twitch clips and VODs via yt-dlp."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm"}


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


def find_cached_video(output_dir: Path, video_id: str | None = None) -> Path | None:
    """Return an already-downloaded video in output_dir, if present."""
    if not output_dir.is_dir():
        return None

    stem = video_id or "video"
    candidates = sorted(output_dir.glob(f"{stem}.*"))
    video_files = [p for p in candidates if p.suffix.lower() in VIDEO_EXTENSIONS]
    if video_files:
        return video_files[0]

    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def resolve_video_path(
    url: str,
    output_dir: Path,
    video_id: str | None = None,
    *,
    skip_download: bool = False,
    redownload: bool = False,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[Path, bool]:
    """
    Return (local_video_path, used_cache).

    Reuses an existing file in output_dir unless redownload is True.
    When skip_download is True, never call yt-dlp (fails if no cache).
    """
    cached = find_cached_video(output_dir, video_id)
    if skip_download:
        if cached is None:
            raise FileNotFoundError(
                f"No cached video in {output_dir}. "
                f"Download first, or omit --skip-download / --from-cache."
            )
        return cached, True
    if cached is not None and not redownload:
        return cached, True
    if redownload and cached is not None:
        cached.unlink()
    downloaded = download_video(
        url,
        output_dir,
        video_id=video_id,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    return downloaded, False


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
        sys.executable,
        "-m",
        "yt_dlp",
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

    print("       downloading — yt-dlp progress below:", flush=True)
    result = subprocess.run(cmd, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (exit {result.returncode})")

    candidates = sorted(output_dir.glob(f"{stem}.*"))
    video_files = [p for p in candidates if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        raise FileNotFoundError(f"No video file found after download in {output_dir}")
    return video_files[0]


def download_clip(url: str, output_dir: Path, clip_id: str | None = None) -> Path:
    """Backward-compatible clip download wrapper."""
    return download_video(url, output_dir, video_id=clip_id)
