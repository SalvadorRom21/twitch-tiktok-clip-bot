"""FFmpeg helpers for trimming vertical shorts and VOD windows."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from twitch_tiktok_bot.analyze.duration import get_video_duration
from twitch_tiktok_bot.config import AppConfig

ProgressCb = Callable[[dict], None]


def _ffmpeg(config: AppConfig) -> str:
    return config.render.ffmpeg_path or "ffmpeg"


def probe_duration(config: AppConfig, video_path: Path) -> float:
    return get_video_duration(video_path, ffmpeg=_ffmpeg(config))


def trim_media(
    config: AppConfig,
    source: Path,
    output: Path,
    *,
    start_sec: float,
    end_sec: float,
    reencode: bool = True,
    on_progress: ProgressCb | None = None,
) -> Path:
    """Cut [start_sec, end_sec) from source into output (vertical-ready MP4)."""
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")
    duration = end_sec - start_sec
    if duration < 0.5:
        raise ValueError("Trim window must be at least 0.5 seconds")

    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg(config)
    # Accurate seek: input seek then duration. Re-encode so keyframes don't drift.
    if reencode:
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            config.render.encode_preset or "medium",
            "-crf",
            str(config.render.crf),
            "-c:a",
            "aac",
            "-b:a",
            config.render.audio_bitrate or "192k",
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            "-nostats",
            "-loglevel",
            "error",
            str(output),
        ]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            "-nostats",
            "-loglevel",
            "error",
            str(output),
        ]

    if on_progress:
        on_progress(
            {
                "phase": "trim",
                "message": f"Trimming {duration:.0f}s clip…",
                "pct": 1.0,
                "current": 0,
                "total": max(1, int(round(duration))),
                "eta_sec": None,
            }
        )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_chunks: list[str] = []
    t0 = time.monotonic()
    last_pct = -1.0

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        for chunk in iter(proc.stderr.readline, ""):
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("out_time_ms="):
            continue
        try:
            out_ms = int(line.split("=", 1)[1])
        except ValueError:
            continue
        # FFmpeg reports microseconds in out_time_ms despite the name.
        out_sec = max(0.0, out_ms / 1_000_000.0)
        frac = min(1.0, out_sec / max(0.001, duration))
        pct = round(frac * 100.0, 1)
        if pct < last_pct + 0.5 and pct < 99.0:
            continue
        last_pct = pct
        wall = max(0.05, time.monotonic() - t0)
        # Encode rate from media-seconds produced per wall-second.
        rate = out_sec / wall if out_sec > 0.05 else 0.0
        remain_media = max(0.0, duration - out_sec)
        eta = (remain_media / rate) if rate > 0.05 else None
        if on_progress:
            on_progress(
                {
                    "phase": "trim",
                    "message": (
                        f"Trimming… {out_sec:.0f}s / {duration:.0f}s encoded"
                    ),
                    "pct": min(99.0, pct),
                    "current": int(round(out_sec)),
                    "total": max(1, int(round(duration))),
                    "eta_sec": round(eta, 0) if eta is not None else None,
                    "elapsed_sec": round(wall, 1),
                }
            )

    code = proc.wait()
    stderr_thread.join(timeout=2.0)
    if code != 0 or not output.exists():
        err = "".join(stderr_chunks)[-800:]
        raise RuntimeError(f"ffmpeg trim failed: {err}")

    if on_progress:
        on_progress(
            {
                "phase": "trim",
                "message": "Trim complete",
                "pct": 100.0,
                "current": max(1, int(round(duration))),
                "total": max(1, int(round(duration))),
                "eta_sec": 0,
            }
        )
    return output


def copy_or_link(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        shutil.copy2(source, dest)
    except OSError:
        raise
    return dest
