"""Audio analysis: loudness peaks and silence detection."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from twitch_tiktok_bot.models import LoudPeak, TimeRange


def _extract_wav(video_path: Path, wav_path: Path, ffmpeg: str = "ffmpeg") -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed:\n{result.stderr}")


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def detect_loud_peaks(
    audio: np.ndarray,
    sample_rate: int,
    percentile: float = 85.0,
    window_sec: float = 0.25,
    min_gap_sec: float = 1.0,
) -> list[LoudPeak]:
    window = max(1, int(sample_rate * window_sec))
    if len(audio) < window:
        return []

    rms = np.array(
        [
            np.sqrt(np.mean(audio[i : i + window] ** 2))
            for i in range(0, len(audio) - window, window)
        ]
    )
    if rms.size == 0:
        return []

    threshold = float(np.percentile(rms, percentile))
    peaks: list[LoudPeak] = []
    last_t = -999.0
    for idx, value in enumerate(rms):
        if value < threshold:
            continue
        t = (idx * window) / sample_rate
        if t - last_t < min_gap_sec:
            continue
        score = float(value / (threshold + 1e-9))
        peaks.append(LoudPeak(time=t, score=score))
        last_t = t
    return peaks


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.02,
    min_duration_sec: float = 0.4,
) -> list[TimeRange]:
    window = max(1, int(sample_rate * 0.05))
    silent_flags: list[bool] = []
    for i in range(0, len(audio), window):
        chunk = audio[i : i + window]
        rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
        silent_flags.append(rms < threshold)

    ranges: list[TimeRange] = []
    start_idx: int | None = None
    for idx, is_silent in enumerate(silent_flags):
        if is_silent and start_idx is None:
            start_idx = idx
        elif not is_silent and start_idx is not None:
            start_t = (start_idx * window) / sample_rate
            end_t = (idx * window) / sample_rate
            if end_t - start_t >= min_duration_sec:
                ranges.append(TimeRange(start=start_t, end=end_t))
            start_idx = None

    if start_idx is not None:
        start_t = (start_idx * window) / sample_rate
        end_t = len(audio) / sample_rate
        if end_t - start_t >= min_duration_sec:
            ranges.append(TimeRange(start=start_t, end=end_t))

    return ranges


def analyze_audio(
    video_path: Path,
    work_dir: Path,
    peak_percentile: float = 85.0,
    silence_threshold_sec: float = 0.4,
    ffmpeg: str = "ffmpeg",
) -> tuple[list[LoudPeak], list[TimeRange], float]:
    wav_path = work_dir / "audio.wav"
    _extract_wav(video_path, wav_path, ffmpeg=ffmpeg)
    audio, sample_rate = _read_wav_mono(wav_path)
    duration = len(audio) / sample_rate
    peaks = detect_loud_peaks(audio, sample_rate, percentile=peak_percentile)
    silence = detect_silence(
        audio, sample_rate, min_duration_sec=silence_threshold_sec
    )
    return peaks, silence, duration
