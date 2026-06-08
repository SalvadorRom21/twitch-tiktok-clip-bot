"""Chunked analysis for long Twitch VODs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from twitch_tiktok_bot.analyze.audio import (
    detect_loud_peaks,
    detect_silence,
    _read_wav_mono,
)
from twitch_tiktok_bot.analyze.duration import get_video_duration
from twitch_tiktok_bot.analyze.face_cam import resolve_face_cam_layout
from twitch_tiktok_bot.analyze.speech import transcribe
from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, LoudPeak, TimeRange, TranscriptSegment


def _extract_wav_segment(
    video_path: Path,
    wav_path: Path,
    start_sec: float,
    duration_sec: float,
    ffmpeg: str = "ffmpeg",
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_sec}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_sec}",
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
        raise RuntimeError(f"ffmpeg segment extract failed:\n{result.stderr}")


def analyze_audio_chunked(
    video_path: Path,
    work_dir: Path,
    chunk_sec: float,
    peak_percentile: float,
    silence_threshold_sec: float,
    ffmpeg: str = "ffmpeg",
) -> tuple[list[LoudPeak], list[TimeRange], float]:
    duration = get_video_duration(video_path, ffmpeg=ffmpeg)
    all_peaks: list[LoudPeak] = []
    all_silence: list[TimeRange] = []
    chunk_dir = work_dir / "audio_chunks"
    chunk_dir.mkdir(exist_ok=True)

    offset = 0.0
    chunk_idx = 0
    while offset < duration:
        seg_len = min(chunk_sec, duration - offset)
        wav_path = chunk_dir / f"chunk_{chunk_idx:04d}.wav"
        print(f"  [analyze] audio scan {offset/60:.0f}-{ (offset+seg_len)/60:.0f} min...")
        _extract_wav_segment(video_path, wav_path, offset, seg_len, ffmpeg=ffmpeg)
        audio, sample_rate = _read_wav_mono(wav_path)
        peaks = detect_loud_peaks(audio, sample_rate, percentile=peak_percentile)
        silence = detect_silence(
            audio, sample_rate, min_duration_sec=silence_threshold_sec
        )
        for peak in peaks:
            all_peaks.append(LoudPeak(time=peak.time + offset, score=peak.score))
        for span in silence:
            all_silence.append(
                TimeRange(start=span.start + offset, end=span.end + offset)
            )
        offset += seg_len
        chunk_idx += 1

    all_peaks.sort(key=lambda p: p.score, reverse=True)
    return all_peaks, all_silence, duration


def transcribe_chunked(
    video_path: Path,
    work_dir: Path,
    chunk_sec: float,
    model_size: str,
    device: str,
    compute_type: str,
    ffmpeg: str = "ffmpeg",
) -> list[TranscriptSegment]:
    duration = get_video_duration(video_path, ffmpeg=ffmpeg)
    segments: list[TranscriptSegment] = []
    chunk_dir = work_dir / "transcript_chunks"
    chunk_dir.mkdir(exist_ok=True)

    offset = 0.0
    chunk_idx = 0
    while offset < duration:
        seg_len = min(chunk_sec, duration - offset)
        chunk_video = chunk_dir / f"chunk_{chunk_idx:04d}.mp4"
        print(f"  [analyze] whisper {offset/60:.0f}-{ (offset+seg_len)/60:.0f} min...")
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{offset}",
            "-i",
            str(video_path),
            "-t",
            f"{seg_len}",
            "-c",
            "copy",
            str(chunk_video),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        if chunk_video.exists() and chunk_video.stat().st_size > 0:
            chunk_segments = transcribe(
                chunk_video,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
            )
            for seg in chunk_segments:
                segments.append(
                    TranscriptSegment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text,
                    )
                )
        offset += seg_len
        chunk_idx += 1

    return segments


def analyze_vod(
    video_path: Path,
    config: AppConfig,
    work_dir: Path,
    vod_title: str = "",
    game_name: str = "",
) -> ClipAnalysis:
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = config.render.ffmpeg_path or "ffmpeg"
    vod_cfg = config.vod
    analysis_cfg = config.analysis

    print("  [analyze] VOD mode — scanning full stream in chunks...")
    peaks, silence, duration = analyze_audio_chunked(
        video_path,
        work_dir,
        chunk_sec=vod_cfg.audio_scan_chunk_sec,
        peak_percentile=config.editing.peak_percentile,
        silence_threshold_sec=config.editing.silence_threshold_sec,
        ffmpeg=ffmpeg,
    )
    print(f"       found {len(peaks)} reaction peaks across {duration/60:.1f} min")

    transcript = transcribe_chunked(
        video_path,
        work_dir,
        chunk_sec=vod_cfg.whisper_chunk_sec,
        model_size=analysis_cfg.whisper_model,
        device=analysis_cfg.whisper_device,
        compute_type=analysis_cfg.whisper_compute_type,
        ffmpeg=ffmpeg,
    )
    print(f"       transcript segments: {len(transcript)}")

    print("  [analyze] face-cam detection (spread across full VOD)...")
    face_layout = resolve_face_cam_layout(
        video_path, work_dir, config, ffmpeg=ffmpeg
    )
    if face_layout.method == "override":
        print("       using face_cam_override from config")
    elif face_layout.face_cam_region and face_layout.method not in (
        "disabled",
        "default",
        "",
    ):
        region = face_layout.face_cam_region
        print(
            f"       face cam overlay at x={region['x']:.2f} y={region['y']:.2f} "
            f"({region['w']:.0%}×{region['h']:.0%}) via {face_layout.method}"
        )
    elif face_layout.method == "default":
        print(
            f"       using default {config.render.face_cam_corner} face cam "
            "(detection failed)"
        )

    result = ClipAnalysis(
        duration=duration,
        transcript_segments=transcript,
        loud_peaks=peaks,
        silence_ranges=silence,
        scene_changes=[],
        vision_frames=[],
        vision_summary="",
        clip_title=vod_title,
        game_name=game_name,
        face_crop_center_x=face_layout.face_crop_center_x,
        face_cam_region=face_layout.face_cam_region,
    )
    (work_dir / "analysis.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    return result
