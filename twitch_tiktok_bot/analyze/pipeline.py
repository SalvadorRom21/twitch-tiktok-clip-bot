"""Run full clip analysis."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from twitch_tiktok_bot.analyze.audio import analyze_audio
from twitch_tiktok_bot.analyze.face import detect_face_crop_center
from twitch_tiktok_bot.analyze.speech import transcribe
from twitch_tiktok_bot.analyze.vision import (
    build_vision_summary,
    describe_frames_llm,
    describe_frames_simple,
    extract_frames,
)
from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, SceneChange


def _detect_scene_changes(video_path: Path, ffmpeg: str = "ffmpeg") -> list[SceneChange]:
    cmd = [
        ffmpeg,
        "-i",
        str(video_path),
        "-filter:v",
        "select='gt(scene,0.35)',showinfo",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = result.stderr
    changes: list[SceneChange] = []
    for line in output.splitlines():
        if "pts_time:" not in line:
            continue
        for part in line.split():
            if part.startswith("pts_time:"):
                try:
                    changes.append(SceneChange(time=float(part.split(":")[1])))
                except ValueError:
                    pass
                break
    return changes


def analyze_clip(
    video_path: Path,
    config: AppConfig,
    work_dir: Path,
    clip_title: str = "",
    game_name: str = "",
) -> ClipAnalysis:
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = config.render.ffmpeg_path or "ffmpeg"
    analysis_cfg = config.analysis

    print("  [analyze] audio peaks + silence...")
    peaks, silence, duration = analyze_audio(
        video_path,
        work_dir,
        peak_percentile=config.editing.peak_percentile,
        silence_threshold_sec=config.editing.silence_threshold_sec,
        ffmpeg=ffmpeg,
    )

    print("  [analyze] speech transcription (whisper)...")
    transcript = transcribe(
        video_path,
        model_size=analysis_cfg.whisper_model,
        device=analysis_cfg.whisper_device,
        compute_type=analysis_cfg.whisper_compute_type,
    )

    print("  [analyze] scene changes...")
    scenes = _detect_scene_changes(video_path, ffmpeg=ffmpeg)

    vision_frames = []
    vision_summary = ""
    if analysis_cfg.vision_frame_interval_sec > 0:
        print("  [analyze] sampling frames...")
        frame_dir = work_dir / "frames"
        frame_paths = extract_frames(
            video_path,
            frame_dir,
            interval_sec=analysis_cfg.vision_frame_interval_sec,
            ffmpeg=ffmpeg,
        )
        llm_cfg = config.llm
        api_key = os.getenv(llm_cfg.api_key_env, "")
        if llm_cfg.enabled and api_key:
            print("  [analyze] vision descriptions (LLM)...")
            vision_frames = describe_frames_llm(
                frame_paths,
                analysis_cfg.vision_frame_interval_sec,
                api_key=api_key,
                model=llm_cfg.model,
                base_url=llm_cfg.base_url,
            )
        else:
            vision_frames = describe_frames_simple(
                frame_paths, analysis_cfg.vision_frame_interval_sec
            )
        vision_summary = build_vision_summary(vision_frames)

    face_center: float | None = None
    if config.render.face_crop_enabled:
        print("  [analyze] face detection for crop...")
        face_center = detect_face_crop_center(
            video_path,
            work_dir,
            sample_count=config.render.face_sample_count,
            ffmpeg=ffmpeg,
        )
        if face_center is not None:
            print(f"       face crop center x={face_center:.2f}")

    result = ClipAnalysis(
        duration=duration,
        transcript_segments=transcript,
        loud_peaks=peaks,
        silence_ranges=silence,
        scene_changes=scenes,
        vision_frames=vision_frames,
        vision_summary=vision_summary,
        clip_title=clip_title,
        game_name=game_name,
        face_crop_center_x=face_center,
    )

    analysis_path = work_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    return result
