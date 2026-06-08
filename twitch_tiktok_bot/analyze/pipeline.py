"""Run full clip analysis."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from twitch_tiktok_bot.analyze.audio import analyze_audio
from twitch_tiktok_bot.analyze.face_cam import resolve_face_cam_layout
from twitch_tiktok_bot.analyze.speech import transcribe
from twitch_tiktok_bot.analyze.vision import (
    build_vision_summary,
    describe_frames_llm,
    describe_frames_simple,
    extract_frames,
)
from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, SceneChange
from twitch_tiktok_bot.progress import step


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

    with step("  [analyze] audio peaks + silence", heartbeat_sec=0):
        peaks, silence, duration = analyze_audio(
            video_path,
            work_dir,
            peak_percentile=config.editing.peak_percentile,
            silence_threshold_sec=config.editing.silence_threshold_sec,
            ffmpeg=ffmpeg,
        )

    with step(
        f"  [analyze] speech transcription (whisper {analysis_cfg.whisper_model}/{analysis_cfg.whisper_device})",
        heartbeat_sec=20,
    ) as update:
        transcript = transcribe(
            video_path,
            model_size=analysis_cfg.whisper_model,
            device=analysis_cfg.whisper_device,
            compute_type=analysis_cfg.whisper_compute_type,
            on_progress=update,
        )

    with step("  [analyze] scene changes", heartbeat_sec=0):
        scenes = _detect_scene_changes(video_path, ffmpeg=ffmpeg)

    vision_frames = []
    vision_summary = ""
    if analysis_cfg.vision_frame_interval_sec > 0:
        with step("  [analyze] sampling frames", heartbeat_sec=0):
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
            with step("  [analyze] vision descriptions (LLM)", heartbeat_sec=15):
                vision_frames = describe_frames_llm(
                    frame_paths,
                    analysis_cfg.vision_frame_interval_sec,
                    api_key=api_key,
                    model=llm_cfg.model,
                    base_url=llm_cfg.base_url,
                )
        else:
            with step("  [analyze] vision descriptions", heartbeat_sec=0):
                vision_frames = describe_frames_simple(
                    frame_paths, analysis_cfg.vision_frame_interval_sec
                )
        vision_summary = build_vision_summary(vision_frames)

    face_center: float | None = None
    face_cam_dict: dict[str, float] | None = None
    with step("  [analyze] face-cam detection", heartbeat_sec=10):
        layout = resolve_face_cam_layout(video_path, work_dir, config, ffmpeg=ffmpeg)
        face_center = layout.face_crop_center_x
        face_cam_dict = layout.face_cam_region
        if layout.method == "override":
            print("       using face_cam_override from config")
        elif layout.face_cam_region and layout.method not in ("disabled", "default", ""):
            region = layout.face_cam_region
            print(
                f"       face cam overlay at x={region['x']:.2f} y={region['y']:.2f} "
                f"({region['w']:.0%}×{region['h']:.0%}) via {layout.method}"
            )
        elif layout.method == "default":
            print(
                f"       using default {config.render.face_cam_corner} face cam "
                "(detection failed)"
            )
        elif face_center is not None:
            print(f"       face center x={face_center:.2f} (no overlay box)")

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
        face_cam_region=face_cam_dict,
    )

    analysis_path = work_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    return result
