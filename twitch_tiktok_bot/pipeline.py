"""End-to-end clip processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from twitch_tiktok_bot.analyze.pipeline import analyze_clip
from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.ingest.download import download_clip
from twitch_tiktok_bot.models import TwitchClip
from twitch_tiktok_bot.plan.editor_llm import create_edit_plan
from twitch_tiktok_bot.render.ffmpeg_build import render_short


def process_clip_url(
    url: str,
    config: AppConfig,
    clip_id: str | None = None,
    clip_title: str = "",
    game_name: str = "",
) -> Path:
    data_dir = config.resolve_path(config.paths.data_dir)
    output_dir = config.resolve_path(config.paths.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = clip_id or "clip"
    work_dir = data_dir / safe_id
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading clip...")
    video_path = download_clip(url, work_dir, clip_id=safe_id)
    print(f"       -> {video_path}")

    print(f"[2/4] Analyzing video...")
    analysis = analyze_clip(
        video_path,
        config,
        work_dir,
        clip_title=clip_title,
        game_name=game_name,
    )
    print(f"       duration={analysis.duration:.1f}s, "
          f"transcript={len(analysis.transcript_segments)} segments, "
          f"peaks={len(analysis.loud_peaks)}")

    print(f"[3/4] Creating edit plan...")
    plan = create_edit_plan(analysis, config)
    print(f"       hook=\"{plan.hook_text}\", segments={len(plan.segments)}")

    print(f"[4/4] Rendering TikTok short...")
    out_path = output_dir / f"{safe_id}_tiktok.mp4"
    render_short(video_path, analysis, plan, out_path, config, work_dir)
    print(f"       -> {out_path}")
    print(f"       caption -> {out_path.with_suffix('.txt')}")

    summary = {
        "clip_id": safe_id,
        "source_url": url,
        "output_video": str(out_path),
        "caption_file": str(out_path.with_suffix(".txt")),
        "analysis": str(work_dir / "analysis.json"),
        "edit_plan": str(work_dir / "edit_plan.json"),
    }
    (work_dir / "job_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return out_path


def process_twitch_clip(clip: TwitchClip, config: AppConfig) -> Path:
    return process_clip_url(
        url=clip.url,
        config=config,
        clip_id=clip.id,
        clip_title=clip.title,
        game_name=clip.game_name,
    )
