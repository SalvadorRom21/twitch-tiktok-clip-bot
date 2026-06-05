"""End-to-end clip and VOD processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from twitch_tiktok_bot.analyze.pipeline import analyze_clip
from twitch_tiktok_bot.analyze.vod import analyze_vod
from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.ingest.download import download_video
from twitch_tiktok_bot.ingest.vod import is_vod_url, parse_vod_id
from twitch_tiktok_bot.models import TwitchClip, TwitchVod
from twitch_tiktok_bot.plan.editor_llm import create_edit_plan
from twitch_tiktok_bot.plan.vod_planner import create_vod_short_plans
from twitch_tiktok_bot.render.ffmpeg_build import render_short
from twitch_tiktok_bot.status import ClipJob, ClipStatus, save_job


def _save_job_summary(
    work_dir: Path,
    safe_id: str,
    source_url: str,
    outputs: list[Path],
    analysis_path: Path,
) -> None:
    summary = {
        "clip_id": safe_id,
        "source_url": source_url,
        "outputs": [str(p) for p in outputs],
        "output_video": str(outputs[0]) if outputs else "",
        "caption_file": str(outputs[0].with_suffix(".txt")) if outputs else "",
        "analysis": str(analysis_path),
    }
    (work_dir / "job_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def process_vod_url(
    url: str,
    config: AppConfig,
    vod_id: str | None = None,
    vod_title: str = "",
    game_name: str = "",
    max_shorts: int | None = None,
) -> list[Path]:
    data_dir = config.resolve_path(config.paths.data_dir)
    output_dir = config.resolve_path(config.paths.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = vod_id or parse_vod_id(url) or "vod"
    work_dir = data_dir / safe_id
    work_dir.mkdir(parents=True, exist_ok=True)

    end_sec = config.vod.max_download_sec if config.vod.max_download_sec > 0 else None

    print(f"[1/4] Downloading VOD...")
    video_path = download_video(
        url,
        work_dir,
        video_id=safe_id,
        end_sec=end_sec,
    )
    print(f"       -> {video_path}")

    print(f"[2/4] Analyzing full VOD (chunked scan)...")
    analysis = analyze_vod(
        video_path,
        config,
        work_dir,
        vod_title=vod_title,
        game_name=game_name,
    )
    print(
        f"       duration={analysis.duration/60:.1f} min, "
        f"transcript={len(analysis.transcript_segments)} segments, "
        f"peaks={len(analysis.loud_peaks)}"
    )

    print(f"[3/4] Finding highlights across stream...")
    plans = create_vod_short_plans(analysis, config, max_shorts=max_shorts)
    print(f"       {len(plans)} TikTok short(s) planned")

    print(f"[4/4] Rendering TikTok shorts...")
    outputs: list[Path] = []
    for idx, plan in enumerate(plans, start=1):
        out_path = output_dir / f"{safe_id}_short_{idx:02d}.mp4"
        short_work = work_dir / f"short_{idx:02d}"
        render_short(video_path, analysis, plan, out_path, config, short_work)
        outputs.append(out_path)
        print(f"       -> {out_path}")

    _save_job_summary(work_dir, safe_id, url, outputs, work_dir / "analysis.json")

    first_plan = plans[0].to_dict() if plans else {}
    save_job(
        config,
        ClipJob(
            id=safe_id,
            clip_url=url,
            title=vod_title,
            status=ClipStatus.READY,
            output_video=str(outputs[0]) if outputs else "",
            caption_file=str(outputs[0].with_suffix(".txt")) if outputs else "",
            hook_text=first_plan.get("hook_text", ""),
            hashtags=list(first_plan.get("hashtags", [])),
            segment_count=len(plans),
            face_crop_center_x=analysis.face_crop_center_x,
            source_type="vod",
            output_videos=[str(p) for p in outputs],
        ),
    )
    return outputs


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
    video_path = download_video(url, work_dir, video_id=safe_id)
    print(f"       -> {video_path}")

    print(f"[2/4] Analyzing video...")
    analysis = analyze_clip(
        video_path,
        config,
        work_dir,
        clip_title=clip_title,
        game_name=game_name,
    )
    print(
        f"       duration={analysis.duration:.1f}s, "
        f"transcript={len(analysis.transcript_segments)} segments, "
        f"peaks={len(analysis.loud_peaks)}"
    )

    print(f"[3/4] Creating edit plan...")
    plan = create_edit_plan(analysis, config)
    print(f"       hook=\"{plan.hook_text}\", segments={len(plan.segments)}")

    print(f"[4/4] Rendering TikTok short...")
    out_path = output_dir / f"{safe_id}_tiktok.mp4"
    render_short(video_path, analysis, plan, out_path, config, work_dir)
    print(f"       -> {out_path}")
    print(f"       caption -> {out_path.with_suffix('.txt')}")

    plan_data = plan.to_dict()
    _save_job_summary(work_dir, safe_id, url, [out_path], work_dir / "analysis.json")
    (work_dir / "edit_plan.json").write_text(
        json.dumps(plan_data, indent=2), encoding="utf-8"
    )

    save_job(
        config,
        ClipJob(
            id=safe_id,
            clip_url=url,
            title=clip_title,
            status=ClipStatus.READY,
            output_video=str(out_path),
            caption_file=str(out_path.with_suffix(".txt")),
            hook_text=plan_data.get("hook_text", ""),
            hashtags=list(plan_data.get("hashtags", [])),
            segment_count=len(plan_data.get("segments", [])),
            face_crop_center_x=analysis.face_crop_center_x,
            source_type="clip",
            output_videos=[str(out_path)],
        ),
    )
    return out_path


def process_media_url(
    url: str,
    config: AppConfig,
    media_id: str | None = None,
    title: str = "",
    game_name: str = "",
    max_shorts: int | None = None,
) -> list[Path]:
    """Route to clip or VOD pipeline based on URL."""
    if is_vod_url(url):
        return process_vod_url(
            url=url,
            config=config,
            vod_id=media_id,
            vod_title=title,
            game_name=game_name,
            max_shorts=max_shorts,
        )
    return [process_clip_url(url, config, clip_id=media_id, clip_title=title, game_name=game_name)]


def process_twitch_clip(clip: TwitchClip, config: AppConfig) -> Path:
    return process_clip_url(
        url=clip.url,
        config=config,
        clip_id=clip.id,
        clip_title=clip.title,
        game_name=clip.game_name,
    )


def process_twitch_vod(vod: TwitchVod, config: AppConfig, max_shorts: int | None = None) -> list[Path]:
    return process_vod_url(
        url=vod.url,
        config=config,
        vod_id=vod.id,
        vod_title=vod.title,
        max_shorts=max_shorts,
    )
