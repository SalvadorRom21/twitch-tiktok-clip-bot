"""Background job runner for the web UI."""

from __future__ import annotations

import json
import threading
import uuid

import copy

from twitch_tiktok_bot.config import AppConfig, apply_game_profile
from twitch_tiktok_bot.ingest.vod import is_vod_url
from twitch_tiktok_bot.status import ClipJob, ClipStatus, load_job, save_job


_lock = threading.Lock()
_active_threads: dict[str, threading.Thread] = {}


def _run_job(config: AppConfig, job: ClipJob, game_profile: str | None = None) -> None:
    from twitch_tiktok_bot.pipeline import process_media_url

    run_config = copy.copy(config)
    run_config.editing = copy.copy(config.editing)
    apply_game_profile(run_config, game_profile)

    try:
        outputs = process_media_url(
            url=job.clip_url,
            config=run_config,
            media_id=job.id,
            title=job.title,
        )
        work_dir = run_config.resolve_path(run_config.paths.data_dir) / job.id
        analysis_path = work_dir / "analysis.json"
        analysis_data = (
            json.loads(analysis_path.read_text(encoding="utf-8"))
            if analysis_path.exists()
            else {}
        )

        first = outputs[0]
        caption = first.with_suffix(".txt")
        summary_path = work_dir / "job_summary.json"
        hook_text = job.title
        hashtags: list[str] = []
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # Read hook from first short caption file
        if caption.exists():
            lines = caption.read_text(encoding="utf-8").splitlines()
            hook_text = lines[0] if lines else hook_text

        job.status = ClipStatus.READY
        job.output_video = str(first)
        job.caption_file = str(caption)
        job.hook_text = hook_text
        job.hashtags = hashtags
        job.segment_count = len(outputs)
        job.face_crop_center_x = analysis_data.get("face_crop_center_x")
        job.source_type = "vod" if is_vod_url(job.clip_url) else "clip"
        job.output_videos = [str(p) for p in outputs]
        job.error = ""
    except Exception as exc:  # noqa: BLE001 — surface pipeline errors to UI
        job.status = ClipStatus.FAILED
        job.error = str(exc)
    save_job(run_config, job)


def start_job(
    config: AppConfig,
    clip_url: str,
    title: str = "",
    clip_id: str | None = None,
    game_profile: str | None = None,
) -> ClipJob:
    job_id = clip_id or uuid.uuid4().hex[:12]
    job = ClipJob(
        id=job_id,
        clip_url=clip_url,
        title=title,
        status=ClipStatus.PROCESSING,
        source_type="vod" if is_vod_url(clip_url) else "clip",
    )
    save_job(config, job)

    def _target() -> None:
        _run_job(config, job, game_profile=game_profile)
        with _lock:
            _active_threads.pop(job_id, None)

    thread = threading.Thread(target=_target, daemon=True)
    with _lock:
        _active_threads[job_id] = thread
    thread.start()
    return job


def update_caption(config: AppConfig, clip_id: str, caption: str) -> ClipJob:
    from pathlib import Path

    job = load_job(config, clip_id)
    if not job:
        raise FileNotFoundError(f"Clip {clip_id} not found")
    caption_path = Path(job.caption_file) if job.caption_file else None
    if caption_path and caption_path.exists():
        caption_path.write_text(caption, encoding="utf-8")
    else:
        output_dir = config.resolve_path(config.paths.output_dir)
        fallback = output_dir / f"{clip_id}_tiktok.txt"
        fallback.write_text(caption, encoding="utf-8")
        job.caption_file = str(fallback)
    job.hook_text = caption.split("\n")[0].strip()
    save_job(config, job)
    return job


def set_status(config: AppConfig, clip_id: str, status: ClipStatus) -> ClipJob:
    job = load_job(config, clip_id)
    if not job:
        raise FileNotFoundError(f"Clip {clip_id} not found")
    job.status = status
    save_job(config, job)
    return job
