"""Background job runner for the web UI."""

from __future__ import annotations

import json
import threading
import uuid

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.status import ClipJob, ClipStatus, load_job, save_job


_lock = threading.Lock()
_active_threads: dict[str, threading.Thread] = {}


def _run_job(config: AppConfig, job: ClipJob) -> None:
    from twitch_tiktok_bot.pipeline import process_clip_url

    try:
        output = process_clip_url(
            url=job.clip_url,
            config=config,
            clip_id=job.id,
            clip_title=job.title,
        )
        plan_path = config.resolve_path(config.paths.data_dir) / job.id / "edit_plan.json"
        analysis_path = config.resolve_path(config.paths.data_dir) / job.id / "analysis.json"
        plan_data = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
        analysis_data = (
            json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
        )

        job.status = ClipStatus.READY
        job.output_video = str(output)
        job.caption_file = str(output.with_suffix(".txt"))
        job.hook_text = plan_data.get("hook_text", "")
        job.hashtags = list(plan_data.get("hashtags", []))
        job.segment_count = len(plan_data.get("segments", []))
        job.face_crop_center_x = analysis_data.get("face_crop_center_x")
        job.error = ""
    except Exception as exc:  # noqa: BLE001 — surface pipeline errors to UI
        job.status = ClipStatus.FAILED
        job.error = str(exc)
    save_job(config, job)


def start_job(
    config: AppConfig,
    clip_url: str,
    title: str = "",
    clip_id: str | None = None,
) -> ClipJob:
    job_id = clip_id or uuid.uuid4().hex[:12]
    job = ClipJob(
        id=job_id,
        clip_url=clip_url,
        title=title,
        status=ClipStatus.PROCESSING,
    )
    save_job(config, job)

    def _target() -> None:
        _run_job(config, job)
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
