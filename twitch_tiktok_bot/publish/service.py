"""High-level publish operations for ClipJobs and trainer VOD windows."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.labels.fights import resolve_vod_video
from twitch_tiktok_bot.publish.captions import youtube_description, youtube_title
from twitch_tiktok_bot.publish.export_pack import pack_root, write_export_pack
from twitch_tiktok_bot.publish.trim import probe_duration, trim_media
from twitch_tiktok_bot.publish.youtube import (
    YouTubeNotConfiguredError,
    load_pack_youtube_texts,
    save_upload_record,
    upload_short,
    youtube_status,
)
from twitch_tiktok_bot.publish.tiktok import tiktok_status
from twitch_tiktok_bot.publish.instagram import instagram_status
from twitch_tiktok_bot.status import ClipJob, ClipStatus, load_job, save_job

_publish_lock = threading.Lock()
_publish_threads: dict[str, threading.Thread] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_caption(job: ClipJob) -> str:
    if job.caption_file and Path(job.caption_file).exists():
        return Path(job.caption_file).read_text(encoding="utf-8")
    if job.hook_text:
        tags = " ".join(f"#{t}" for t in job.hashtags)
        return f"{job.hook_text}\n\n{tags}\n".strip() + "\n"
    return (job.title or job.id) + "\n"


def _write_caption(job: ClipJob, config: AppConfig, text: str) -> None:
    path = Path(job.caption_file) if job.caption_file else None
    if not path:
        path = config.resolve_path(config.paths.output_dir) / f"{job.id}_tiktok.txt"
        job.caption_file = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    job.hook_text = text.split("\n")[0].strip()


def get_publish_status(config: AppConfig, clip_id: str) -> dict:
    job = load_job(config, clip_id)
    if not job:
        raise FileNotFoundError(f"Clip {clip_id} not found")

    video = Path(job.output_video) if job.output_video else None
    duration = None
    if video and video.exists():
        try:
            duration = probe_duration(config, video)
        except Exception:  # noqa: BLE001
            duration = None

    root = pack_root(config, clip_id)
    manifest = None
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    upload_rec = None
    upload_path = config.resolve_path(config.paths.data_dir) / clip_id / "youtube_upload.json"
    if upload_path.exists():
        upload_rec = json.loads(upload_path.read_text(encoding="utf-8"))

    return {
        "clip": job.to_dict(),
        "duration_sec": duration,
        "edit_start_sec": job.edit_start_sec,
        "edit_end_sec": job.edit_end_sec,
        "pack": manifest,
        "pack_dir": str(root) if root.exists() else "",
        "youtube": youtube_status(config),
        "youtube_upload": upload_rec,
        "tiktok": tiktok_status(config),
        "instagram": instagram_status(config),
    }


def trim_clip_job(
    config: AppConfig,
    clip_id: str,
    *,
    start_sec: float,
    end_sec: float,
    caption: str | None = None,
) -> ClipJob:
    """Re-trim the current output video and mark job ready."""
    job = load_job(config, clip_id)
    if not job:
        raise FileNotFoundError(f"Clip {clip_id} not found")
    if not job.output_video or not Path(job.output_video).exists():
        raise FileNotFoundError("Clip has no output video to trim")

    source = Path(job.output_video)
    # Keep an original once so repeated trims don't stack-lossy forever
    work = config.resolve_path(config.paths.data_dir) / clip_id
    work.mkdir(parents=True, exist_ok=True)
    master = work / "publish_master.mp4"
    if not master.exists():
        master.write_bytes(source.read_bytes())

    out_dir = config.resolve_path(config.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}_tiktok.mp4"
    # Trim into a temp then replace
    tmp = work / "publish_trim_tmp.mp4"
    trim_media(
        config,
        master,
        tmp,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        reencode=True,
    )
    tmp.replace(out_path)

    job.output_video = str(out_path)
    job.output_videos = [str(out_path)]
    job.edit_start_sec = float(start_sec)
    job.edit_end_sec = float(end_sec)
    job.segment_count = 1
    if caption is not None:
        _write_caption(job, config, caption)
    elif not job.caption_file:
        _write_caption(job, config, _read_caption(job))
    if job.status in (ClipStatus.REJECTED, ClipStatus.FAILED):
        job.status = ClipStatus.READY
    job.error = ""
    save_job(config, job)
    return job


def export_clip_pack(config: AppConfig, clip_id: str, *, caption: str | None = None) -> dict:
    job = load_job(config, clip_id)
    if not job:
        raise FileNotFoundError(f"Clip {clip_id} not found")
    if not job.output_video or not Path(job.output_video).exists():
        raise FileNotFoundError("Clip has no output video")
    if caption is not None:
        _write_caption(job, config, caption)
        save_job(config, job)
    text = _read_caption(job)
    pack = write_export_pack(
        config,
        clip_id=clip_id,
        video_path=Path(job.output_video),
        caption=text,
        title=job.title or job.hook_text or clip_id,
    )
    job.publish_dir = pack["pack_dir"]
    save_job(config, job)
    return pack


def upload_clip_to_youtube(config: AppConfig, clip_id: str, *, caption: str | None = None) -> dict:
    job = load_job(config, clip_id)
    if not job:
        raise FileNotFoundError(f"Clip {clip_id} not found")
    if caption is not None:
        _write_caption(job, config, caption)
        save_job(config, job)

    # Ensure pack exists for title/description consistency
    root = pack_root(config, clip_id)
    if not root.exists():
        export_clip_pack(config, clip_id)

    text = _read_caption(job)
    title, description = load_pack_youtube_texts(root)
    if not title:
        title = youtube_title(text, title=job.title)
    if not description:
        description = youtube_description(text, title=job.title)

    video = Path(job.output_video)
    try:
        result = upload_short(
            config,
            video_path=video,
            title=title,
            description=description,
            tags=job.hashtags or None,
        )
    except YouTubeNotConfiguredError:
        raise

    record = {
        "uploaded_at": _now(),
        **result,
    }
    rec_path = config.resolve_path(config.paths.data_dir) / clip_id / "youtube_upload.json"
    save_upload_record(rec_path, record)
    job.youtube_video_id = str(result.get("video_id") or "")
    job.youtube_url = str(result.get("url") or "")
    if job.status == ClipStatus.READY:
        job.status = ClipStatus.APPROVED
    save_job(config, job)
    return record


def create_job_from_vod_window(
    config: AppConfig,
    *,
    vod_id: str,
    start_sec: float,
    end_sec: float,
    title: str = "",
    caption: str = "",
    clip_id: str | None = None,
    on_progress=None,
) -> ClipJob:
    """Extract a VOD window into a ready ClipJob for the publish desk."""
    _, video = resolve_vod_video(config, vod_id)
    job_id = clip_id or f"pub_{uuid.uuid4().hex[:10]}"
    out_dir = config.resolve_path(config.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job_id}_tiktok.mp4"

    work = config.resolve_path(config.paths.data_dir) / job_id
    work.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(
            {
                "phase": "prepare",
                "message": "Preparing trim…",
                "pct": 0.5,
                "eta_sec": None,
            }
        )
    trim_media(
        config,
        video,
        out_path,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        reencode=True,
        on_progress=on_progress,
    )
    # Master for later re-trims relative to this extract
    if on_progress:
        on_progress(
            {
                "phase": "finalize",
                "message": "Saving publish short…",
                "pct": 99.0,
                "eta_sec": 1,
            }
        )
    master = work / "publish_master.mp4"
    master.write_bytes(out_path.read_bytes())

    hook = (
        title
        or (caption.split("\n")[0].strip() if caption.strip() else "")
        or f"{vod_id} boss fight"
    ).strip()
    text = caption.strip() + "\n" if caption.strip() else f"{hook}\n\n#eldenring #bossfight #fyp\n"
    cap_path = out_path.with_suffix(".txt")
    cap_path.write_text(text, encoding="utf-8")

    job = ClipJob(
        id=job_id,
        clip_url=f"local://{vod_id}#{start_sec:.1f}-{end_sec:.1f}",
        title=title or hook,
        status=ClipStatus.READY,
        output_video=str(out_path),
        caption_file=str(cap_path),
        hook_text=hook,
        hashtags=["eldenring", "bossfight", "fyp"],
        segment_count=1,
        source_type="vod",
        output_videos=[str(out_path)],
        edit_start_sec=0.0,
        edit_end_sec=float(end_sec) - float(start_sec),
        source_vod_id=vod_id,
        source_start_sec=float(start_sec),
        source_end_sec=float(end_sec),
    )
    save_job(config, job)
    return job


def start_publish_from_train(
    config: AppConfig,
    *,
    vod_id: str,
    start_sec: float,
    end_sec: float,
    title: str = "",
    caption: str = "",
) -> str:
    """Start async VOD-window publish; returns job_id for progress polling."""
    from twitch_tiktok_bot.web.job_progress import finish_job, start_job, update_job

    job_id = f"publish_{uuid.uuid4().hex[:10]}"
    duration = max(0.5, float(end_sec) - float(start_sec))
    start_job(
        job_id,
        kind="publish",
        message=f"Queued trim ({duration:.0f}s clip)…",
    )
    update_job(
        job_id,
        phase="prepare",
        message=f"Starting trim of {duration:.0f}s…",
        pct=0.5,
        current=0,
        total=max(1, int(round(duration))),
        clip_duration_sec=duration,
        source_vod_id=vod_id,
    )

    def _run() -> None:
        try:

            def on_progress(fields: dict) -> None:
                update_job(job_id, **fields)

            clip = create_job_from_vod_window(
                config,
                vod_id=vod_id,
                start_sec=start_sec,
                end_sec=end_sec,
                title=title,
                caption=caption,
                on_progress=on_progress,
            )
            update_job(
                job_id,
                result_clip_id=clip.id,
                result=clip.to_dict(),
            )
            finish_job(
                job_id,
                status="done",
                message=f"Ready — {clip.id}",
            )
        except Exception as exc:  # noqa: BLE001
            finish_job(job_id, status="failed", message="Publish failed", error=str(exc))
        finally:
            with _publish_lock:
                _publish_threads.pop(job_id, None)

    thread = threading.Thread(target=_run, daemon=True)
    with _publish_lock:
        _publish_threads[job_id] = thread
    thread.start()
    return job_id
