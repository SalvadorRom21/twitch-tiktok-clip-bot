"""Clip job status persisted on disk."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from twitch_tiktok_bot.config import AppConfig


class ClipStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ClipJob:
    id: str
    clip_url: str
    title: str = ""
    status: ClipStatus = ClipStatus.PROCESSING
    created_at: str = ""
    updated_at: str = ""
    error: str = ""
    output_video: str = ""
    caption_file: str = ""
    hook_text: str = ""
    hashtags: list[str] = field(default_factory=list)
    segment_count: int = 0
    face_crop_center_x: float | None = None
    source_type: str = "clip"  # clip | vod
    output_videos: list[str] = field(default_factory=list)
    # Publish desk: trim window relative to publish_master / current short
    edit_start_sec: float | None = None
    edit_end_sec: float | None = None
    publish_dir: str = ""
    youtube_video_id: str = ""
    youtube_url: str = ""
    # When created from an Elden trainer VOD window
    source_vod_id: str = ""
    source_start_sec: float | None = None
    source_end_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(config: AppConfig, clip_id: str) -> Path:
    return config.resolve_path(config.paths.data_dir) / clip_id / "status.json"


def save_job(config: AppConfig, job: ClipJob) -> None:
    path = _status_path(config, job.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not job.created_at:
        existing = load_job(config, job.id)
        if existing and existing.created_at:
            job.created_at = existing.created_at
    if not job.created_at:
        job.created_at = _now_iso()
    job.updated_at = _now_iso()
    path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")


def load_job(config: AppConfig, clip_id: str) -> ClipJob | None:
    path = _status_path(config, clip_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ClipJob(
        id=data["id"],
        clip_url=data.get("clip_url", ""),
        title=data.get("title", ""),
        status=ClipStatus(data.get("status", ClipStatus.READY.value)),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        error=data.get("error", ""),
        output_video=data.get("output_video", ""),
        caption_file=data.get("caption_file", ""),
        hook_text=data.get("hook_text", ""),
        hashtags=list(data.get("hashtags", [])),
        segment_count=int(data.get("segment_count", 0)),
        face_crop_center_x=data.get("face_crop_center_x"),
        source_type=data.get("source_type", "clip"),
        output_videos=list(data.get("output_videos", [])),
        edit_start_sec=data.get("edit_start_sec"),
        edit_end_sec=data.get("edit_end_sec"),
        publish_dir=str(data.get("publish_dir", "") or ""),
        youtube_video_id=str(data.get("youtube_video_id", "") or ""),
        youtube_url=str(data.get("youtube_url", "") or ""),
        source_vod_id=str(data.get("source_vod_id", "") or ""),
        source_start_sec=data.get("source_start_sec"),
        source_end_sec=data.get("source_end_sec"),
    )


def list_jobs(config: AppConfig) -> list[ClipJob]:
    data_dir = config.resolve_path(config.paths.data_dir)
    if not data_dir.exists():
        return []
    jobs: list[ClipJob] = []
    for child in sorted(data_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        job = load_job(config, child.name)
        if job:
            jobs.append(job)
    return jobs
