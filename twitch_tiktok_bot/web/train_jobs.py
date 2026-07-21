"""Background scan jobs for supervised Elden Ring training (terminal-first hybrid)."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.labels.elden_boss_detect import save_scan_result
from twitch_tiktok_bot.labels.elden_ml.hybrid_scan import scan_boss_fights_hybrid
from twitch_tiktok_bot.labels.supervised import (
    ensure_supervised_store,
    load_supervised_store,
    save_supervised_store,
)
from twitch_tiktok_bot.web.job_progress import finish_job, start_job, update_job

_lock = threading.Lock()
_active: dict[str, threading.Thread] = {}


def _run_scan(config: AppConfig, vod_id: str) -> None:
    work_dir = config.resolve_path(config.paths.data_dir) / vod_id
    store = load_supervised_store(work_dir) or ensure_supervised_store(work_dir, vod_id)
    store.scan_status = "scanning"
    store.scan_error = ""
    save_supervised_store(work_dir, store)
    start_job(vod_id, kind="scan", message="Starting terminal-first boss attempt scan…")

    try:
        video = _find_video(work_dir, vod_id)
        ffmpeg = config.render.ffmpeg_path or "ffmpeg"
        data_dir = config.resolve_path(config.paths.data_dir)

        def on_progress(fields: dict) -> None:
            update_job(vod_id, **fields)

        result = scan_boss_fights_hybrid(
            video,
            work_dir,
            vod_id,
            data_dir,
            ffmpeg=ffmpeg,
            on_progress=on_progress,
        )
        save_scan_result(work_dir, result)
        store = load_supervised_store(work_dir) or store
        store.scan_status = "done"
        store.scan_error = ""
        finish_job(
            vod_id,
            status="done",
            message=f"Done — {len(result.candidates)} attempt clip(s) found",
        )
    except Exception as exc:  # noqa: BLE001
        store = load_supervised_store(work_dir) or store
        store.scan_status = "failed"
        store.scan_error = str(exc)
        finish_job(vod_id, status="failed", message="Scan failed", error=str(exc))
    save_supervised_store(work_dir, store)
    with _lock:
        _active.pop(vod_id, None)


def _find_video(work_dir: Path, vod_id: str) -> Path:
    for path in sorted(work_dir.glob("*.mp4")):
        return path
    for path in sorted(work_dir.glob("*.mkv")):
        return path
    for path in sorted(work_dir.glob("*.webm")):
        return path
    raise FileNotFoundError(f"No video in {work_dir}")


def save_upload(
    config: AppConfig,
    filename: str,
    data: bytes,
    title: str = "",
) -> tuple[str, Path]:
    """Save uploaded TikTok VOD; return (vod_id, work_dir)."""
    vod_id = uuid.uuid4().hex[:12]
    work_dir = config.resolve_path(config.paths.data_dir) / vod_id
    work_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower() or ".mp4"
    if ext not in {".mp4", ".mkv", ".webm", ".mov"}:
        ext = ".mp4"
    dest = work_dir / f"{vod_id}{ext}"
    dest.write_bytes(data)
    store = ensure_supervised_store(work_dir, vod_id, title=title or filename)
    store.source = "tiktok_upload"
    store.title = title or Path(filename).stem
    save_supervised_store(work_dir, store)
    return vod_id, work_dir


def start_scan(config: AppConfig, vod_id: str) -> str:
    with _lock:
        if vod_id in _active:
            return "already_running"
    thread = threading.Thread(target=_run_scan, args=(config, vod_id), daemon=True)
    with _lock:
        _active[vod_id] = thread
    thread.start()
    return "started"


def scan_status(config: AppConfig, vod_id: str) -> str:
    with _lock:
        if vod_id in _active:
            return "scanning"
    work_dir = config.resolve_path(config.paths.data_dir) / vod_id
    store = load_supervised_store(work_dir)
    return store.scan_status if store else "idle"
