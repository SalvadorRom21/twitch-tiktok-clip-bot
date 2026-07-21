"""Manual training clips — human in/out points expanded into ML frame labels."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from twitch_tiktok_bot.labels.elden_ml.config import CLASS_TO_IDX, ml_root
from twitch_tiktok_bot.labels.elden_ml.dataset import (
    append_label,
    ensure_ml_dirs,
    label_counts,
    remove_labels_for_clip,
)

# How a human-labeled clip ends / what it is.
CLIP_KINDS = (
    "you_died",       # boss attempt ending in YOU DIED
    "enemy_felled",   # boss attempt ending in ENEMY FELLED
    "incomplete",     # boss HUD present, no terminal yet
    "not_fight",      # cutscene / explore / trash — hard negative
)


def clips_path(data_dir: Path) -> Path:
    return ml_root(data_dir) / "clips.jsonl"


def clips_dir_for_vod(data_dir: Path, vod_id: str) -> Path:
    return data_dir / vod_id / "manual_clips"


def read_clips(data_dir: Path, *, vod_id: str | None = None) -> list[dict]:
    path = clips_path(data_dir)
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if vod_id and row.get("vod_id") != vod_id:
            continue
        rows.append(row)
    return rows


def _append_clip_row(data_dir: Path, row: dict) -> None:
    ensure_ml_dirs(data_dir)
    path = clips_path(data_dir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _rewrite_clips(data_dir: Path, rows: list[dict]) -> None:
    ensure_ml_dirs(data_dir)
    path = clips_path(data_dir)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )


def _find_video(work_dir: Path, vod_id: str) -> Path:
    for pat in (f"{vod_id}.mp4", f"{vod_id}.mkv", f"{vod_id}.webm", f"{vod_id}.mov"):
        p = work_dir / pat
        if p.exists():
            return p
    for pat in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
        found = sorted(work_dir.glob(pat))
        if found:
            return found[0]
    raise FileNotFoundError(f"No video in {work_dir}")


def _extract_frame(video: Path, t: float, out: Path, ffmpeg: str = "ffmpeg") -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-ss", f"{t:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "3", str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.exists() and out.stat().st_size > 0


def _extract_clip_mp4(
    video: Path,
    start: float,
    end: float,
    out: Path,
    ffmpeg: str = "ffmpeg",
) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.5, end - start)
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video),
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=False, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.exists() and out.stat().st_size > 0


def _sample_times(start: float, end: float, interval: float = 2.0) -> list[float]:
    if end <= start:
        return [start]
    times: list[float] = []
    t = start
    while t <= end + 1e-6:
        times.append(round(t, 2))
        t += interval
    if times[-1] < end - 0.25:
        times.append(round(end, 2))
    return times


def expand_clip_to_frame_labels(
    data_dir: Path,
    *,
    video_path: Path,
    vod_id: str,
    clip_id: str,
    start_sec: float,
    end_sec: float,
    kind: str,
    ffmpeg: str = "ffmpeg",
    interval_sec: float = 2.0,
) -> list[dict]:
    """Sample frames from a manual clip and write ML labels."""
    work_tmp = data_dir / vod_id / "manual_clips" / "_frames" / clip_id
    work_tmp.mkdir(parents=True, exist_ok=True)
    labeled: list[dict] = []

    # Hard-negative context just before a real fight.
    pre_start = max(0.0, start_sec - 12.0)
    if kind != "not_fight" and pre_start < start_sec - 1.0:
        for t in _sample_times(pre_start, start_sec - 1.0, interval_sec):
            fp = work_tmp / f"pre_{int(t * 10):06d}.jpg"
            if _extract_frame(video_path, t, fp, ffmpeg=ffmpeg):
                labeled.append(
                    append_label(
                        data_dir,
                        label="other",
                        source_path=fp,
                        vod_id=vod_id,
                        time_sec=t,
                        clip_id=clip_id,
                    )
                )

    times = _sample_times(start_sec, end_sec, interval_sec)
    terminal_window = 4.0  # last N seconds get terminal class
    for t in times:
        if kind == "not_fight":
            label = "other"
        elif kind in ("you_died", "enemy_felled") and t >= end_sec - terminal_window:
            label = kind
        elif kind == "incomplete":
            label = "boss_hud"
        else:
            # Fighting body of the attempt.
            if t >= end_sec - terminal_window and kind in CLASS_TO_IDX:
                label = kind
            else:
                label = "boss_hud"

        fp = work_tmp / f"t_{int(t * 10):06d}.jpg"
        if not _extract_frame(video_path, t, fp, ffmpeg=ffmpeg):
            continue
        # Skip blank extracts
        img = cv2.imread(str(fp))
        if img is None:
            continue
        labeled.append(
            append_label(
                data_dir,
                label=label,
                source_path=fp,
                vod_id=vod_id,
                time_sec=t,
                clip_id=clip_id,
            )
        )
    return labeled


def save_manual_clip(
    data_dir: Path,
    *,
    vod_id: str,
    start_sec: float,
    end_sec: float,
    kind: str,
    notes: str = "",
    extract_video: bool = True,
    expand_frames: bool = True,
    ffmpeg: str = "ffmpeg",
) -> dict:
    """Save a human-marked clip, extract preview mp4, expand into ML frame labels."""
    kind = (kind or "").strip()
    if kind not in CLIP_KINDS:
        raise ValueError(f"kind must be one of {CLIP_KINDS}")
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 1.0, float(end_sec))
    work_dir = data_dir / vod_id
    if not work_dir.is_dir():
        raise FileNotFoundError(f"VOD not found: {vod_id}")
    video = _find_video(work_dir, vod_id)
    clip_id = uuid.uuid4().hex[:10]
    out_mp4: str | None = None
    if extract_video:
        dest = clips_dir_for_vod(data_dir, vod_id) / f"{clip_id}.mp4"
        if _extract_clip_mp4(video, start_sec, end_sec, dest, ffmpeg=ffmpeg):
            out_mp4 = str(dest).replace("\\", "/")

    frame_rows: list[dict] = []
    if expand_frames:
        frame_rows = expand_clip_to_frame_labels(
            data_dir,
            video_path=video,
            vod_id=vod_id,
            clip_id=clip_id,
            start_sec=start_sec,
            end_sec=end_sec,
            kind=kind,
            ffmpeg=ffmpeg,
        )

    row = {
        "id": clip_id,
        "vod_id": vod_id,
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "kind": kind,
        "notes": notes.strip(),
        "clip_path": out_mp4,
        "frame_labels": len(frame_rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_clip_row(data_dir, row)
    return {
        "clip": row,
        "frame_labels_added": len(frame_rows),
        "label_counts": label_counts(data_dir),
    }


def update_manual_clip(
    data_dir: Path,
    clip_id: str,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    kind: str | None = None,
    notes: str | None = None,
    extract_video: bool = True,
    expand_frames: bool = True,
    ffmpeg: str = "ffmpeg",
) -> dict:
    """Re-edit an existing clip: timing, kind, notes — refresh frames + preview."""
    rows = read_clips(data_dir)
    idx = next((i for i, r in enumerate(rows) if r.get("id") == clip_id), None)
    if idx is None:
        raise FileNotFoundError(clip_id)
    existing = dict(rows[idx])
    vod_id = str(existing["vod_id"])
    new_kind = (kind if kind is not None else existing.get("kind") or "you_died").strip()
    if new_kind not in CLIP_KINDS:
        raise ValueError(f"kind must be one of {CLIP_KINDS}")
    new_start = float(start_sec if start_sec is not None else existing["start_sec"])
    new_end = float(end_sec if end_sec is not None else existing["end_sec"])
    new_start = max(0.0, new_start)
    new_end = max(new_start + 1.0, new_end)
    new_notes = existing.get("notes") or ""
    if notes is not None:
        new_notes = notes.strip()

    work_dir = data_dir / vod_id
    video = _find_video(work_dir, vod_id)

    # Replace training frames tied to this clip.
    removed = remove_labels_for_clip(data_dir, clip_id) if expand_frames else 0

    out_mp4 = existing.get("clip_path")
    if extract_video:
        dest = clips_dir_for_vod(data_dir, vod_id) / f"{clip_id}.mp4"
        if _extract_clip_mp4(video, new_start, new_end, dest, ffmpeg=ffmpeg):
            out_mp4 = str(dest).replace("\\", "/")

    frame_rows: list[dict] = []
    if expand_frames:
        frame_rows = expand_clip_to_frame_labels(
            data_dir,
            video_path=video,
            vod_id=vod_id,
            clip_id=clip_id,
            start_sec=new_start,
            end_sec=new_end,
            kind=new_kind,
            ffmpeg=ffmpeg,
        )

    updated = {
        **existing,
        "start_sec": round(new_start, 2),
        "end_sec": round(new_end, 2),
        "kind": new_kind,
        "notes": new_notes,
        "clip_path": out_mp4,
        "frame_labels": len(frame_rows) if expand_frames else existing.get("frame_labels", 0),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    rows[idx] = updated
    _rewrite_clips(data_dir, rows)
    return {
        "clip": updated,
        "frame_labels_removed": removed,
        "frame_labels_added": len(frame_rows),
        "label_counts": label_counts(data_dir),
    }


def delete_manual_clip(data_dir: Path, clip_id: str) -> dict:
    rows = read_clips(data_dir)
    kept = [r for r in rows if r.get("id") != clip_id]
    removed = next((r for r in rows if r.get("id") == clip_id), None)
    if removed is None:
        raise FileNotFoundError(clip_id)
    _rewrite_clips(data_dir, kept)
    clip_path = removed.get("clip_path")
    if clip_path:
        p = Path(clip_path)
        if p.exists():
            p.unlink(missing_ok=True)
    frames_removed = remove_labels_for_clip(data_dir, clip_id)
    return {
        "deleted": clip_id,
        "remaining": len(kept),
        "frame_labels_removed": frames_removed,
        "label_counts": label_counts(data_dir),
    }
