"""Curated hard-negative / dual-boss label packs for Elden ML.

These teach the classifier the cases OpenCV color heuristics miss:
outdoor gold wash, map UI, pre-fight exploration, and dual boss HUD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from twitch_tiktok_bot.labels.elden_ml.clips import read_clips, save_manual_clip
from twitch_tiktok_bot.labels.elden_ml.dataset import label_counts

# Merica (ELDEN RING + Merica) — dual Tree Sentinels + Leyndell gold FPs.
MERICA_VOD_ID = "76cd3cf1089f"

# (kind, start, end, notes) — times from deep analysis of first dual fight.
MERICA_PACK: list[tuple[str, float, float, str]] = [
    (
        "not_fight",
        0.0,
        80.0,
        "Merica hard-neg: early explore / false red FX (no boss HUD)",
    ),
    (
        "not_fight",
        80.0,
        200.0,
        "Merica hard-neg: outdoor Leyndell wash / false bar blips",
    ),
    (
        "not_fight",
        400.0,
        448.0,
        "Merica hard-neg: stairs approach before dual HUD onset",
    ),
    (
        "incomplete",
        446.0,
        550.0,
        "Merica dual Tree Sentinel fight (two stacked boss bars)",
    ),
    (
        "not_fight",
        570.0,
        680.0,
        "Merica hard-neg: post-fight outdoor gold / map false FELLED",
    ),
]


def _pack_already_loaded(data_dir: Path, vod_id: str, notes_prefix: str) -> bool:
    """Skip re-import when the same curated notes already exist."""
    for row in read_clips(data_dir, vod_id=vod_id):
        notes = str(row.get("notes") or "")
        if notes.startswith(notes_prefix) or "Merica hard-neg" in notes or "Merica dual" in notes:
            return True
    return False


def apply_merica_hardneg_pack(
    data_dir: Path,
    *,
    force: bool = False,
    extract_video: bool = False,
    ffmpeg: str = "ffmpeg",
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Import Merica hard-negatives + dual-fight positives into the ML dataset."""
    work = data_dir / MERICA_VOD_ID
    if not work.is_dir():
        raise FileNotFoundError(
            f"Merica VOD folder missing: {work}. Upload/scan the VOD first."
        )
    if not force and _pack_already_loaded(data_dir, MERICA_VOD_ID, "Merica"):
        return {
            "ok": True,
            "skipped": True,
            "reason": "Merica pack already present (pass force=true to re-add)",
            "vod_id": MERICA_VOD_ID,
            "clips_added": 0,
            "frame_labels_added": 0,
            "label_counts": label_counts(data_dir),
        }

    clips_out: list[dict] = []
    frames_added = 0
    total = len(MERICA_PACK)
    for i, (kind, start, end, notes) in enumerate(MERICA_PACK):
        if on_progress:
            on_progress(
                {
                    "phase": "hardneg_pack",
                    "message": f"Merica pack ({i + 1}/{total}): {notes[:48]}…",
                    "current": i + 1,
                    "total": total,
                    "pct": round(((i + 1) / total) * 100.0, 1),
                }
            )
        result = save_manual_clip(
            data_dir,
            vod_id=MERICA_VOD_ID,
            start_sec=start,
            end_sec=end,
            kind=kind,
            notes=notes,
            extract_video=extract_video,
            expand_frames=True,
            ffmpeg=ffmpeg,
        )
        clips_out.append(result.get("clip") or {})
        frames_added += int(result.get("frame_labels_added") or 0)

    return {
        "ok": True,
        "skipped": False,
        "vod_id": MERICA_VOD_ID,
        "clips_added": len(clips_out),
        "frame_labels_added": frames_added,
        "clips": clips_out,
        "label_counts": label_counts(data_dir),
    }
