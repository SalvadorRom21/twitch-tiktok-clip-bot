"""Labeled frame dataset I/O for Elden ML training."""

from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2

from twitch_tiktok_bot.labels.elden_ml.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    frames_dir,
    labels_path,
    ml_root,
)
from twitch_tiktok_bot.labels.elden_ml.crop import crop_and_resize


def ensure_ml_dirs(data_dir: Path) -> Path:
    root = ml_root(data_dir)
    frames_dir(data_dir).mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_labels(data_dir: Path) -> list[dict]:
    path = labels_path(data_dir)
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def label_counts(data_dir: Path) -> dict[str, int]:
    counts = Counter(r.get("label") for r in read_labels(data_dir) if r.get("label") in CLASS_TO_IDX)
    return {name: int(counts.get(name, 0)) for name in CLASS_NAMES}


def append_label(
    data_dir: Path,
    *,
    label: str,
    source_path: Path,
    vod_id: str = "",
    time_sec: float | None = None,
    image_size: int = 224,
    clip_id: str = "",
) -> dict:
    """Copy a cropped frame into the ML dataset and append a labels.jsonl row."""
    if label not in CLASS_TO_IDX:
        raise ValueError(f"Unknown label {label!r}; expected one of {CLASS_NAMES}")
    ensure_ml_dirs(data_dir)
    img = cv2.imread(str(source_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {source_path}")
    crop = crop_and_resize(img, size=image_size)
    frame_id = uuid.uuid4().hex[:12]
    rel = f"{label}/{frame_id}.jpg"
    dest = frames_dir(data_dir) / label
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{frame_id}.jpg"
    cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    row = {
        "id": frame_id,
        "label": label,
        "path": rel.replace("\\", "/"),
        "vod_id": vod_id,
        "time_sec": time_sec,
        "source": str(source_path).replace("\\", "/"),
        "clip_id": clip_id or None,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }
    with labels_path(data_dir).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def rewrite_labels(data_dir: Path, rows: list[dict]) -> None:
    ensure_ml_dirs(data_dir)
    path = labels_path(data_dir)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )


def remove_labels_for_clip(data_dir: Path, clip_id: str) -> int:
    """Delete frame labels (and crop files) that came from a manual clip."""
    if not clip_id:
        return 0
    rows = read_labels(data_dir)
    keep: list[dict] = []
    removed = 0
    root = frames_dir(data_dir)
    for row in rows:
        belongs = row.get("clip_id") == clip_id
        source = str(row.get("source") or "").replace("\\", "/")
        if not belongs and f"/manual_clips/_frames/{clip_id}/" in source:
            belongs = True
        if belongs:
            rel = row.get("path")
            if rel:
                crop = root / rel
                if crop.exists():
                    crop.unlink(missing_ok=True)
            removed += 1
            continue
        keep.append(row)
    if removed:
        rewrite_labels(data_dir, keep)
    # Clear temp extract dir for this clip
    tmp = None
    # Prefer searching under data/*/manual_clips/_frames/{clip_id}
    for child in data_dir.iterdir() if data_dir.exists() else []:
        if not child.is_dir() or child.name == "reference":
            continue
        candidate = child / "manual_clips" / "_frames" / clip_id
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
    return removed


def list_unlabeled_scan_frames(
    data_dir: Path,
    *,
    vod_id: str | None = None,
    limit: int = 40,
) -> list[dict]:
    """Return VOD scan-frame paths not yet used as ML label sources."""
    labeled_sources = {
        Path(r["source"]).name
        for r in read_labels(data_dir)
        if r.get("source")
    }
    labeled_sources |= {
        Path(r["source"]).as_posix()
        for r in read_labels(data_dir)
        if r.get("source")
    }
    out: list[dict] = []
    children = sorted(data_dir.iterdir())
    for child in children:
        if not child.is_dir() or child.name == "reference":
            continue
        if vod_id and child.name != vod_id:
            continue
        frames = child / "boss_scan_frames"
        if not frames.is_dir():
            continue
        for path in sorted(frames.glob("frame_*.jpg")):
            if path.name in labeled_sources:
                continue
            # time from filename frame_XXXXX @ 2s interval
            try:
                idx = int(path.stem.split("_")[1])
                t = idx * 2.0
            except (IndexError, ValueError):
                t = None
            out.append(
                {
                    "vod_id": child.name,
                    "path": str(path).replace("\\", "/"),
                    "rel": f"{child.name}/boss_scan_frames/{path.name}",
                    "time_sec": t,
                    "name": path.name,
                }
            )
            if len(out) >= limit:
                return out
    return out


def clear_ml_dataset(data_dir: Path, *, keep_config: bool = True) -> dict:
    """Remove labeled crops + labels.jsonl. Optionally keep config.json."""
    root = ml_root(data_dir)
    removed = []
    labels = labels_path(data_dir)
    if labels.exists():
        labels.unlink()
        removed.append("labels.jsonl")
    clips = root / "clips.jsonl"
    if clips.exists():
        clips.unlink()
        removed.append("clips.jsonl")
    fd = frames_dir(data_dir)
    if fd.is_dir():
        shutil.rmtree(fd, ignore_errors=True)
        removed.append("frames/")
    model = root / "model.pt"
    if model.exists():
        model.unlink()
        removed.append("model.pt")
    if not keep_config:
        cfg = root / "config.json"
        if cfg.exists():
            cfg.unlink()
            removed.append("config.json")
    return {"removed": removed}
