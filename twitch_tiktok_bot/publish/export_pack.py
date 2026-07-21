"""Build ready-to-post packs for TikTok, Instagram Reels, and YouTube Shorts."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.publish.captions import (
    instagram_caption,
    tiktok_caption,
    youtube_description,
    youtube_title,
)
from twitch_tiktok_bot.publish.trim import copy_or_link


def pack_root(config: AppConfig, clip_id: str) -> Path:
    return config.resolve_path(config.paths.output_dir) / "publish" / clip_id


def write_export_pack(
    config: AppConfig,
    *,
    clip_id: str,
    video_path: Path,
    caption: str,
    title: str = "",
) -> dict:
    """Copy video + platform captions into output/publish/{clip_id}/."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video missing: {video_path}")

    root = pack_root(config, clip_id)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    platforms = {
        "tiktok": {
            "caption": tiktok_caption(caption, title=title),
            "files": {"caption.txt": None},
        },
        "instagram": {
            "caption": instagram_caption(caption, title=title),
            "files": {"caption.txt": None},
        },
        "youtube": {
            "caption": None,
            "files": {
                "title.txt": youtube_title(caption, title=title),
                "description.txt": youtube_description(caption, title=title),
            },
        },
    }

    result_platforms: dict[str, dict] = {}
    for name, spec in platforms.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        dest_video = folder / "video.mp4"
        copy_or_link(video_path, dest_video)
        files_written = {"video": str(dest_video)}
        if spec["caption"] is not None:
            cap_path = folder / "caption.txt"
            cap_path.write_text(str(spec["caption"]), encoding="utf-8")
            files_written["caption"] = str(cap_path)
        for fname, content in (spec["files"] or {}).items():
            if content is None:
                continue
            path = folder / fname
            path.write_text(str(content), encoding="utf-8")
            files_written[fname.replace(".txt", "")] = str(path)
        # How-to note for manual mobile upload
        note = folder / "HOW_TO_POST.txt"
        if name == "tiktok":
            note.write_text(
                "Open TikTok app → + → Upload → pick video.mp4 → paste caption.txt\n",
                encoding="utf-8",
            )
        elif name == "instagram":
            note.write_text(
                "Open Instagram app → Reels → upload video.mp4 → paste caption.txt\n",
                encoding="utf-8",
            )
        else:
            note.write_text(
                "Upload via Preview UI (YouTube button) or YouTube Studio → Shorts.\n"
                "Use title.txt + description.txt.\n",
                encoding="utf-8",
            )
        result_platforms[name] = {
            "dir": str(folder),
            "files": files_written,
        }

    manifest = {
        "clip_id": clip_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(video_path),
        "title": title,
        "platforms": result_platforms,
        "tiktok_api": "not_configured — use pack + mobile app",
        "instagram_api": "not_configured — use pack + mobile app",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "pack_dir": str(root),
        "manifest": str(manifest_path),
        "platforms": result_platforms,
    }
