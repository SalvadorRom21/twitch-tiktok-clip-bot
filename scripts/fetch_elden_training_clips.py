#!/usr/bin/env python3
"""Discover Elden Ring clips on the broadcaster's Twitch channel and download for labeling."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twitch_tiktok_bot.config import load_config
from twitch_tiktok_bot.ingest.download import resolve_video_path

ELDEN_PATTERN = re.compile(
    r"elden|elden lord|margit|godrick|renala|radahn|malenia|"
    r"mohg|rykard|maliketh|radagon|tree sentinel|red wolf|"
    r"boss|great enemy|legend felled|you died",
    re.I,
)

SKIP_PATTERN = re.compile(
    r"shaco|overwatch|outer wilds|league|apex|valorant|"
    r"daily challenge|sip and hit",
    re.I,
)


def discover_clips(channel: str) -> list[dict]:
    """List clips from Twitch via yt-dlp (works when Helix returns none)."""
    url = f"https://www.twitch.tv/{channel}/clips?filter=clips&range=all"
    venv_ytdlp = ROOT / ".venv" / "Scripts" / "yt-dlp.exe"
    ytdlp = str(venv_ytdlp if venv_ytdlp.exists() else "yt-dlp")
    proc = subprocess.run(
        [
            ytdlp,
            "--no-update",
            "--flat-playlist",
            "--print",
            "%(id)s|||%(title)s|||%(url)s",
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "yt-dlp failed")

    clips: list[dict] = []
    for line in proc.stdout.splitlines():
        if "|||" not in line:
            continue
        clip_id, title, clip_url = line.split("|||", 2)
        title = title.strip()
        if SKIP_PATTERN.search(title):
            continue
        if not ELDEN_PATTERN.search(title):
            continue
        clips.append(
            {
                "id": clip_id.strip(),
                "title": title,
                "url": clip_url.strip(),
            }
        )
    return clips


def download_clips(clips: list[dict], *, limit: int = 12) -> list[dict]:
    config = load_config(project_root=ROOT)
    data_dir = config.resolve_path(config.paths.data_dir)
    manifest: list[dict] = []

    for clip in clips[:limit]:
        work_dir = data_dir / clip["id"]
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{clip['id']}] {clip['title'][:60]}")
        try:
            video_path, cached = resolve_video_path(
                clip["url"],
                work_dir,
                video_id=clip["id"],
            )
            status = "cached" if cached else "downloaded"
            print(f"  -> {status}: {video_path.name}")
            manifest.append({**clip, "status": status, "video": video_path.name})
        except Exception as exc:
            print(f"  -> FAILED: {exc}")
            manifest.append({**clip, "status": "failed", "error": str(exc)})

    out = data_dir / "elden_training_clips.json"
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "channel": config.twitch.broadcaster_id,
        "clips": manifest,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nManifest -> {out}")
    ok = [c for c in manifest if c.get("status") in ("cached", "downloaded")]
    print(f"Ready to label: {len(ok)} clip(s)")
    return manifest


def main() -> int:
    channel = "salplaystv"
    if len(sys.argv) > 1:
        channel = sys.argv[1].lstrip("@")
    limit = 12
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])

    print(f"Discovering Elden Ring clips on @{channel}...")
    clips = discover_clips(channel)
    if not clips:
        print("No Elden Ring clips found.")
        return 1

    print(f"Found {len(clips)} candidate clip(s):")
    for clip in clips:
        print(f"  • {clip['title'][:55]} ({clip['id']})")

    download_clips(clips, limit=limit)
    print("\nOpen the labeler: http://127.0.0.1:8081/label")
    return 0


if __name__ == "__main__":
    sys.exit(main())
