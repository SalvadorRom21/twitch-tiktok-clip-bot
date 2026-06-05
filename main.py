#!/usr/bin/env python3
"""CLI for Twitch-to-TikTok clip automation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from twitch_tiktok_bot.config import load_config
from twitch_tiktok_bot.ingest.twitch import fetch_recent_clips, fetch_recent_vods
from twitch_tiktok_bot.pipeline import (
    process_clip_url,
    process_media_url,
    process_twitch_clip,
    process_twitch_vod,
    process_vod_url,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Twitch clips/VODs, analyze them, and render TikTok shorts.",
    )
    parser.add_argument(
        "--clip-url",
        help="Process a single Twitch clip URL",
    )
    parser.add_argument(
        "--vod-url",
        help="Process a full Twitch VOD/stream recording URL",
    )
    parser.add_argument(
        "--media-url",
        help="Auto-detect clip or VOD URL and process",
    )
    parser.add_argument(
        "--clip-id",
        help="Optional ID for output naming",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title metadata for hooks/hashtags",
    )
    parser.add_argument(
        "--max-shorts",
        type=int,
        default=None,
        help="Max TikTok shorts to generate from a VOD (default from config)",
    )
    parser.add_argument(
        "--fetch-clips",
        action="store_true",
        help="Fetch recent clips from Twitch API and process each one",
    )
    parser.add_argument(
        "--fetch-vods",
        action="store_true",
        help="Fetch recent stream VODs from Twitch API and process each one",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days back to fetch clips/VODs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config.local.yaml or config.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root for resolving relative paths",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the preview web UI server",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Web server host (with --web, default from config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Web server port (with --web, default from config)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(config_path=args.config, project_root=args.project_root)

    if args.web:
        if args.host:
            config.web.host = args.host
        if args.port:
            config.web.port = args.port
        from twitch_tiktok_bot.web.app import run_server

        print(f"Starting preview UI at http://{config.web.host}:{config.web.port}")
        run_server(config)
        return 0

    if args.vod_url:
        outputs = process_vod_url(
            url=args.vod_url,
            config=config,
            vod_id=args.clip_id,
            vod_title=args.title,
            max_shorts=args.max_shorts,
        )
        print(f"\nGenerated {len(outputs)} short(s).")
        return 0

    if args.media_url:
        outputs = process_media_url(
            url=args.media_url,
            config=config,
            media_id=args.clip_id,
            title=args.title,
            max_shorts=args.max_shorts,
        )
        print(f"\nGenerated {len(outputs)} short(s).")
        return 0

    if args.clip_url:
        process_clip_url(
            url=args.clip_url,
            config=config,
            clip_id=args.clip_id,
            clip_title=args.title,
        )
        return 0

    if args.fetch_clips:
        clips = fetch_recent_clips(config, days_back=args.days)
        if not clips:
            print("No clips found.")
            return 0
        print(f"Found {len(clips)} clip(s). Processing...")
        for clip in clips:
            print(f"\n=== {clip.title} ({clip.id}) ===")
            process_twitch_clip(clip, config)
        return 0

    if args.fetch_vods:
        vods = fetch_recent_vods(config, days_back=args.days)
        if not vods:
            print("No VODs found.")
            return 0
        print(f"Found {len(vods)} VOD(s). Processing...")
        for vod in vods:
            print(f"\n=== {vod.title} ({vod.id}) — {vod.duration_sec/60:.0f} min ===")
            process_twitch_vod(vod, config, max_shorts=args.max_shorts)
        return 0

    parser.print_help()
    print("\nExamples:")
    print("  python main.py --clip-url https://clips.twitch.tv/SomeClipSlug")
    print("  python main.py --vod-url https://www.twitch.tv/videos/1234567890")
    print("  python main.py --media-url https://www.twitch.tv/videos/1234567890")
    print("  python main.py --fetch-vods --max-shorts 3")
    print("  python main.py --web")
    return 1


if __name__ == "__main__":
    sys.exit(main())
