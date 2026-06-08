"""Long-form video URL helpers (Twitch VOD, YouTube, etc.)."""

from __future__ import annotations

import re

from twitch_tiktok_bot.ingest.vod import is_vod_url, parse_vod_id

_YOUTUBE_PATTERNS = (
    re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/live/)([\w-]{11})", re.I),
    re.compile(r"youtube\.com/embed/([\w-]{11})", re.I),
    re.compile(r"youtube\.com/shorts/([\w-]{11})", re.I),
)


def is_youtube_url(url: str) -> bool:
    return any(pattern.search(url) for pattern in _YOUTUBE_PATTERNS)


def parse_youtube_id(url: str) -> str | None:
    for pattern in _YOUTUBE_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def is_longform_url(url: str) -> bool:
    """True for multi-minute streams/VODs (Twitch VOD or YouTube watch/live)."""
    return is_vod_url(url) or is_youtube_url(url)


def parse_media_id(url: str) -> str | None:
    """Best-effort stable cache folder id from a media URL."""
    return parse_vod_id(url) or parse_youtube_id(url)
