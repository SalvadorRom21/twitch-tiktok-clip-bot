"""Twitch VOD URL helpers."""

from __future__ import annotations

import re


_VOD_PATTERNS = (
    re.compile(r"twitch\.tv/videos/(\d+)", re.I),
    re.compile(r"twitch\.tv/.+/v/(\d+)", re.I),
)


def is_vod_url(url: str) -> bool:
    return any(p.search(url) for p in _VOD_PATTERNS)


def is_clip_url(url: str) -> bool:
    lower = url.lower()
    return "clips.twitch.tv" in lower or "/clip/" in lower


def parse_vod_id(url: str) -> str | None:
    for pattern in _VOD_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None
