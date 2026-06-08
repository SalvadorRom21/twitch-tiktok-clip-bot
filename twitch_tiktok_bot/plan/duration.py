"""Clip duration limits (0 = no cap — export full labeled fights)."""

from __future__ import annotations

from twitch_tiktok_bot.config import EditingConfig

_UNCAPPED = 1e9


def has_duration_cap(editing: EditingConfig) -> bool:
    return editing.max_duration_sec > 0


def max_clip_duration_sec(editing: EditingConfig) -> float:
    if editing.max_duration_sec <= 0:
        return _UNCAPPED
    return editing.max_duration_sec


def max_segment_duration_sec(editing: EditingConfig) -> float:
    if editing.max_segment_sec <= 0:
        return _UNCAPPED
    return editing.max_segment_sec


def target_clip_duration_sec(editing: EditingConfig) -> float:
    if editing.target_duration_sec <= 0:
        return _UNCAPPED
    if editing.max_duration_sec <= 0:
        return editing.target_duration_sec
    return min(editing.target_duration_sec, editing.max_duration_sec)
