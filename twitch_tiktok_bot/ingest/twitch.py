"""Fetch recent clips from the Twitch Helix API."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import TwitchClip, TwitchVod


def _parse_api_datetime(value: object) -> datetime | None:
    """Parse Twitch API timestamps (str or datetime) to UTC datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except ValueError:
            return None
    return None


def _format_api_datetime(value: object) -> str:
    dt = _parse_api_datetime(value)
    return dt.isoformat() if dt else ""


def _parse_duration_seconds(value: object) -> float:
    if not value:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    dur = str(value)
    h = re.search(r"(\d+)h", dur)
    m = re.search(r"(\d+)m", dur)
    s = re.search(r"(\d+)s", dur)
    return float(
        (int(h.group(1)) * 3600 if h else 0)
        + (int(m.group(1)) * 60 if m else 0)
        + (int(s.group(1)) if s else 0)
    )


async def _fetch_clips_async(config: AppConfig, days_back: int = 7) -> list[TwitchClip]:
    twitch = config.twitch
    if not all([twitch.client_id, twitch.client_secret, twitch.broadcaster_id]):
        raise ValueError(
            "Twitch credentials missing. Copy config.local.yaml.example to config.local.yaml "
            "and set client_id, client_secret, and broadcaster_id."
        )

    from twitchAPI.twitch import Twitch

    api = await Twitch(twitch.client_id, twitch.client_secret)

    started_at = datetime.now(timezone.utc) - timedelta(days=days_back)
    clips: list[TwitchClip] = []

    async for clip in api.get_clips(
        broadcaster_id=twitch.broadcaster_id,
        started_at=started_at,
        first=min(twitch.max_clips, 100),
    ):
        clips.append(
            TwitchClip(
                id=clip.id,
                url=clip.url,
                title=clip.title or "",
                game_id=clip.game_id or "",
                view_count=clip.view_count or 0,
                created_at=clip.created_at or "",
            )
        )
        if len(clips) >= twitch.max_clips:
            break

    game_ids = list({c.game_id for c in clips if c.game_id})
    game_names: dict[str, str] = {}
    if game_ids:
        try:
            async for game in api.get_games(ids=game_ids):
                game_names[game.id] = game.name or ""
        except Exception:
            pass

    for clip in clips:
        if clip.game_id and clip.game_id in game_names:
            clip.game_name = game_names[clip.game_id]

    await api.close()
    return clips


def fetch_recent_clips(config: AppConfig, days_back: int = 7) -> list[TwitchClip]:
    """Synchronous wrapper around the Twitch API clip fetch."""
    return asyncio.run(_fetch_clips_async(config, days_back=days_back))


async def _fetch_vods_async(config: AppConfig, days_back: int = 30) -> list[TwitchVod]:
    twitch = config.twitch
    if not all([twitch.client_id, twitch.client_secret, twitch.broadcaster_id]):
        raise ValueError(
            "Twitch credentials missing. Copy config.local.yaml.example to config.local.yaml "
            "and set client_id, client_secret, and broadcaster_id."
        )

    from twitchAPI.twitch import Twitch
    from twitchAPI.type import VideoType

    api = await Twitch(twitch.client_id, twitch.client_secret)

    started_at = datetime.now(timezone.utc) - timedelta(days=days_back)
    vods: list[TwitchVod] = []

    async for video in api.get_videos(
        user_id=twitch.broadcaster_id,
        video_type=VideoType.ARCHIVE,
        first=min(twitch.max_vods, 100),
    ):
        created_dt = _parse_api_datetime(video.created_at)
        if created_dt and created_dt < started_at:
            continue

        vods.append(
            TwitchVod(
                id=video.id,
                url=video.url or f"https://www.twitch.tv/videos/{video.id}",
                title=video.title or "",
                duration_sec=_parse_duration_seconds(video.duration),
                view_count=video.view_count or 0,
                created_at=_format_api_datetime(video.created_at),
            )
        )
        if len(vods) >= twitch.max_vods:
            break

    await api.close()
    return vods


def fetch_recent_vods(config: AppConfig, days_back: int = 30) -> list[TwitchVod]:
    """Fetch recent stream VODs from the Twitch API."""
    return asyncio.run(_fetch_vods_async(config, days_back=days_back))
