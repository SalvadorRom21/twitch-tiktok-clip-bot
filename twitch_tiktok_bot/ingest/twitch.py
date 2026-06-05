"""Fetch recent clips from the Twitch Helix API."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import TwitchClip, TwitchVod


async def _fetch_clips_async(config: AppConfig, days_back: int = 7) -> list[TwitchClip]:
    twitch = config.twitch
    if not all([twitch.client_id, twitch.client_secret, twitch.broadcaster_id]):
        raise ValueError(
            "Twitch credentials missing. Set client_id, client_secret, and broadcaster_id "
            "in config.yaml or TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET / TWITCH_BROADCASTER_ID."
        )

    from twitchAPI.twitch import Twitch

    api = await Twitch(twitch.client_id, twitch.client_secret)
    await api.set_app_authentication()

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

    await api.close()
    return clips


def fetch_recent_clips(config: AppConfig, days_back: int = 7) -> list[TwitchClip]:
    """Synchronous wrapper around the Twitch API clip fetch."""
    return asyncio.run(_fetch_clips_async(config, days_back=days_back))


async def _fetch_vods_async(config: AppConfig, days_back: int = 30) -> list[TwitchVod]:
    twitch = config.twitch
    if not all([twitch.client_id, twitch.client_secret, twitch.broadcaster_id]):
        raise ValueError(
            "Twitch credentials missing. Set client_id, client_secret, and broadcaster_id."
        )

    from twitchAPI.twitch import Twitch

    api = await Twitch(twitch.client_id, twitch.client_secret)
    await api.set_app_authentication()

    started_at = datetime.now(timezone.utc) - timedelta(days=days_back)
    vods: list[TwitchVod] = []

    async for video in api.get_videos(
        user_id=twitch.broadcaster_id,
        video_type="archive",
        first=min(twitch.max_vods, 100),
    ):
        created = video.created_at or ""
        if created:
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if created_dt < started_at:
                    continue
            except ValueError:
                pass

        duration = 0.0
        if video.duration:
            dur = video.duration
            h = re.search(r"(\d+)h", dur)
            m = re.search(r"(\d+)m", dur)
            s = re.search(r"(\d+)s", dur)
            duration = (
                (int(h.group(1)) * 3600 if h else 0)
                + (int(m.group(1)) * 60 if m else 0)
                + (int(s.group(1)) if s else 0)
            )

        vods.append(
            TwitchVod(
                id=video.id,
                url=video.url or f"https://www.twitch.tv/videos/{video.id}",
                title=video.title or "",
                duration_sec=duration,
                view_count=video.view_count or 0,
                created_at=created,
            )
        )
        if len(vods) >= twitch.max_vods:
            break

    await api.close()
    return vods


def fetch_recent_vods(config: AppConfig, days_back: int = 30) -> list[TwitchVod]:
    """Fetch recent stream VODs from the Twitch API."""
    return asyncio.run(_fetch_vods_async(config, days_back=days_back))
