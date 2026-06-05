"""Fetch recent clips from the Twitch Helix API."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import TwitchClip


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
