"""ML Elden Ring boss-attempt detector.

Visual classes (research-backed):
  - other         — exploration, menus, fog, cutscenes, trash combat
  - boss_hud      — centered red boss HP bar + white boss name (attempt active)
  - you_died      — centered red YOU DIED on darkened frame (attempt end)
  - enemy_felled  — centered gold ENEMY FELLED banner (attempt end)

Attempt = sustained boss_hud onset → you_died | enemy_felled.
TikTok VODs: gameplay lives in the middle third of the frame.
"""

from twitch_tiktok_bot.labels.elden_ml.config import (
    CLASS_NAMES,
    MLConfig,
    load_ml_config,
    ml_root,
    save_ml_config,
)
from twitch_tiktok_bot.labels.elden_ml.scan import scan_boss_fights_ml

__all__ = [
    "CLASS_NAMES",
    "MLConfig",
    "load_ml_config",
    "ml_root",
    "save_ml_config",
    "scan_boss_fights_ml",
]
