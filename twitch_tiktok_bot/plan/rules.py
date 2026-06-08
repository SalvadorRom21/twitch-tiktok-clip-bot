"""Rule-based edit planning (no LLM required)."""

from __future__ import annotations

from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan
from twitch_tiktok_bot.plan.refine import build_catchy_plan


def create_rule_based_plan(
    analysis: ClipAnalysis,
    config: AppConfig,
    work_dir: Path | None = None,
) -> EditPlan:
    return build_catchy_plan(analysis, config, work_dir=work_dir)
