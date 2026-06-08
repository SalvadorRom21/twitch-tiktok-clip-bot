"""
Reference profile from curator-approved Apex YouTube Shorts.

Used to tune montage length, segment count, and which moment sources we trust.
"""

from __future__ import annotations

from dataclasses import dataclass

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.labels.curator import CuratorFightRef, best_clip_in_curator_fight
from twitch_tiktok_bot.models import ClipAnalysis, EditSegment
from twitch_tiktok_bot.plan.action import (
    expand_gunfight_window,
    fight_arc_quality,
    peak_cluster_is_gunfire,
    segment_action_score,
    transcript_combat_score,
    transcript_text_in_range,
)
from twitch_tiktok_bot.plan.duration import (
    has_duration_cap,
    max_clip_duration_sec,
    max_segment_duration_sec,
)
from twitch_tiktok_bot.plan.game_profiles import apex_fight_windows
from twitch_tiktok_bot.plan.moments import RankedMoment, tighten_segment


@dataclass(frozen=True)
class ReferenceShort:
    url: str
    duration_sec: float
    theme: str
    notes: str


# Curator picks — team wipe, duel, movement outplay; 11–21s, one payoff each.
REFERENCE_GOOD_SHORTS: tuple[ReferenceShort, ...] = (
    ReferenceShort(
        url="https://www.youtube.com/shorts/-y03bptzHN4",
        duration_sec=14.0,
        theme="team_wipe",
        notes="Pro squad wipe with cast payoff; single continuous fight",
    ),
    ReferenceShort(
        url="https://www.youtube.com/shorts/3oxM2mBZSKw",
        duration_sec=21.0,
        theme="1v1_duel",
        notes="Head-to-head fight; tight focus on combat",
    ),
    ReferenceShort(
        url="https://www.youtube.com/shorts/26sN8DeY9d0",
        duration_sec=11.0,
        theme="movement_outplay",
        notes="Short single highlight; instant hook, no filler",
    ),
)

REFERENCE_TARGET_DURATION_SEC = 15.0
REFERENCE_MAX_DURATION_SEC = 22.0
REFERENCE_MIN_DURATION_SEC = 10.0
REFERENCE_MAX_SEGMENTS = 2
REFERENCE_MIN_FIGHT_SCORE = 10.0
REFERENCE_MIN_ACTION_SCORE = 6.0

PRIMARY_MOMENT_REASONS = frozenset({"gunfight"})
FILLER_MOMENT_REASONS = frozenset({"pov sweep", "reaction peak"})


def apex_shorts_active(config: AppConfig) -> bool:
    style = (config.editing.clip_style or "default").strip().lower()
    if style == "apex_shorts":
        return True
    profile = (config.editing.game_profile or "auto").strip().lower()
    return profile == "apex" and style in ("default", "auto")


def reference_duration_stats() -> tuple[float, float, float]:
    durations = [ref.duration_sec for ref in REFERENCE_GOOD_SHORTS]
    return min(durations), sum(durations) / len(durations), max(durations)


def moment_source_score_boost(reason: str, config: AppConfig) -> float:
    if not apex_shorts_active(config):
        return 1.0
    if reason == "gunfight":
        return 2.2
    if reason == "payoff line":
        return 1.1
    if reason in FILLER_MOMENT_REASONS:
        return 0.15
    return 0.6


def skip_pov_sweep(config: AppConfig, profile: str) -> bool:
    return apex_shorts_active(config) and profile == "apex"


def fight_window_has_payoff(analysis: ClipAnalysis, start: float, end: float) -> bool:
    text = transcript_text_in_range(analysis, start, end)
    if transcript_combat_score(text) >= 4.0:
        return True
    peaks = [p for p in analysis.loud_peaks if start <= p.time <= end]
    return len(peaks) >= 4 and peak_cluster_is_gunfire(peaks)


def try_single_fight_short(
    candidates: list[RankedMoment],
    analysis: ClipAnalysis,
    config: AppConfig,
    *,
    curator_refs: list[CuratorFightRef] | None = None,
) -> list[EditSegment] | None:
    """
    One continuous gunfight clip (like 11–14s reference Shorts) when action is clear.
    """
    if not apex_shorts_active(config):
        return None

    editing = config.editing
    clip_cap = max_clip_duration_sec(editing)
    seg_cap = max_segment_duration_sec(editing)

    if curator_refs:
        best_curator: tuple[float, float, float] | None = None
        for ref in curator_refs:
            window = best_clip_in_curator_fight(
                analysis,
                ref,
                max_duration=clip_cap,
                min_duration=REFERENCE_MIN_DURATION_SEC,
            )
            if window is None:
                continue
            if best_curator is None or window[2] > best_curator[2]:
                best_curator = window
        if best_curator is not None and best_curator[2] >= REFERENCE_MIN_FIGHT_SCORE:
            start, end, _score = best_curator
            seg = EditSegment(start=start, end=end, reason="gunfight")
            seg = tighten_segment(
                seg,
                analysis,
                editing.min_segment_sec,
                seg_cap,
            )
            length = seg.end - seg.start
            if length >= REFERENCE_MIN_DURATION_SEC and (
                not has_duration_cap(editing) or length <= clip_cap
            ):
                return [seg]

    def _fight_rank(moment: RankedMoment) -> float:
        return moment.score + fight_arc_quality(
            analysis, moment.start, moment.end
        ) * 2.5

    fights = sorted(
        [m for m in candidates if m.reason == "gunfight"],
        key=_fight_rank,
        reverse=True,
    )
    if not fights:
        return None

    best = fights[0]
    if best.score < REFERENCE_MIN_FIGHT_SCORE:
        return None

    start, end = expand_gunfight_window(
        analysis,
        best.start,
        best.end,
        max_duration=clip_cap,
        max_prelude=20.0,
        min_duration=REFERENCE_MIN_DURATION_SEC,
    )
    seg = EditSegment(start=start, end=end, reason=best.reason)
    seg = tighten_segment(
        seg,
        analysis,
        editing.min_segment_sec,
        seg_cap,
    )
    length = seg.end - seg.start
    if length < REFERENCE_MIN_DURATION_SEC:
        return None
    if has_duration_cap(editing) and length > clip_cap:
        return None

    action_sc = segment_action_score(
        analysis,
        seg.start,
        seg.end,
        require_combat_for_apex=True,
    )
    if action_sc < REFERENCE_MIN_ACTION_SCORE:
        return None
    if not fight_window_has_payoff(analysis, seg.start, seg.end):
        return None

    return [seg]


def prefer_gunfight_anchor(candidates: list[RankedMoment]) -> float | None:
    """Cluster montage around the best gunfight, not a random loud moment."""
    fights = sorted(
        [m for m in candidates if m.reason == "gunfight"],
        key=lambda m: m.score,
        reverse=True,
    )
    if not fights:
        return None
    return (fights[0].start + fights[0].end) / 2


def apex_fight_windows_for_shorts(analysis: ClipAnalysis) -> list[tuple[float, float, float]]:
    """Stricter fight detection aligned with short-form Apex highlights."""
    windows = apex_fight_windows(analysis, window_sec=16.0, min_peaks=4)
    filtered: list[tuple[float, float, float]] = []
    for start, end, score in windows:
        if score < REFERENCE_MIN_FIGHT_SCORE:
            continue
        if not fight_window_has_payoff(analysis, start, end):
            continue
        length = end - start
        if length < 6.0 or length > REFERENCE_MAX_DURATION_SEC + 4.0:
            continue
        filtered.append((start, end, score))
    return filtered
