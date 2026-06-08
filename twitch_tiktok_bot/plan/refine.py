"""Polish edit plans for punchy TikTok pacing."""

from __future__ import annotations

import re
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditEffect, EditPlan, EditSegment
from twitch_tiktok_bot.plan.action import segment_action_score
from twitch_tiktok_bot.plan.action_cuts import apply_action_cuts_to_plan
from twitch_tiktok_bot.plan.duration import target_clip_duration_sec
from twitch_tiktok_bot.plan.moments import (
    SETUP_WORDS,
    WEAK_HOOK,
    build_hook_text,
    rank_moments,
    score_window,
    tighten_segment,
)
from twitch_tiktok_bot.plan.pov_montage import discover_pov_moments, select_pov_montage_segments

WEAK_HOOK_PHRASES = re.compile(
    r"(how do i edit|how does it show|let me do \d+|testing|one sec)",
    re.IGNORECASE,
)


def _segments_overlap(a: EditSegment, b: EditSegment, gap: float = 1.5) -> bool:
    return not (a.end + gap <= b.start or b.end + gap <= a.start)


def _moment_to_segment(
    moment_start: float,
    moment_end: float,
    analysis: ClipAnalysis,
    config: AppConfig,
    reason: str,
) -> EditSegment:
    editing = config.editing
    seg = EditSegment(start=moment_start, end=moment_end, reason=reason)
    return tighten_segment(
        seg,
        analysis,
        editing.min_segment_sec,
        editing.max_segment_sec,
    )


def build_catchy_plan(
    analysis: ClipAnalysis,
    config: AppConfig,
    work_dir: Path | None = None,
) -> EditPlan:
    """Rule-based plan: scan full POV, pick diverse clips, mash into one short."""
    editing = config.editing

    if editing.scan_full_pov:
        candidates = discover_pov_moments(analysis, config, work_dir=work_dir)
        selected = select_pov_montage_segments(
            candidates, analysis, config, work_dir=work_dir
        )
        best_quote = candidates[0].quote if candidates else ""
    else:
        moments = rank_moments(
            analysis,
            min_len=editing.min_segment_sec,
            max_len=editing.max_segment_sec,
            limit=editing.max_montage_segments + 4,
            game_profile=editing.game_profile,
        )
        selected = []
        total = 0.0
        target = target_clip_duration_sec(editing)
        min_score = editing.min_moment_score
        apex_mode = editing.game_profile.lower() == "apex"

        for moment in moments:
            if len(selected) >= editing.max_montage_segments:
                break
            if moment.score < min_score:
                continue
            seg = _moment_to_segment(
                moment.start, moment.end, analysis, config, moment.reason
            )
            if seg.end - seg.start < editing.min_segment_sec:
                continue
            action_sc = segment_action_score(
                analysis,
                seg.start,
                seg.end,
                require_combat_for_apex=apex_mode,
            )
            if action_sc < min_score - 2.0:
                continue
            if any(_segments_overlap(seg, existing) for existing in selected):
                continue
            if total + (seg.end - seg.start) > target + 1.0 and selected:
                continue
            selected.append(seg)
            total += seg.end - seg.start

        if not selected and moments:
            best = moments[0]
            selected = [
                _moment_to_segment(
                    best.start, best.end, analysis, config, best.reason
                )
            ]
        best_quote = moments[0].quote if moments else ""

    if not selected:
        return EditPlan(
            target_duration_sec=min(analysis.duration, editing.target_duration_sec),
            segments=[
                EditSegment(
                    start=0.0,
                    end=min(analysis.duration, editing.max_duration_sec),
                    reason="full clip",
                )
            ],
            hook_text=build_hook_text(analysis, game_profile=editing.game_profile),
            hashtags=_default_hashtags(analysis),
            caption_style=editing.caption_style,
        )

    selected.sort(key=lambda s: s.start)

    effects = _collect_zooms(analysis, selected, config)

    plan = EditPlan(
        target_duration_sec=sum(s.end - s.start for s in selected),
        segments=selected,
        effects=effects,
        hook_text=build_hook_text(
            analysis, best_quote, game_profile=editing.game_profile
        ),
        hashtags=_default_hashtags(analysis),
        caption_style=editing.caption_style,
    )
    if editing.action_cut_enabled:
        plan = apply_action_cuts_to_plan(plan, analysis, config)
        if plan.segments:
            print(
                f"  [plan] action-cut edit: {len(plan.segments)} beats, "
                f"{plan.target_duration_sec:.0f}s total"
            )
    return plan


def refine_plan(
    plan: EditPlan, analysis: ClipAnalysis, config: AppConfig
) -> EditPlan:
    """Tighten and reorder any plan (LLM or rules) for TikTok delivery."""
    if config.editing.scan_full_pov:
        return build_catchy_plan(analysis, config)

    editing = config.editing
    segments: list[EditSegment] = []

    for seg in plan.segments:
        tightened = tighten_segment(
            seg, analysis, editing.min_segment_sec, editing.max_segment_sec
        )
        if tightened.end - tightened.start < editing.min_segment_sec:
            continue
        window_score = score_window(
            analysis, tightened.start, tightened.end, editing.game_profile
        )
        action_sc = segment_action_score(
            analysis,
            tightened.start,
            tightened.end,
            require_combat_for_apex=(editing.game_profile.lower() == "apex"),
        )
        if window_score < editing.min_moment_score - 1.0:
            continue
        if action_sc < editing.min_moment_score - 2.0:
            continue
        segments.append(tightened)

    if not segments:
        return build_catchy_plan(analysis, config)

    segments.sort(key=lambda s: s.start)

    hook = plan.hook_text.strip()
    if (
        not hook
        or WEAK_HOOK.match(hook)
        or WEAK_HOOK_PHRASES.search(hook)
        or SETUP_WORDS.search(hook)
    ):
        hook = build_hook_text(analysis, game_profile=editing.game_profile)

    effects = plan.effects or _collect_zooms(analysis, segments, config)
    hashtags = plan.hashtags or _default_hashtags(analysis)

    return EditPlan(
        target_duration_sec=sum(s.end - s.start for s in segments),
        segments=segments,
        effects=effects,
        hook_text=hook[:60],
        hashtags=hashtags[:6],
        caption_style=plan.caption_style or editing.caption_style,
    )


def _collect_zooms(
    analysis: ClipAnalysis,
    segments: list[EditSegment],
    config: AppConfig,
) -> list[EditEffect]:
    effects: list[EditEffect] = []
    ranges = [(s.start, s.end) for s in segments]
    zoom_count = 0

    ranked_peaks = sorted(analysis.loud_peaks, key=lambda p: p.score, reverse=True)
    for peak in ranked_peaks:
        if zoom_count >= config.editing.max_zoom_effects:
            break
        if any(start <= peak.time <= end for start, end in ranges):
            effects.append(
                EditEffect(
                    time=peak.time,
                    effect_type="zoom",
                    scale=config.editing.zoom_scale,
                    duration=config.editing.zoom_duration_sec,
                )
            )
            zoom_count += 1
    return effects


def _default_hashtags(analysis: ClipAnalysis) -> list[str]:
    tags = ["twitch", "gaming", "fyp", "clips"]
    if analysis.game_name:
        tag = re.sub(r"[^a-z0-9]", "", analysis.game_name.lower())
        if tag:
            tags.insert(0, tag)
    name = f"{analysis.game_name} {analysis.clip_title}".lower()
    if "apex" in name and "apexlegends" not in tags:
        tags.insert(0, "apexlegends")
    return tags[:6]
