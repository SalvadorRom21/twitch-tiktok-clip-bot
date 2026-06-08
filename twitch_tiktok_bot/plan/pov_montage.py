"""Build montages by scanning the full POV and picking diverse highlight clips."""

from __future__ import annotations

from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.labels.curator import (
    apply_curator_to_gunfight_window,
    curator_gunfight_candidates,
    load_curator_references_for_work_dir,
)
from twitch_tiktok_bot.models import ClipAnalysis, EditSegment
from twitch_tiktok_bot.plan.action import segment_action_score, transcript_combat_score
from twitch_tiktok_bot.plan.duration import (
    max_clip_duration_sec,
    max_segment_duration_sec,
    target_clip_duration_sec,
)
from twitch_tiktok_bot.plan.apex_reference import (
    PRIMARY_MOMENT_REASONS,
    apex_fight_windows_for_shorts,
    apex_shorts_active,
    moment_source_score_boost,
    prefer_gunfight_anchor,
    skip_pov_sweep,
    try_single_fight_short,
)
from twitch_tiktok_bot.plan.game_profiles import apex_fight_windows, detect_game_profile
from twitch_tiktok_bot.plan.moments import (
    RankedMoment,
    score_window,
    tighten_segment,
)


def _moment_mid(moment: RankedMoment) -> float:
    return (moment.start + moment.end) / 2


def discover_pov_moments(
    analysis: ClipAnalysis,
    config: AppConfig,
    work_dir: Path | None = None,
) -> list[RankedMoment]:
    """
    Scan the entire source timeline for montage candidates.

    Unlike peak-only ranking, this sweeps the full POV so moments from early,
    middle, and late gameplay can all make the final mash-up.
    """
    editing = config.editing
    duration = max(analysis.duration, 0.1)
    min_len = editing.min_segment_sec
    max_len = max_segment_duration_sec(editing)
    profile = detect_game_profile(analysis, editing.game_profile)

    candidates: list[RankedMoment] = []
    seen: set[tuple[int, int]] = set()

    def add_candidate(start: float, end: float, reason: str, quote: str = "") -> None:
        start = max(0.0, start)
        end = min(duration, end)
        if end - start < min_len:
            end = min(duration, start + min_len)
        if end - start < min_len:
            return

        seg = EditSegment(start=start, end=end, reason=reason)
        seg = tighten_segment(seg, analysis, min_len, max_len)
        key = (int(seg.start * 2), int(seg.end * 2))
        if key in seen:
            return
        seen.add(key)

        score = score_window(analysis, seg.start, seg.end, profile)
        if profile == "apex" and reason == "gunfight":
            from twitch_tiktok_bot.plan.action import fight_arc_quality

            score += fight_arc_quality(analysis, seg.start, seg.end) * 2.5
        score *= moment_source_score_boost(reason, config)
        candidates.append(
            RankedMoment(
                start=seg.start,
                end=seg.end,
                score=score,
                reason=reason,
                quote=quote,
            )
        )

    shorts = apex_shorts_active(config) and profile == "apex"
    curator_refs = (
        load_curator_references_for_work_dir(work_dir) if work_dir else []
    )
    ref_max = max_clip_duration_sec(editing)
    ref_min = max(editing.min_segment_sec, 10.0)

    # 0) Curator-labeled fights — anchor clips at your push-in start, not audio tail.
    if curator_refs and profile == "apex":
        for start, end, fight_score, quote in curator_gunfight_candidates(
            analysis,
            curator_refs,
            max_duration=ref_max,
            min_duration=ref_min,
        ):
            add_candidate(start, end, "gunfight", quote[:80])

    # 1) Apex gunfights first — matches 11–21s wipe/duel Shorts (curator reference).
    if profile == "apex":
        fight_source = (
            apex_fight_windows_for_shorts(analysis)
            if shorts
            else apex_fight_windows(analysis)
        )
        for start, end, fight_score in fight_source:
            if curator_refs:
                start, end = apply_curator_to_gunfight_window(
                    analysis,
                    start,
                    end,
                    curator_refs,
                    max_duration=ref_max,
                    min_duration=ref_min,
                )
            add_candidate(start, end, "gunfight", f"fight score {fight_score:.0f}")

    # 2) Slide windows — disabled for apex_shorts (fills montage with non-fights).
    if not skip_pov_sweep(config, profile):
        seg_len = (min_len + max_len) / 2
        step = max(1.0, min_len * 0.45)
        t = 0.0
        while t < duration - min_len * 0.5:
            add_candidate(t, t + seg_len, "pov sweep")
            t += step

    # 3) Peak-anchored windows (lower priority than gunfights for apex_shorts).
    if not shorts:
        for peak in analysis.loud_peaks:
            half = max_len * 0.45
            add_candidate(
                peak.time - half * 0.35,
                peak.time + half * 0.65,
                "reaction peak",
            )

    # 4) Transcript payoff lines — only when combat language is present in apex_shorts.
    for seg in analysis.transcript_segments:
        if seg.end - seg.start < 0.2:
            continue
        if shorts:
            if transcript_combat_score(seg.text) < 4.0:
                continue
        mid = (seg.start + seg.end) / 2
        add_candidate(mid - max_len * 0.35, mid + max_len * 0.35, "payoff line", seg.text.strip())

    candidates.sort(key=lambda m: m.score, reverse=True)
    return candidates


def _cluster_candidates(
    ranked: list[RankedMoment], anchor_mid: float, radius_sec: float
) -> list[RankedMoment]:
    return [m for m in ranked if abs(_moment_mid(m) - anchor_mid) <= radius_sec]


def select_pov_montage_segments(
    candidates: list[RankedMoment],
    analysis: ClipAnalysis,
    config: AppConfig,
    work_dir: Path | None = None,
) -> list[EditSegment]:
    """Pick non-overlapping clips clustered around the best POV (smooth montage)."""
    editing = config.editing
    duration = max(analysis.duration, 0.1)
    max_segments = editing.max_montage_segments
    target = target_clip_duration_sec(editing)
    min_len = editing.min_segment_sec
    max_len = max_segment_duration_sec(editing)
    min_gap = max(1.0, min_len * 0.35)
    min_score = editing.min_moment_score
    apex_mode = editing.game_profile.lower() == "apex"
    cluster_sec = editing.montage_cluster_sec

    curator_refs = (
        load_curator_references_for_work_dir(work_dir) if work_dir else []
    )
    single = try_single_fight_short(
        candidates, analysis, config, curator_refs=curator_refs
    )
    if single:
        return single

    ranked_all = sorted(candidates, key=lambda m: m.score, reverse=True)
    if not ranked_all:
        return []

    gunfight_anchor = prefer_gunfight_anchor(ranked_all)
    anchor_mid = gunfight_anchor if gunfight_anchor is not None else _moment_mid(ranked_all[0])
    gunfights_available = any(m.reason == "gunfight" for m in ranked_all)
    ranked = _cluster_candidates(ranked_all, anchor_mid, cluster_sec)
    for expanded in (cluster_sec * 1.5, cluster_sec * 2.5, cluster_sec * 4, duration):
        if len(ranked) >= max_segments:
            break
        ranked = _cluster_candidates(ranked_all, anchor_mid, min(expanded, duration))
    ranked = sorted(ranked, key=lambda m: m.score, reverse=True)

    def to_segment(moment: RankedMoment) -> EditSegment:
        seg = EditSegment(start=moment.start, end=moment.end, reason=moment.reason)
        return tighten_segment(seg, analysis, min_len, max_len)

    def overlaps_existing(seg: EditSegment, selected: list[EditSegment]) -> bool:
        for existing in selected:
            if not (seg.end + min_gap <= existing.start or existing.end + min_gap <= seg.start):
                return True
        return False

    def passes_filters(
        seg: EditSegment, score: float, reason: str, *, relaxed: bool
    ) -> bool:
        threshold = min_score * (0.45 if relaxed else 1.0)
        if score < threshold:
            return False
        if seg.end - seg.start < min_len:
            return False
        if (
            apex_shorts_active(config)
            and gunfights_available
            and reason not in PRIMARY_MOMENT_REASONS
            and not relaxed
        ):
            return False
        action_sc = segment_action_score(
            analysis,
            seg.start,
            seg.end,
            require_combat_for_apex=apex_mode,
        )
        action_threshold = (min_score - 4.0) if relaxed else (min_score - 1.0)
        return action_sc >= action_threshold

    selected: list[EditSegment] = []
    total = 0.0

    # Fill montage slots from the local cluster (same fight / same game stretch).
    for moment in ranked:
        if len(selected) >= max_segments:
            break
        if total >= target + 0.5 and selected:
            break
        seg = to_segment(moment)
        if overlaps_existing(seg, selected):
            continue
        if not passes_filters(seg, moment.score, moment.reason, relaxed=False):
            continue
        length = seg.end - seg.start
        if total + length > target + 1.0 and selected:
            continue
        selected.append(seg)
        total += length

    # Pass 3: if still empty, gunfights only (never fall back to pov sweep for apex_shorts).
    if not selected:
        fallback = [m for m in ranked if m.reason == "gunfight"] if gunfights_available else ranked
        for moment in fallback[: max_segments * 3]:
            seg = to_segment(moment)
            if overlaps_existing(seg, selected):
                continue
            if seg.end - seg.start < min_len:
                continue
            if not passes_filters(seg, moment.score, moment.reason, relaxed=True):
                continue
            selected.append(seg)
            if len(selected) >= max_segments:
                break

    selected.sort(key=lambda s: s.start)
    return selected
