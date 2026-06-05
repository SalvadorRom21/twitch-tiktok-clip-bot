"""Find multiple TikTok shorts inside a full VOD."""

from __future__ import annotations

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import (
    ClipAnalysis,
    EditPlan,
    EditSegment,
    LoudPeak,
    TimeRange,
    TranscriptSegment,
)
from twitch_tiktok_bot.plan.rules import (
    EXCITING_WORDS,
    _build_hashtags,
    _collect_effects,
    _overlaps,
    _pick_montage_segments,
    _score_window,
)


def _hook_for_window(analysis: ClipAnalysis, start: float, end: float) -> str:
    for seg in analysis.transcript_segments:
        if seg.start < start or seg.start > end:
            continue
        if EXCITING_WORDS.search(seg.text):
            return seg.text[:60]
        return seg.text[:60]
    if analysis.clip_title:
        return analysis.clip_title[:60]
    return f"Stream highlight @ {int(start // 60)}m"


def pick_vod_highlight_windows(
    analysis: ClipAnalysis,
    config: AppConfig,
    max_shorts: int | None = None,
) -> list[tuple[float, float]]:
    """Return non-overlapping (start, end) windows for separate TikTok shorts."""
    vod = config.vod
    limit = max_shorts or vod.max_shorts_per_vod
    target = min(config.editing.target_duration_sec, config.editing.max_duration_sec)
    min_gap = vod.min_short_gap_sec
    half = target / 2

    candidates: list[tuple[float, float, float]] = []

    for peak in analysis.loud_peaks[: vod.peak_candidate_limit]:
        start = max(0.0, peak.time - half * 0.5)
        end = min(analysis.duration, start + target)
        if end - start < config.editing.min_duration_sec:
            start = max(0.0, end - config.editing.min_duration_sec)
        score = _score_window(analysis, start, end)
        candidates.append((score, start, end))

    for seg in analysis.transcript_segments:
        if not EXCITING_WORDS.search(seg.text):
            continue
        mid = (seg.start + seg.end) / 2
        start = max(0.0, mid - half * 0.5)
        end = min(analysis.duration, start + target)
        score = _score_window(analysis, start, end) + 2.0
        candidates.append((score, start, end))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, float]] = []

    for _, start, end in candidates:
        if len(selected) >= limit:
            break
        window = EditSegment(start=start, end=end)
        if any(
            _overlaps(window, EditSegment(start=s, end=e), gap=min_gap)
            for s, e in selected
        ):
            continue
        if end - start < config.editing.min_duration_sec:
            continue
        selected.append((start, end))

    if not selected and analysis.duration > 0:
        best_score = -1.0
        best_window = (0.0, min(target, analysis.duration))
        step = max(30.0, target / 2)
        for start in [i * step for i in range(int(analysis.duration / step) + 1)]:
            end = min(start + target, analysis.duration)
            if end - start < config.editing.min_duration_sec:
                continue
            score = _score_window(analysis, start, end)
            if score > best_score:
                best_score = score
                best_window = (start, end)
        selected = [best_window]

    selected.sort(key=lambda w: w[0])
    return selected


def _slice_analysis(
    analysis: ClipAnalysis, start: float, end: float
) -> ClipAnalysis:
    return ClipAnalysis(
        duration=end - start,
        transcript_segments=[
            TranscriptSegment(
                start=max(0.0, s.start - start),
                end=min(end - start, s.end - start),
                text=s.text,
            )
            for s in analysis.transcript_segments
            if s.end >= start and s.start <= end
        ],
        loud_peaks=[
            LoudPeak(time=p.time - start, score=p.score)
            for p in analysis.loud_peaks
            if start <= p.time <= end
        ],
        silence_ranges=[
            TimeRange(
                start=max(0.0, r.start - start),
                end=min(end - start, r.end - start),
            )
            for r in analysis.silence_ranges
            if r.end >= start and r.start <= end
        ],
        scene_changes=[],
        vision_frames=[],
        vision_summary=analysis.vision_summary,
        clip_title=analysis.clip_title,
        game_name=analysis.game_name,
        face_crop_center_x=analysis.face_crop_center_x,
    )


def create_vod_short_plans(
    analysis: ClipAnalysis,
    config: AppConfig,
    max_shorts: int | None = None,
) -> list[EditPlan]:
    windows = pick_vod_highlight_windows(analysis, config, max_shorts=max_shorts)
    plans: list[EditPlan] = []

    for idx, (start, end) in enumerate(windows):
        sliced = _slice_analysis(analysis, start, end)
        if config.editing.montage_enabled and (end - start) > config.editing.target_duration_sec:
            segments = _pick_montage_segments(sliced, config)
            segments = [
                EditSegment(
                    start=s.start + start,
                    end=s.end + start,
                    reason=s.reason,
                )
                for s in segments
            ]
        else:
            segments = [EditSegment(start=start, end=end, reason="vod highlight")]

        effects = _collect_effects(analysis, segments, config)
        hook = _hook_for_window(analysis, start, end)
        if analysis.clip_title and idx == 0:
            hook = analysis.clip_title[:60]

        plans.append(
            EditPlan(
                target_duration_sec=sum(s.end - s.start for s in segments),
                segments=segments,
                effects=effects,
                hook_text=hook,
                hashtags=_build_hashtags(analysis) + ["vod", "stream"],
                caption_style=config.editing.caption_style,
            )
        )

    return plans
