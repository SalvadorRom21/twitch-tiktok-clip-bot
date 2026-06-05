"""Rule-based edit planning (no LLM required)."""

from __future__ import annotations

import re

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditEffect, EditPlan, EditSegment


EXCITING_WORDS = re.compile(
    r"\b(omg|no way|what|bro|insane|crazy|clutch|let's go|holy|wtf|damn|yes|win|died)\b",
    re.IGNORECASE,
)


def _score_window(analysis: ClipAnalysis, start: float, end: float) -> float:
    score = 0.0
    for peak in analysis.loud_peaks:
        if start <= peak.time <= end:
            score += peak.score * 2.0
    for seg in analysis.transcript_segments:
        if seg.end < start or seg.start > end:
            continue
        if EXCITING_WORDS.search(seg.text):
            score += 3.0
        score += min(len(seg.text) / 40.0, 1.0)
    if analysis.vision_summary and any(
        word in analysis.vision_summary.lower()
        for word in ("clutch", "win", "kill", "react", "scream")
    ):
        score += 1.5
    return score


def _pick_highlight_window(analysis: ClipAnalysis, target: float) -> tuple[float, float]:
    duration = analysis.duration
    if duration <= target:
        return 0.0, duration

    step = 1.0
    best_start = 0.0
    best_score = -1.0
    for start in [i * step for i in range(int(max(duration - target, 0) / step) + 1)]:
        end = min(start + target, duration)
        actual_len = end - start
        if actual_len < target * 0.6:
            continue
        score = _score_window(analysis, start, end)
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, min(best_start + target, duration)


def _overlaps(a: EditSegment, b: EditSegment, gap: float = 2.0) -> bool:
    return not (a.end + gap <= b.start or b.end + gap <= a.start)


def _window_around_time(
    time: float,
    duration: float,
    min_len: float,
    max_len: float,
) -> EditSegment:
    half = max_len / 2
    start = max(0.0, time - half * 0.6)
    end = min(duration, start + max_len)
    if end - start < min_len:
        start = max(0.0, end - min_len)
    return EditSegment(start=start, end=end, reason="highlight moment")


def _pick_montage_segments(analysis: ClipAnalysis, config: AppConfig) -> list[EditSegment]:
    editing = config.editing
    duration = analysis.duration
    if duration <= editing.min_duration_sec:
        return [EditSegment(start=0.0, end=duration, reason="full clip")]

    candidates: list[tuple[float, EditSegment]] = []

    for peak in analysis.loud_peaks:
        seg = _window_around_time(
            peak.time,
            duration,
            editing.min_segment_sec,
            editing.max_segment_sec,
        )
        seg.reason = "reaction peak"
        candidates.append((_score_window(analysis, seg.start, seg.end), seg))

    for transcript in analysis.transcript_segments:
        if not EXCITING_WORDS.search(transcript.text):
            continue
        mid = (transcript.start + transcript.end) / 2
        seg = _window_around_time(
            mid,
            duration,
            editing.min_segment_sec,
            editing.max_segment_sec,
        )
        seg.reason = "exciting line"
        candidates.append((_score_window(analysis, seg.start, seg.end) + 1.0, seg))

    if not candidates:
        start, end = _pick_highlight_window(analysis, editing.target_duration_sec)
        return [EditSegment(start=start, end=end, reason="highlight window")]

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[EditSegment] = []
    total = 0.0
    target = min(editing.target_duration_sec, editing.max_duration_sec)

    for _, seg in candidates:
        if len(selected) >= editing.max_montage_segments:
            break
        seg_len = seg.end - seg.start
        if seg_len < editing.min_segment_sec:
            continue
        if any(_overlaps(seg, existing) for existing in selected):
            continue
        if total + seg_len > target + 2.0:
            continue
        selected.append(seg)
        total += seg_len

    if not selected:
        start, end = _pick_highlight_window(analysis, target)
        return [EditSegment(start=start, end=end, reason="highlight window")]

    selected.sort(key=lambda s: s.start)
    return selected


def _build_hook_text(analysis: ClipAnalysis) -> str:
    if analysis.clip_title:
        title = analysis.clip_title.strip()
        if len(title) <= 60:
            return title
        return title[:57] + "..."
    for seg in analysis.transcript_segments:
        if EXCITING_WORDS.search(seg.text):
            return seg.text[:60]
    if analysis.transcript_segments:
        return analysis.transcript_segments[0].text[:60]
    return "Wait for it..."


def _build_hashtags(analysis: ClipAnalysis) -> list[str]:
    tags = ["twitch", "gaming", "clips"]
    if analysis.game_name:
        tag = re.sub(r"[^a-z0-9]", "", analysis.game_name.lower())
        if tag:
            tags.insert(0, tag)
    return tags[:6]


def _collect_effects(
    analysis: ClipAnalysis,
    segments: list[EditSegment],
    config: AppConfig,
) -> list[EditEffect]:
    effects: list[EditEffect] = []
    zoom_count = 0
    seg_ranges = [(s.start, s.end) for s in segments]

    for peak in analysis.loud_peaks:
        if zoom_count >= config.editing.max_zoom_effects:
            break
        if any(start <= peak.time <= end for start, end in seg_ranges):
            effects.append(
                EditEffect(
                    time=peak.time,
                    effect_type="zoom",
                    scale=1.12,
                    duration=0.7,
                )
            )
            zoom_count += 1

    for transcript in analysis.transcript_segments:
        if not any(start <= transcript.start <= end for start, end in seg_ranges):
            continue
        match = EXCITING_WORDS.search(transcript.text)
        if match:
            effects.append(
                EditEffect(
                    time=transcript.start,
                    effect_type="caption_emphasis",
                    text=match.group(0).upper(),
                    duration=1.0,
                )
            )
            break

    return effects


def create_rule_based_plan(analysis: ClipAnalysis, config: AppConfig) -> EditPlan:
    editing = config.editing

    if editing.montage_enabled and analysis.duration > editing.min_duration_sec + 5:
        segments = _pick_montage_segments(analysis, config)
    else:
        target = min(editing.target_duration_sec, editing.max_duration_sec)
        start, end = _pick_highlight_window(analysis, target)
        segments = [EditSegment(start=start, end=end, reason="highlight window")]

    total_duration = sum(s.end - s.start for s in segments)
    effects = _collect_effects(analysis, segments, config)

    return EditPlan(
        target_duration_sec=total_duration,
        segments=segments,
        effects=effects,
        hook_text=_build_hook_text(analysis),
        hashtags=_build_hashtags(analysis),
        caption_style=editing.caption_style,
    )
