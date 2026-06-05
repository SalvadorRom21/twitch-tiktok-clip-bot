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


def _trim_silence_segments(
    segments: list[EditSegment], silence_threshold: float
) -> list[EditSegment]:
    if not segments:
        return segments
    # For MVP we keep one contiguous block; silence trimming happens at render time.
    return segments


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


def create_rule_based_plan(analysis: ClipAnalysis, config: AppConfig) -> EditPlan:
    editing = config.editing
    target = min(editing.target_duration_sec, editing.max_duration_sec)
    start, end = _pick_highlight_window(analysis, target)

    segments = [EditSegment(start=start, end=end, reason="highlight window")]
    segments = _trim_silence_segments(segments, editing.silence_threshold_sec)

    effects: list[EditEffect] = []
    zoom_count = 0
    for peak in analysis.loud_peaks:
        if zoom_count >= editing.max_zoom_effects:
            break
        if start <= peak.time <= end:
            effects.append(
                EditEffect(
                    time=peak.time,
                    effect_type="zoom",
                    scale=1.12,
                    duration=0.7,
                )
            )
            zoom_count += 1

    for seg in analysis.transcript_segments:
        if not (start <= seg.start <= end):
            continue
        match = EXCITING_WORDS.search(seg.text)
        if match:
            effects.append(
                EditEffect(
                    time=seg.start,
                    effect_type="caption_emphasis",
                    text=match.group(0).upper(),
                    duration=1.0,
                )
            )
            break

    return EditPlan(
        target_duration_sec=end - start,
        segments=segments,
        effects=effects,
        hook_text=_build_hook_text(analysis),
        hashtags=_build_hashtags(analysis),
        caption_style=editing.caption_style,
    )
