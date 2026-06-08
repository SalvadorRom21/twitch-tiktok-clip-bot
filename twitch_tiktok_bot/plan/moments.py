"""Score and rank highlight moments for TikTok-style edits."""

from __future__ import annotations

import re
from dataclasses import dataclass

from twitch_tiktok_bot.models import ClipAnalysis, EditSegment, TranscriptSegment
from twitch_tiktok_bot.plan.action import segment_action_score
from twitch_tiktok_bot.plan.game_profiles import (
    apex_fight_windows,
    detect_game_profile,
    game_hook_text,
    game_score_bonus,
)

PAYOFF_WORDS = re.compile(
    r"\b("
    r"nice|let'?s go|yes|omg|no way|insane|crazy|clutch|holy|wtf|damn|"
    r"got him|got em|sheesh|bro|what|how|dude|yooo|let'?s gooo|"
    r"win|won|died|rip|actually|finally|there we go|here we go"
    r")\b",
    re.IGNORECASE,
)

SETUP_WORDS = re.compile(
    r"\b(how do i|how does|let me do|let me|wait|hold on|one sec|testing|edit|"
    r"settings|menu|loading|chat|requeue|exit|firing range|not too bad)\b",
    re.IGNORECASE,
)

WEAK_HOOK = re.compile(
    r"^(how do|how does|let me|okay|ok|testing|wait|um|uh)\b",
    re.IGNORECASE,
)


@dataclass
class RankedMoment:
    start: float
    end: float
    score: float
    reason: str
    quote: str = ""


def _peak_score_at(analysis: ClipAnalysis, time: float) -> float:
    best = 0.0
    for peak in analysis.loud_peaks:
        dist = abs(peak.time - time)
        if dist <= 2.0:
            best = max(best, peak.score * max(0.2, 1.0 - dist / 2.0))
    return best


def _transcript_at(
    analysis: ClipAnalysis, time: float
) -> TranscriptSegment | None:
    for seg in analysis.transcript_segments:
        if seg.start <= time <= seg.end:
            return seg
        if seg.start - 0.5 <= time <= seg.end + 0.5:
            return seg
    return None


def score_time(
    analysis: ClipAnalysis, time: float, game_profile: str = "generic"
) -> float:
    """Higher = better TikTok moment."""
    duration = max(analysis.duration, 0.1)
    score = _peak_score_at(analysis, time) * 3.0

    transcript = _transcript_at(analysis, time)
    if transcript:
        text = transcript.text.strip()
        if PAYOFF_WORDS.search(text):
            score += 4.0
        if SETUP_WORDS.search(text):
            score -= 3.0
        if len(text) <= 40:
            score += 0.5

    # Favor later payoff over early setup (last 70% of clip)
    if time > duration * 0.35:
        score += 1.5
    if time < duration * 0.15:
        score -= 2.0

    return score + game_score_bonus(analysis, max(0.0, time - 1), time + 1, game_profile)


def score_window(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    game_profile: str = "generic",
) -> float:
    mid = (start + end) / 2
    base = score_time(analysis, mid, game_profile)
    for peak in analysis.loud_peaks:
        if start <= peak.time <= end:
            base += peak.score
    for seg in analysis.transcript_segments:
        if seg.end < start or seg.start > end:
            continue
        if PAYOFF_WORDS.search(seg.text):
            base += 2.0
        if SETUP_WORDS.search(seg.text):
            base -= 1.5
    action = segment_action_score(
        analysis,
        start,
        end,
        require_combat_for_apex=(game_profile == "apex"),
    )
    return base + game_score_bonus(analysis, start, end, game_profile) + action * 0.6


def rank_moments(
    analysis: ClipAnalysis,
    min_len: float = 2.5,
    max_len: float = 7.0,
    limit: int = 12,
    game_profile: str = "auto",
) -> list[RankedMoment]:
    duration = analysis.duration
    candidates: list[RankedMoment] = []
    profile = detect_game_profile(analysis, game_profile)

    anchor_times: list[tuple[float, str, str]] = []

    if profile == "apex":
        for start, end, fight_score in apex_fight_windows(analysis)[:6]:
            mid = (start + end) / 2
            anchor_times.append((mid, "gunfight", f"fight score {fight_score:.0f}"))

    for peak in analysis.loud_peaks:
        anchor_times.append((peak.time, "reaction peak", ""))
    for seg in analysis.transcript_segments:
        if PAYOFF_WORDS.search(seg.text):
            mid = (seg.start + seg.end) / 2
            anchor_times.append((mid, "payoff line", seg.text.strip()))

    seen: set[int] = set()
    for time, reason, quote in anchor_times:
        bucket = int(time * 2)
        if bucket in seen:
            continue
        seen.add(bucket)

        half = max_len * 0.45
        start = max(0.0, time - half * 0.4)
        end = min(duration, start + max_len)
        if end - start < min_len:
            start = max(0.0, end - min_len)

        seg = EditSegment(start=start, end=end, reason=reason)
        seg = tighten_segment(seg, analysis, min_len, max_len)
        sc = score_window(analysis, seg.start, seg.end, profile)
        if profile == "apex" and sc < 3.0:
            continue
        if not quote:
            t = _transcript_at(analysis, time)
            quote = t.text.strip() if t else ""
        candidates.append(
            RankedMoment(
                start=seg.start,
                end=seg.end,
                score=sc,
                reason=reason,
                quote=quote,
            )
        )

    if not candidates and duration > 0:
        candidates.append(
            RankedMoment(
                start=0.0,
                end=min(duration, max_len),
                score=1.0,
                reason="full clip",
            )
        )

    candidates.sort(key=lambda m: m.score, reverse=True)
    return candidates[:limit]


def tighten_segment(
    seg: EditSegment,
    analysis: ClipAnalysis,
    min_len: float,
    max_len: float,
) -> EditSegment:
    """Trim dead air at segment edges using silence ranges."""
    start, end = seg.start, seg.end
    for silent in analysis.silence_ranges:
        if silent.end <= start + 0.3 or silent.start >= end - 0.3:
            continue
        if silent.start <= start + 0.5 and silent.end > start:
            start = min(silent.end, end - min_len)
        if silent.end >= end - 0.5 and silent.start < end:
            end = max(silent.start, start + min_len)

    if max_len > 0 and end - start > max_len:
        if seg.reason in ("gunfight", "labeled_fight"):
            start = max(seg.start, end - max_len)
        else:
            end = start + max_len
    if end - start < min_len:
        end = min(analysis.duration, start + min_len)
    return EditSegment(start=start, end=end, reason=seg.reason)


def build_hook_text(
    analysis: ClipAnalysis,
    best_quote: str = "",
    game_profile: str = "auto",
) -> str:
    profile = detect_game_profile(analysis, game_profile)
    apex_hook = game_hook_text(analysis, profile, best_quote)
    if apex_hook:
        return apex_hook

    if analysis.clip_title:
        title = analysis.clip_title.strip()
        title = re.sub(r"\s*\|\s*", " — ", title)
        if len(title) <= 55 and not WEAK_HOOK.match(title):
            return title

    quote = (best_quote or "").strip()
    if not quote:
        for moment in rank_moments(analysis, limit=3):
            if moment.quote and not SETUP_WORDS.search(moment.quote):
                quote = moment.quote
                break

    lower = quote.lower()
    if "nice" in lower or "not too bad" in lower:
        return "He actually hit it"
    if "here we go" in lower or "let's go" in lower:
        return "Watch what happens next"
    if "no way" in lower or "what" in lower:
        return "You won't believe this"
    if "clutch" in lower or "win" in lower:
        return "Clutch or throw?"
    if "died" in lower or "rip" in lower:
        return "He thought he had it"
    if quote and not WEAK_HOOK.match(quote) and len(quote) <= 45:
        if not SETUP_WORDS.search(quote):
            return quote[:45]

    return "Wait for it..."
