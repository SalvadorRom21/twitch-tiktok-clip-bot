"""Detect real combat/action vs setup talk for highlight editing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from twitch_tiktok_bot.models import ClipAnalysis, LoudPeak

# Real in-game action language (not menu/testing chatter).
COMBAT_WORDS = re.compile(
    r"\b("
    r"knock|knocked|crack|cracked|broken|broke|shield|armor|"
    r"wipe|wiped|squad|clutch|1v\d|one v|solo|duo|"
    r"beam|beamed|melt|melted|deleted|one shot|oneshot|"
    r"kraber|peacekeeper|wingman|sentinel|"
    r"champion|kill leader|"
    r"got him|got em|got them|down|finished|killed|kill|"
    r"third part|push|pushed|rez|res|revive|"
    r"cracked them|shield break"
    r")\b",
    re.IGNORECASE,
)

# Menu, testing, setup — not TikTok highlight material.
SETUP_MENU_WORDS = re.compile(
    r"\b("
    r"how do i|how does|how do|let me do|let me|edit this|edit|"
    r"testing|test|menu|lobby|settings|loading|queue|requeue|exit|"
    r"ranked points|firing range|damage range|survey|wait|hold on|"
    r"one sec|chat|inventory|crafting|buy station|"
    r"wasn't too bad|not too bad|okay|ok\b"
    r")\b",
    re.IGNORECASE,
)

# Weak reactions without combat context.
WEAK_REACTION_WORDS = re.compile(
    r"\b(nice|here we go|let's go)\b",
    re.IGNORECASE,
)


@dataclass
class ActionSummary:
    has_combat: bool
    combat_score: float
    setup_ratio: float
    warning: str
    best_window: tuple[float, float] | None
    best_window_score: float = 0.0


def _peaks_in_range(
    peaks: list[LoudPeak], start: float, end: float
) -> list[LoudPeak]:
    return [p for p in peaks if start <= p.time <= end]


def _avg_peak_gap(peaks: list[LoudPeak]) -> float:
    if len(peaks) < 2:
        return 999.0
    times = sorted(p.time for p in peaks)
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return sum(gaps) / len(gaps)


def peak_cluster_is_gunfire(peaks: list[LoudPeak]) -> bool:
    """
    Gunfights = many loud peaks close together (rapid fire).
    Talking = sparse peaks seconds apart.
    """
    if len(peaks) < 4:
        return False
    avg_gap = _avg_peak_gap(peaks)
    if avg_gap > 2.2:
        return False
    intensity = sum(p.score for p in peaks) / len(peaks)
    return intensity >= 1.05


def transcript_text_in_range(
    analysis: ClipAnalysis, start: float, end: float
) -> str:
    parts: list[str] = []
    for seg in analysis.transcript_segments:
        if seg.end < start or seg.start > end:
            continue
        parts.append(seg.text)
    return " ".join(parts).strip()


def transcript_combat_score(text: str) -> float:
    if not text:
        return 0.0
    score = 0.0
    if COMBAT_WORDS.search(text):
        score += 8.0
    if SETUP_MENU_WORDS.search(text):
        score -= 6.0
    if WEAK_REACTION_WORDS.search(text) and not COMBAT_WORDS.search(text):
        score -= 2.0
    return score


def transcript_setup_ratio(analysis: ClipAnalysis, start: float, end: float) -> float:
    text = transcript_text_in_range(analysis, start, end)
    if not text:
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    setup_hits = len(SETUP_MENU_WORDS.findall(text))
    return min(1.0, setup_hits / max(len(words) / 3, 1))


def segment_action_score(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    *,
    require_combat_for_apex: bool = False,
) -> float:
    """Higher = real highlight. Low/negative = boring talk/menu."""
    text = transcript_text_in_range(analysis, start, end)
    peaks = _peaks_in_range(analysis.loud_peaks, start, end)

    score = transcript_combat_score(text)
    setup_ratio = transcript_setup_ratio(analysis, start, end)
    score -= setup_ratio * 10.0

    if peaks:
        score += min(4.0, sum(p.score for p in peaks) / len(peaks))
        if peak_cluster_is_gunfire(peaks):
            score += 6.0
        elif len(peaks) <= 2:
            score -= 1.0

    if require_combat_for_apex:
        has_combat_words = bool(COMBAT_WORDS.search(text))
        has_gunfire = peak_cluster_is_gunfire(peaks)
        if not has_combat_words and not has_gunfire:
            score -= 8.0
        if setup_ratio > 0.35:
            score -= 5.0

    return score


def expand_gunfight_window(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    *,
    max_duration: float = 22.0,
    max_prelude: float = 20.0,
    min_duration: float = 10.0,
) -> tuple[float, float]:
    """
    Include the full fight arc: lead-up, knocks, payoff — not just the last shots.

    When the window is too long, trim post-fight chatter at the tail before
    cutting early action.
    """
    duration = analysis.duration
    peaks = sorted(analysis.loud_peaks, key=lambda p: p.time)
    new_start = start
    prelude_limit = max(0.0, start - max_prelude)

    for peak in reversed(peaks):
        if peak.time >= start - 0.5:
            break
        if peak.time < prelude_limit:
            break
        if peak.score >= 1.0:
            new_start = min(new_start, max(0.0, peak.time - 2.5))

    for seg in analysis.transcript_segments:
        if seg.end < prelude_limit or seg.start > end + 4.0:
            continue
        if seg.end < start - 1.0 and transcript_combat_score(seg.text) < 3.0:
            continue
        if transcript_combat_score(seg.text) >= 3.0 and seg.end >= prelude_limit:
            candidate = max(prelude_limit, seg.start - 2.0)
            if seg.end >= start - 8.0:
                new_start = min(new_start, candidate)

    new_end = end
    for seg in analysis.transcript_segments:
        if seg.start < start - 2.0 or seg.start > end + 10.0:
            continue
        if COMBAT_WORDS.search(seg.text):
            new_end = max(new_end, min(duration, seg.end + 1.5))

    if peaks:
        in_range = [p.time for p in peaks if start <= p.time <= end + 2.0]
        if in_range:
            cluster_end = max(in_range)
            new_end = max(new_end, min(duration, cluster_end + 4.0))

    new_end = min(duration, max(new_end, new_start + min_duration))

    # Keep payoff at the end; slide start back to fit (lead-up > tail-only clips).
    if max_duration > 0 and new_end - new_start > max_duration:
        new_start = max(prelude_limit, new_end - max_duration)

    return new_start, new_end


def fight_arc_quality(
    analysis: ClipAnalysis, start: float, end: float
) -> float:
    """Higher when the window shows a full fight arc, not just the last shots."""
    length = max(end - start, 0.5)
    peaks = _peaks_in_range(analysis.loud_peaks, start, end)
    score = 0.0

    if peaks:
        first_rel = (peaks[0].time - start) / length
        late_share = sum(
            1 for p in peaks if (p.time - start) / length > 0.72
        ) / len(peaks)
        if first_rel < 0.4:
            score += 5.0
        if late_share > 0.75:
            score -= 10.0
        if peak_cluster_is_gunfire(peaks):
            mid_peaks = [
                p for p in peaks if 0.2 <= (p.time - start) / length <= 0.85
            ]
            if len(mid_peaks) >= 3:
                score += 4.0

    first_half = transcript_text_in_range(analysis, start, start + length * 0.55)
    second_half = transcript_text_in_range(analysis, start + length * 0.45, end)
    if COMBAT_WORDS.search(first_half):
        score += 6.0
    if COMBAT_WORDS.search(second_half):
        score += 2.0
    if SETUP_MENU_WORDS.search(second_half) and not COMBAT_WORDS.search(second_half):
        score -= 4.0

    return score


def find_best_action_window(
    analysis: ClipAnalysis,
    window_sec: float = 12.0,
) -> tuple[float, float, float] | None:
    """Slide a window across the clip; return best (start, end, score)."""
    duration = analysis.duration
    if duration < 3.0:
        return None

    best: tuple[float, float, float] | None = None
    step = 1.0
    t = 0.0
    while t < duration - 2.0:
        end = min(duration, t + window_sec)
        sc = segment_action_score(analysis, t, end, require_combat_for_apex=True)
        if best is None or sc > best[2]:
            best = (t, end, sc)
        t += step
    return best


def summarize_clip_action(
    analysis: ClipAnalysis, game_profile: str = "generic"
) -> ActionSummary:
    profile = (game_profile or "generic").lower()
    apex_mode = profile == "apex"

    full_text = transcript_text_in_range(analysis, 0, analysis.duration)
    setup_ratio = transcript_setup_ratio(analysis, 0, analysis.duration)
    combat_score = segment_action_score(
        analysis,
        0,
        analysis.duration,
        require_combat_for_apex=apex_mode,
    )

    best = find_best_action_window(analysis)
    best_window = (best[0], best[1]) if best else None
    best_window_score = best[2] if best else 0.0

    has_combat_words = bool(COMBAT_WORDS.search(full_text))
    has_gunfire = any(
        peak_cluster_is_gunfire(cluster)
        for cluster in _peak_clusters(analysis.loud_peaks, 10.0)
    )

    has_combat = (
        combat_score >= 6.0
        or (has_combat_words and best_window_score >= 4.0)
        or (has_gunfire and best_window_score >= 3.0)
    )

    warning = ""
    if apex_mode and not has_combat:
        warning = (
            "Low action clip — mostly menu/setup/talk, no knocks or gunfights detected. "
            "Use a clip with actual fights for better TikTok shorts."
        )
    elif combat_score < 3.0:
        warning = (
            "This clip is mostly talking, not gameplay highlights. "
            "Try a clip with fights, clutches, or big reactions during action."
        )

    return ActionSummary(
        has_combat=has_combat,
        combat_score=combat_score,
        setup_ratio=setup_ratio,
        warning=warning,
        best_window=best_window,
        best_window_score=best_window_score,
    )


def _peak_clusters(
    peaks: list[LoudPeak], window_sec: float
) -> list[list[LoudPeak]]:
    if not peaks:
        return []
    sorted_peaks = sorted(peaks, key=lambda p: p.time)
    clusters: list[list[LoudPeak]] = []
    i = 0
    while i < len(sorted_peaks):
        cluster = [sorted_peaks[i]]
        j = i + 1
        while j < len(sorted_peaks) and sorted_peaks[j].time - cluster[0].time <= window_sec:
            cluster.append(sorted_peaks[j])
            j += 1
        clusters.append(cluster)
        i = j if j > i + 1 else i + 1
    return clusters
