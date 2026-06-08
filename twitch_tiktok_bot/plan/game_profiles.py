"""Game-specific highlight scoring (Apex Legends, etc.)."""

from __future__ import annotations

import re
from typing import TypedDict

from twitch_tiktok_bot.models import ClipAnalysis, LoudPeak
from twitch_tiktok_bot.plan.action import (
    expand_gunfight_window,
    fight_arc_quality,
    peak_cluster_is_gunfire,
    transcript_combat_score,
    transcript_setup_ratio,
    transcript_text_in_range,
)


class GameProfileOption(TypedDict):
    id: str
    label: str
    description: str


GAME_PROFILE_OPTIONS: list[GameProfileOption] = [
    {
        "id": "auto",
        "label": "Auto-detect",
        "description": "Infer game from Twitch title or metadata",
    },
    {
        "id": "apex",
        "label": "Apex Legends",
        "description": "Gunfights, knocks, cracks, wipes, clutches",
    },
    {
        "id": "smash",
        "label": "Super Smash Bros.",
        "description": "Sets (Bo3/Bo5), stocks, KOs, and game boundaries",
    },
    {
        "id": "generic",
        "label": "Generic gaming",
        "description": "Reactions and loud moments (no game-specific rules)",
    },
]

VALID_GAME_PROFILES = {opt["id"] for opt in GAME_PROFILE_OPTIONS}


def normalize_game_profile(profile: str | None) -> str:
    value = (profile or "auto").strip().lower()
    return value if value in VALID_GAME_PROFILES else "auto"


APEX_NAME = re.compile(r"apex", re.IGNORECASE)

APEX_PAYOFF = re.compile(
    r"\b("
    r"knock|knocked|crack|cracked|broken|broke|shield|armor|"
    r"wipe|wiped|squad|team|clutch|one v|1v\d|solo|duo|"
    r"beam|beamed|melt|melted|deleted|one shot|oneshot|"
    r"kraber|sentinel|wingman|peacekeeper|pk|"
    r"champion|win|won|let'?s go|nice|insane|crazy|no way|"
    r"push|pushed|third|party|rez|res|revive|"
    r"kill leader|cracked them|got him|got em|down|finished"
    r")\b",
    re.IGNORECASE,
)

APEX_WEAK = re.compile(
    r"\b(lobby|menu|settings|how do|ranked points|loading|queue|requeue|exit|"
    r"crafting|inventory|buy station|survey|wait|edit|let me do|firing range|"
    r"not too bad|wasn't too bad)\b",
    re.IGNORECASE,
)

APEX_HOOKS = [
    (re.compile(r"wipe|wiped|squad", re.I), "Squad wipe incoming"),
    (re.compile(r"clutch|1v|one v", re.I), "1v3 and he does THIS"),
    (re.compile(r"crack|cracked|broken", re.I), "Cracked — then this happened"),
    (re.compile(r"kraber|one shot|deleted", re.I), "One shot. That's it."),
    (re.compile(r"champion|win|won", re.I), "Champion squad."),
    (re.compile(r"beam|melt", re.I), "He got BEAMED"),
]


SMASH_NAME = re.compile(
    r"smash|ssbu|super smash|ultimate|melee|mang\d|genesis|evo\b",
    re.IGNORECASE,
)

SMASH_PAYOFF = re.compile(
    r"\b("
    r"stock|stocks|ko|k\.o|kill|combo|zero to death|ztd|"
    r"game[!]?|set|counterpick|counter pick|"
    r"oh my|no way|let'?s go|insane|crazy|clutch|"
    r"reverse|edgeguard|spike|finish"
    r")\b",
    re.IGNORECASE,
)

SMASH_HOOKS = [
    (re.compile(r"zero to death|ztd", re.I), "Zero to death."),
    (re.compile(r"counterpick|counter pick", re.I), "The counterpick worked."),
    (re.compile(r"clutch|reverse", re.I), "Clutch comeback."),
    (re.compile(r"stock|ko|k\.o", re.I), "Stock taken."),
]


def detect_game_profile(analysis: ClipAnalysis, profile: str = "auto") -> str:
    if profile and profile.lower() not in ("auto", ""):
        return profile.lower()
    name = f"{analysis.game_name} {analysis.clip_title}".lower()
    if SMASH_NAME.search(name):
        return "smash"
    if APEX_NAME.search(name) or "apex" in name:
        return "apex"
    return "generic"


def apex_fight_windows(
    analysis: ClipAnalysis,
    window_sec: float = 14.0,
    min_peaks: int = 4,
) -> list[tuple[float, float, float]]:
    """Return (start, end, score) for gunfight-style peak clusters."""
    if not analysis.loud_peaks:
        return []

    peaks = sorted(analysis.loud_peaks, key=lambda p: p.time)
    duration = analysis.duration
    clusters: list[tuple[float, float, float]] = []
    i = 0

    while i < len(peaks):
        cluster: list[LoudPeak] = [peaks[i]]
        j = i + 1
        while j < len(peaks) and peaks[j].time - cluster[0].time <= window_sec:
            cluster.append(peaks[j])
            j += 1

        if len(cluster) >= min_peaks and peak_cluster_is_gunfire(cluster):
            start = max(0.0, cluster[0].time - 6.0)
            end = min(duration, cluster[-1].time + 6.0)
            start, end = expand_gunfight_window(
                analysis,
                start,
                end,
                max_duration=24.0,
                max_prelude=22.0,
            )
            intensity = sum(p.score for p in cluster)
            density = len(cluster) / max(end - start, 1.0)
            score = intensity * 2.0 + density * 8.0

            text = transcript_text_in_range(analysis, start, end)
            score += transcript_combat_score(text)
            score += fight_arc_quality(analysis, start, end)
            score -= transcript_setup_ratio(analysis, start, end) * 10.0

            if score < 4.0:
                i = j if j > i + 1 else i + 1
                continue

            clusters.append((start, end, score))

        i = j if j > i + 1 else i + 1

    clusters.sort(key=lambda item: item[2], reverse=True)
    return clusters


def game_score_bonus(analysis: ClipAnalysis, start: float, end: float, profile: str) -> float:
    if profile == "smash":
        bonus = 0.0
        for seg in analysis.transcript_segments:
            if seg.end < start or seg.start > end:
                continue
            if SMASH_PAYOFF.search(seg.text):
                bonus += 3.0
        return bonus

    if profile != "apex":
        return 0.0

    bonus = 0.0
    for seg in analysis.transcript_segments:
        if seg.end < start or seg.start > end:
            continue
        if APEX_PAYOFF.search(seg.text):
            bonus += 4.0
        if APEX_WEAK.search(seg.text):
            bonus -= 3.0

    for start_w, end_w, fight_score in apex_fight_windows(analysis)[:5]:
        overlap = max(0.0, min(end, end_w) - max(start, start_w))
        if overlap > 1.0:
            bonus += fight_score * 0.5

    return bonus


def game_hook_text(analysis: ClipAnalysis, profile: str, quote: str = "") -> str | None:
    if profile == "smash":
        text = f"{quote} {analysis.clip_title}".lower()
        for pattern, hook in SMASH_HOOKS:
            if pattern.search(text):
                return hook
        if SMASH_PAYOFF.search(text):
            return "Smash moment you missed"
        return None

    if profile != "apex":
        return None

    text = f"{quote} {analysis.clip_title}".lower()
    for pattern, hook in APEX_HOOKS:
        if pattern.search(text):
            return hook

    if APEX_PAYOFF.search(text):
        return "Apex moment you missed"
    return None
