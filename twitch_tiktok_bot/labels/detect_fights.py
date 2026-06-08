"""Auto-detect Apex fight windows inside a labeled match."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan, EditSegment, LoudPeak
from twitch_tiktok_bot.plan.moments import build_hook_text
from twitch_tiktok_bot.plan.action import (
    COMBAT_WORDS,
    expand_gunfight_window,
    peak_cluster_is_gunfire,
    segment_action_score,
    transcript_combat_score,
    transcript_text_in_range,
    _avg_peak_gap,
    _peaks_in_range,
)
from twitch_tiktok_bot.plan.game_profiles import apex_fight_windows

CONTACT_CUE = re.compile(
    r"where are they|missed every shot|someone'?s here|someone'?s shooting|"
    r"push|contact|cracked|knock|down|he'?s one|he is one|got him|got em|"
    r"shield|beam|wipe|third|subject in view|too far away|they'?re here|"
    r"let'?s go.*where|i got him|i got it|scared|one shot",
    re.I,
)
STRONG_CUE = re.compile(
    r"where are they|missed every shot|someone'?s here|he'?s one|knock|"
    r"cracked|wipe|push|got him|fuck me|out of ammo",
    re.I,
)
DEATH_DEBRIEF_CUE = re.compile(
    r"fuck me|teammates left|out of ammo that whole|how do i not get any kills|"
    r"first game|decent first game|decent game",
    re.I,
)
LOOTING_CUE = re.compile(
    r"went all the way down|high ground that last|haven't lost any rp|"
    r"spread out here|think we're good",
    re.I,
)
PRE_TAIL_END_CUE = re.compile(r"stay up here|why don't we just stay", re.I)
TAIL_FIGHT_NOTE = "Match tail fight through elimination"


@dataclass
class DetectedFight:
    id: str
    start_sec: float
    end_sec: float
    source: str = "auto"
    score: float = 0.0
    confidence: str = "medium"
    start_cue: str = ""
    end_cue: str = ""
    notes: str = ""
    use_for_clips: bool = True
    created_at: str = ""

    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DetectedFight:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            source=str(data.get("source", "auto")),
            score=float(data.get("score", 0.0)),
            confidence=str(data.get("confidence", "medium")),
            start_cue=str(data.get("start_cue", "")),
            end_cue=str(data.get("end_cue", "")),
            notes=str(data.get("notes", "")),
            use_for_clips=bool(data.get("use_for_clips", True)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class DetectedFightStore:
    vod_id: str
    match_id: str = ""
    match_start_sec: float = 0.0
    match_end_sec: float = 0.0
    fights: list[DetectedFight] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "vod_id": self.vod_id,
            "match_id": self.match_id,
            "match_start_sec": self.match_start_sec,
            "match_end_sec": self.match_end_sec,
            "fights": [fight.to_dict() for fight in self.fights],
        }

    @classmethod
    def from_dict(cls, data: dict) -> DetectedFightStore:
        return cls(
            vod_id=str(data.get("vod_id", "")),
            match_id=str(data.get("match_id", "")),
            match_start_sec=float(data.get("match_start_sec", 0.0)),
            match_end_sec=float(data.get("match_end_sec", 0.0)),
            fights=[DetectedFight.from_dict(item) for item in data.get("fights", [])],
        )


def detected_fights_path(work_dir: Path) -> Path:
    return work_dir / "detected_fights.json"


def load_detected_fights(work_dir: Path) -> DetectedFightStore | None:
    path = detected_fights_path(work_dir)
    if not path.exists():
        return None
    try:
        return DetectedFightStore.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_detected_fights(work_dir: Path, store: DetectedFightStore) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = detected_fights_path(work_dir)
    path.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return path


def _window_action_score(analysis: ClipAnalysis, start: float, end: float) -> float:
    return segment_action_score(
        analysis, start, end, require_combat_for_apex=True
    )


def _peak_density(analysis: ClipAnalysis, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    count = sum(1 for p in analysis.loud_peaks if start <= p.time < end)
    return count / max(end - start, 1.0)


def _death_debrief_time(
    analysis: ClipAnalysis, match_start: float, match_end: float
) -> float | None:
    for seg in analysis.transcript_segments:
        if seg.start < match_start or seg.start > match_end:
            continue
        if DEATH_DEBRIEF_CUE.search(seg.text):
            return seg.start
    return None


def _expand_fight_arc(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    *,
    match_start: float,
    match_end: float,
    max_span: float = 200.0,
    step: float = 2.0,
    cold_gap: float = 5.0,
    transcript_led: bool = False,
    extend_to_match_end: bool = False,
) -> tuple[float, float]:
    cur_start = start
    while cur_start > match_start:
        probe = max(match_start, cur_start - step)
        threshold = 0.5 if transcript_led else 1.2
        if _window_action_score(analysis, probe, cur_start) < threshold:
            break
        cur_start = probe
        if start - cur_start > 50:
            break

    cur_end = end
    cold_since: float | None = None
    hard_end = match_end if extend_to_match_end else match_end
    death = _death_debrief_time(analysis, match_start, match_end)
    if extend_to_match_end and death is not None:
        hard_end = min(match_end, death + 2.0)

    while cur_end < hard_end:
        probe = min(hard_end, cur_end + step)
        score = _window_action_score(analysis, max(cur_start, probe - 4), probe)
        peaks = _peaks_in_range(analysis.loud_peaks, cur_end, probe)
        hot = (
            score >= (0.8 if transcript_led else 1.5)
            or (len(peaks) >= 2 and peak_cluster_is_gunfire(peaks))
            or (transcript_led and len(peaks) >= 1)
            or (extend_to_match_end and len(peaks) >= 1)
        )
        if hot:
            cold_since = None
            cur_end = probe
        else:
            if cold_since is None:
                cold_since = cur_end
            gap_needed = 8.0 if extend_to_match_end else (10.0 if transcript_led else cold_gap)
            if probe - cold_since >= gap_needed:
                break
            cur_end = probe
        if cur_end - start > max_span and not extend_to_match_end:
            cur_end = min(hard_end, start + max_span)
            break

    return max(match_start, cur_start - 2.0), min(hard_end, cur_end + 1.5)


def _audio_fight_seeds(
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
) -> list[tuple[float, float, float]]:
    seeds: list[tuple[float, float, float]] = []
    for start, end, score in apex_fight_windows(
        analysis, window_sec=18.0, min_peaks=4
    ):
        if end < match_start or start > match_end:
            continue
        start = max(match_start, start)
        end = min(match_end, end)
        es, ee = expand_gunfight_window(
            analysis,
            start,
            end,
            max_duration=0,
            max_prelude=45.0,
            min_duration=8.0,
        )
        seeds.append((max(match_start, es), min(match_end, ee), score))
    return seeds


def _sparse_peak_seeds(
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
    *,
    min_peaks: int = 2,
) -> list[tuple[float, float, float]]:
    """Quiet fights: sparse peaks with little comms (e.g. 18:33 → elimination)."""
    peaks = sorted(
        (p for p in analysis.loud_peaks if match_start <= p.time <= match_end),
        key=lambda p: p.time,
    )
    if len(peaks) < min_peaks:
        return []

    seeds: list[tuple[float, float, float]] = []
    window = 130.0
    step = 15.0
    t = match_start
    seen: set[tuple[int, int]] = set()

    while t < match_end - 35:
        bucket = [p for p in peaks if t <= p.time < t + window]
        if len(bucket) >= min_peaks:
            avg_gap = _avg_peak_gap(bucket)
            intensity = sum(p.score for p in bucket) / len(bucket)
            span = bucket[-1].time - bucket[0].time
            if span >= 12 and avg_gap <= 35 and intensity >= 1.2:
                key = (int(bucket[0].time // 25), int(bucket[-1].time // 25))
                if key not in seen:
                    seen.add(key)
                    score = len(bucket) * 3.0 + intensity * 4.0
                    seeds.append(
                        (
                            max(match_start, bucket[0].time - 15.0),
                            min(match_end, bucket[-1].time + 25.0),
                            score,
                        )
                    )
        t += step
    return seeds


def _transcript_cluster_seeds(
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
    *,
    cluster_gap_sec: float = 85.0,
) -> list[tuple[float, float, float, str]]:
    """Group combat transcript cues into engagement arcs (quiet fights with little gunfire)."""
    anchors: list[tuple[float, float, float, str]] = []
    for seg in analysis.transcript_segments:
        if seg.end < match_start or seg.start > match_end:
            continue
        text = seg.text.strip()
        if not text:
            continue
        if not (COMBAT_WORDS.search(text) or CONTACT_CUE.search(text)):
            continue
        strong = bool(STRONG_CUE.search(text))
        anchors.append(
            (seg.start, seg.end, 10.0 if strong else 6.0, text[:100])
        )
    if not anchors:
        return []

    anchors.sort(key=lambda item: item[0])
    clusters: list[tuple[float, float, float, str]] = []
    cur_start, cur_end, cur_score, cur_cue = anchors[0]

    for start, end, score, cue in anchors[1:]:
        if start - cur_end <= cluster_gap_sec:
            cur_end = max(cur_end, end)
            cur_score = max(cur_score, score)
            if STRONG_CUE.search(cue):
                cur_cue = cue
        else:
            clusters.append(
                (
                    max(match_start, cur_start - 18.0),
                    min(match_end, cur_end + 22.0),
                    cur_score,
                    cur_cue,
                )
            )
            cur_start, cur_end, cur_score, cur_cue = start, end, score, cue

    clusters.append(
        (
            max(match_start, cur_start - 18.0),
            min(match_end, cur_end + 22.0),
            cur_score,
            cur_cue,
        )
    )
    return clusters


def _transcript_fight_seeds(
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
) -> list[tuple[float, float, float, str]]:
    seeds: list[tuple[float, float, float, str]] = []
    for seg in analysis.transcript_segments:
        if seg.end < match_start or seg.start > match_end:
            continue
        text = seg.text.strip()
        if not text:
            continue
        if not (COMBAT_WORDS.search(text) or CONTACT_CUE.search(text)):
            continue
        strong = bool(STRONG_CUE.search(text))
        local_peaks = _peaks_in_range(
            analysis.loud_peaks,
            max(match_start, seg.start - 45),
            min(match_end, seg.end + 90),
        )
        if len(local_peaks) < 1 and not strong:
            continue
        start = max(match_start, seg.start - 10.0)
        end = min(match_end, seg.end + 15.0)
        seeds.append(
            (start, end, transcript_combat_score(text) + (6.0 if strong else 4.0), text[:100])
        )
    return seeds


def _tail_region_start(match_start: float, match_end: float) -> float:
    return match_start + (match_end - match_start) * 0.72


def _match_tail_fight_seed(
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
    existing: list[DetectedFight],
) -> tuple[float, float, float] | None:
    """Last fight through elimination — often sparse until death audio."""
    tail_start = _tail_region_start(match_start, match_end)

    peaks = [
        p for p in analysis.loud_peaks
        if tail_start <= p.time <= match_end - 10
    ]
    if len(peaks) < 1:
        return None

    first_peak = peaks[0].time
    death = _death_debrief_time(analysis, match_start, match_end)
    end = death if death is not None else match_end
    if end - first_peak < 20:
        return None

    if any(
        _overlap_ratio(f.start_sec, f.end_sec, first_peak - 15, end) > 0.35
        for f in existing
    ):
        return None

    return (max(match_start, first_peak - 18.0), end, 22.0)


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    shorter = max(0.1, min(a_end - a_start, b_end - b_start))
    return overlap / shorter


def _split_fights_at_cold_gaps(
    analysis: ClipAnalysis,
    fights: list[DetectedFight],
    *,
    gap_sec: float = 22.0,
) -> list[DetectedFight]:
    out: list[DetectedFight] = []
    for fight in fights:
        span = fight.end_sec - fight.start_sec
        if span < gap_sec * 1.35:
            out.append(fight)
            continue

        step = 3.0
        chunks: list[tuple[float, float]] = []
        chunk_start = fight.start_sec
        cold_start: float | None = None
        t = fight.start_sec

        while t < fight.end_sec - 5:
            w_end = min(fight.end_sec, t + step)
            density = _peak_density(analysis, t, w_end)
            score = _window_action_score(analysis, t, w_end)
            cold = density < 0.035 and score < 1.8
            if cold:
                if cold_start is None:
                    cold_start = t
            else:
                if cold_start is not None and t - cold_start >= gap_sec:
                    if cold_start - chunk_start >= 14:
                        chunks.append((chunk_start, cold_start))
                        chunk_start = t
                cold_start = None
            t += step

        if fight.end_sec - chunk_start >= 14:
            chunks.append((chunk_start, fight.end_sec))

        if len(chunks) <= 1:
            out.append(fight)
            continue

        for start, end in chunks:
            out.append(
                DetectedFight(
                    id=uuid.uuid4().hex[:10],
                    start_sec=round(start, 2),
                    end_sec=round(end, 2),
                    source=fight.source,
                    score=round(_window_action_score(analysis, start, end), 1),
                    confidence=fight.confidence,
                    start_cue=transcript_text_in_range(analysis, start, start + 12)[:80],
                    end_cue=transcript_text_in_range(analysis, max(start, end - 12), end)[:80],
                    notes=(fight.notes + " (split)").strip(),
                    created_at=fight.created_at,
                )
            )
    return out


def _is_audio_fight(fight: DetectedFight) -> bool:
    return "Audio gunfire" in fight.notes


def _is_tail_fight(fight: DetectedFight) -> bool:
    return fight.notes == TAIL_FIGHT_NOTE


def _collapse_overlapping_fights(
    fights: list[DetectedFight],
    *,
    overlap_threshold: float = 0.38,
) -> list[DetectedFight]:
    """Drop weaker detections that heavily overlap a stronger one."""
    if len(fights) <= 1:
        return fights

    ranked = sorted(
        fights,
        key=lambda f: (
            _is_audio_fight(f),
            f.score * max(f.duration_sec(), 25.0),
            _is_tail_fight(f),
            f.duration_sec(),
        ),
        reverse=True,
    )
    kept: list[DetectedFight] = []
    for fight in ranked:
        dominated = False
        for other in kept:
            if _is_audio_fight(other) and _is_audio_fight(fight):
                continue
            overlap = _overlap_ratio(
                fight.start_sec, fight.end_sec, other.start_sec, other.end_sec
            )
            if overlap < overlap_threshold:
                continue
            if _is_audio_fight(fight) and not _is_audio_fight(other):
                continue
            if STRONG_CUE.search(fight.start_cue) and not STRONG_CUE.search(other.start_cue):
                continue
            if _is_audio_fight(other) and not _is_audio_fight(fight):
                dominated = True
                break
            dominated = True
            break
        if not dominated:
            kept.append(fight)
    kept.sort(key=lambda f: f.start_sec)
    return kept


def _is_opening_fight(
    fight: DetectedFight, match_start: float, match_end: float
) -> bool:
    return fight.start_sec < match_start + (match_end - match_start) * 0.22


def _cap_fight_count(
    fights: list[DetectedFight],
    match_start: float,
    match_end: float,
    *,
    max_fights: int = 6,
    min_center_gap: float = 78.0,
) -> list[DetectedFight]:
    protected = [
        f
        for f in fights
        if _is_audio_fight(f)
        or _is_tail_fight(f)
        or _is_opening_fight(f, match_start, match_end)
        or STRONG_CUE.search(f.start_cue)
    ]
    optional = [f for f in fights if f not in protected]
    if len(protected) + len(optional) <= max_fights:
        fights.sort(key=lambda f: f.start_sec)
        return fights

    ranked = sorted(
        optional,
        key=lambda f: (f.score * max(f.duration_sec(), 20.0), f.duration_sec()),
        reverse=True,
    )
    kept = list(protected)
    for fight in ranked:
        mid = (fight.start_sec + fight.end_sec) / 2
        if any(
            abs((other.start_sec + other.end_sec) / 2 - mid) < min_center_gap
            for other in kept
        ):
            continue
        kept.append(fight)
        if len(kept) >= max_fights:
            break
    kept.sort(key=lambda f: f.start_sec)
    return kept


def _region_covered(
    fights: list[DetectedFight], center: float, *, radius: float = 75.0
) -> bool:
    return any(
        fight.start_sec - radius <= center <= fight.end_sec + radius
        for fight in fights
    )


def _fill_sparse_gap_seeds(
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
    existing: list[DetectedFight],
) -> list[tuple[float, float, float]]:
    """Sparse peaks for early/quiet fights not already covered by other seeds."""
    seeds: list[tuple[float, float, float]] = []
    for start, end, score in _sparse_peak_seeds(
        analysis, match_start, match_end, min_peaks=3
    ):
        mid = (start + end) / 2
        if _region_covered(existing, mid, radius=85.0):
            continue
        if mid >= _tail_region_start(match_start, match_end) - 30:
            continue
        seeds.append((start, end, score))
    return seeds


def _dedupe_transcript_echoes(fights: list[DetectedFight]) -> list[DetectedFight]:
    """Drop transcript-led duplicates that repeat the same cue in one engagement."""
    if len(fights) <= 1:
        return fights

    ranked = sorted(
        fights,
        key=lambda f: (f.duration_sec(), f.score, f.start_sec),
        reverse=True,
    )
    kept: list[DetectedFight] = []
    for fight in ranked:
        if _is_audio_fight(fight) or _is_tail_fight(fight):
            kept.append(fight)
            continue
        cue_key = fight.start_cue[:36].strip().lower()
        dominated = False
        for other in kept:
            if _is_tail_fight(other):
                continue
            other_key = other.start_cue[:36].strip().lower()
            same_cue = cue_key and cue_key == other_key
            overlap = _overlap_ratio(
                fight.start_sec, fight.end_sec, other.start_sec, other.end_sec
            )
            if same_cue and overlap >= 0.25:
                dominated = True
                break
            if (
                _is_audio_fight(other)
                and not _is_audio_fight(fight)
                and overlap >= 0.35
            ):
                dominated = True
                break
        if not dominated:
            kept.append(fight)
    kept.sort(key=lambda f: f.start_sec)
    return kept


def _drop_looting_followups(fights: list[DetectedFight]) -> list[DetectedFight]:
    """Drop post-fight looting / rotation that is not its own engagement."""
    kept: list[DetectedFight] = []
    for fight in fights:
        if _is_audio_fight(fight) or _is_tail_fight(fight):
            kept.append(fight)
            continue
        if LOOTING_CUE.search(fight.start_cue) and fight.score < 20:
            continue
        kept.append(fight)
    return kept


def _drop_pre_tail_rotation(
    fights: list[DetectedFight],
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
) -> list[DetectedFight]:
    """Remove quiet rotation before the final tail fight (not its own engagement)."""
    tail_start = _tail_region_start(match_start, match_end)
    late_cutoff = match_start + (match_end - match_start) * 0.68
    has_tail = any(_is_tail_fight(fight) for fight in fights)
    kept: list[DetectedFight] = []
    for fight in fights:
        if _is_tail_fight(fight) or _is_audio_fight(fight):
            kept.append(fight)
            continue
        if fight.start_sec >= tail_start - 30:
            continue
        if (
            has_tail
            and late_cutoff <= fight.start_sec
            and not _is_audio_fight(fight)
            and fight.score < 20
        ):
            continue
        if has_tail and PRE_TAIL_END_CUE.search(fight.end_cue):
            continue
        kept.append(fight)
    return kept


def _ensure_tail_fight(
    fights: list[DetectedFight],
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
    now: str,
) -> list[DetectedFight]:
    """Guarantee the last fight (e.g. 18:33 → elimination) is present."""
    fights = [f for f in fights if not _is_tail_fight(f)]
    seed = _match_tail_fight_seed(analysis, match_start, match_end, fights)
    if seed is None:
        tail_start = _tail_region_start(match_start, match_end)
        sparse = _sparse_peak_seeds(analysis, tail_start, match_end)
        if sparse:
            seed = sparse[0]
        else:
            return fights

    fight = _seed_to_fight(
        analysis,
        seed[0],
        seed[1],
        seed[2],
        match_start=match_start,
        match_end=match_end,
        now=now,
        notes=TAIL_FIGHT_NOTE,
        extend_to_match_end=True,
        min_duration=20.0,
    )
    if fight:
        fights = list(fights) + [fight]
        fights.sort(key=lambda f: f.start_sec)
    return fights


def _dedupe_overlapping_fights(fights: list[DetectedFight]) -> list[DetectedFight]:
    if len(fights) <= 1:
        return fights
    ranked = sorted(
        fights,
        key=lambda f: (
            _is_audio_fight(f),
            STRONG_CUE.search(f.start_cue) is not None,
            f.duration_sec(),
            f.score,
        ),
        reverse=True,
    )
    kept: list[DetectedFight] = []
    for fight in ranked:
        dominated = False
        for other in kept:
            overlap = _overlap_ratio(
                fight.start_sec, fight.end_sec, other.start_sec, other.end_sec
            )
            if overlap < 0.38:
                continue
            if _is_audio_fight(other) and not _is_audio_fight(fight):
                dominated = True
                break
            if STRONG_CUE.search(other.start_cue) and not STRONG_CUE.search(
                fight.start_cue
            ):
                dominated = True
                break
            if overlap > 0.72:
                dominated = True
                break
        if not dominated:
            kept.append(fight)
    kept.sort(key=lambda f: f.start_sec)
    return kept


def _merge_adjacent_fights(
    fights: list[DetectedFight],
    *,
    max_gap_sec: float = 38.0,
    max_combined_sec: float = 85.0,
) -> list[DetectedFight]:
    if len(fights) <= 1:
        return fights
    fights = sorted(fights, key=lambda f: f.start_sec)
    merged: list[DetectedFight] = [fights[0]]
    for fight in fights[1:]:
        prev = merged[-1]
        gap = fight.start_sec - prev.end_sec
        combined = max(prev.end_sec, fight.end_sec) - prev.start_sec
        same_engagement = (
            0 < gap <= max_gap_sec
            and combined <= max_combined_sec
            and not _is_tail_fight(prev)
            and not _is_tail_fight(fight)
        )
        if same_engagement:
            merged[-1] = DetectedFight(
                id=prev.id,
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, fight.end_sec),
                source=prev.source,
                score=max(prev.score, fight.score),
                confidence=prev.confidence,
                start_cue=prev.start_cue or fight.start_cue,
                end_cue=fight.end_cue or prev.end_cue,
                notes=(prev.notes + " + adjacent").strip(),
                use_for_clips=prev.use_for_clips,
                created_at=prev.created_at,
            )
        else:
            merged.append(fight)
    return merged


def _merge_fights(fights: list[DetectedFight]) -> list[DetectedFight]:
    if not fights:
        return fights
    fights = sorted(fights, key=lambda f: f.start_sec)
    merged: list[DetectedFight] = []
    for fight in fights:
        if not merged:
            merged.append(fight)
            continue
        prev = merged[-1]
        gap = fight.start_sec - prev.end_sec
        overlap = _overlap_ratio(
            prev.start_sec, prev.end_sec, fight.start_sec, fight.end_sec
        )
        if gap > 16 and overlap < 0.2:
            merged.append(fight)
            continue
        if overlap > 0.35:
            new_start = min(prev.start_sec, fight.start_sec)
            new_end = max(prev.end_sec, fight.end_sec)
            if new_end - new_start > 105:
                merged.append(fight)
                continue
            merged[-1] = DetectedFight(
                id=prev.id,
                start_sec=new_start,
                end_sec=new_end,
                source="auto",
                score=max(prev.score, fight.score),
                confidence="high" if max(prev.score, fight.score) > 18 else "medium",
                start_cue=prev.start_cue or fight.start_cue,
                end_cue=fight.end_cue or prev.end_cue,
                notes=prev.notes,
                use_for_clips=prev.use_for_clips,
                created_at=prev.created_at or fight.created_at,
            )
        else:
            merged.append(fight)
    return merged


def _seed_to_fight(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    score: float,
    *,
    match_start: float,
    match_end: float,
    now: str,
    notes: str,
    start_cue: str = "",
    transcript_led: bool = False,
    extend_to_match_end: bool = False,
    min_duration: float = 15.0,
    max_span: float | None = None,
) -> DetectedFight | None:
    span = max_span
    if span is None:
        span = 100.0 if not extend_to_match_end else 300.0
    es, ee = _expand_fight_arc(
        analysis,
        start,
        end,
        match_start=match_start,
        match_end=match_end,
        max_span=span,
        cold_gap=5.0,
        transcript_led=transcript_led,
        extend_to_match_end=extend_to_match_end,
    )
    if ee - es < min_duration * 0.6:
        return None
    final_score = _window_action_score(analysis, es, ee)
    strong = bool(STRONG_CUE.search(start_cue))
    contact = bool(CONTACT_CUE.search(start_cue))
    high_audio = "Audio gunfire" in notes and score >= 18.0
    if (
        final_score < 3.0
        and score < 10
        and not strong
        and not contact
        and not extend_to_match_end
        and not transcript_led
        and not high_audio
    ):
        return None
    return DetectedFight(
        id=uuid.uuid4().hex[:10],
        start_sec=round(es, 2),
        end_sec=round(ee, 2),
        score=round(max(final_score, score * 0.4, 6.0 if strong else 0.0), 1),
        confidence="high" if score >= 20 or strong or extend_to_match_end else "medium",
        start_cue=start_cue[:80]
        or transcript_text_in_range(analysis, es, es + 12)[:80],
        end_cue=transcript_text_in_range(analysis, max(es, ee - 12), ee)[:80],
        notes=notes,
        created_at=now,
    )


def detect_fights_in_match(
    analysis: ClipAnalysis,
    match_start: float,
    match_end: float,
    *,
    min_duration: float = 15.0,
) -> list[DetectedFight]:
    raw: list[DetectedFight] = []
    now = datetime.now(timezone.utc).isoformat()

    for start, end, score in _audio_fight_seeds(analysis, match_start, match_end):
        fight = _seed_to_fight(
            analysis, start, end, score,
            match_start=match_start, match_end=match_end, now=now,
            notes="Audio gunfire cluster",
            min_duration=min_duration,
        )
        if fight:
            raw.append(fight)

    for start, end, score, cue in _transcript_cluster_seeds(
        analysis, match_start, match_end
    ):
        fight = _seed_to_fight(
            analysis, start, end, score,
            match_start=match_start, match_end=match_end, now=now,
            notes="Transcript engagement cluster",
            start_cue=cue,
            transcript_led=True,
            min_duration=min_duration * 0.55,
        )
        if fight:
            raw.append(fight)

    for start, end, score, cue in _transcript_fight_seeds(
        analysis, match_start, match_end
    ):
        fight = _seed_to_fight(
            analysis, start, end, score,
            match_start=match_start, match_end=match_end, now=now,
            notes="Transcript combat cue",
            start_cue=cue,
            transcript_led=True,
            min_duration=min_duration * 0.55,
        )
        if fight:
            raw.append(fight)

    for start, end, score in _fill_sparse_gap_seeds(
        analysis, match_start, match_end, raw
    ):
        fight = _seed_to_fight(
            analysis, start, end, score,
            match_start=match_start, match_end=match_end, now=now,
            notes="Sparse peak cluster (quiet fight)",
            min_duration=min_duration * 0.55,
            max_span=125.0,
        )
        if fight:
            raw.append(fight)

    merged = _merge_fights(raw)
    merged = _split_fights_at_cold_gaps(analysis, merged, gap_sec=30.0)
    merged = _merge_adjacent_fights(merged, max_gap_sec=46.0, max_combined_sec=115.0)
    merged = _dedupe_transcript_echoes(merged)
    merged = _collapse_overlapping_fights(merged)
    merged = _dedupe_overlapping_fights(merged)
    merged = _ensure_tail_fight(merged, analysis, match_start, match_end, now)
    merged = _drop_looting_followups(merged)
    merged = _drop_pre_tail_rotation(merged, analysis, match_start, match_end)
    merged = _cap_fight_count(
        merged, match_start, match_end, max_fights=7, min_center_gap=65.0
    )

    death = _death_debrief_time(analysis, match_start, match_end)
    filtered: list[DetectedFight] = []
    for fight in merged:
        if fight.start_sec < match_start + 40:
            continue
        if fight.duration_sec() < min_duration * 0.5 and fight.score < 8:
            if not _is_tail_fight(fight):
                continue
        if death is not None and fight.start_sec >= death + 1:
            continue
        if fight.score < -6 and fight.duration_sec() < 45 and not _is_tail_fight(fight):
            continue
        filtered.append(fight)
    return filtered


def plan_from_detected_fight(
    fight: DetectedFight,
    analysis: ClipAnalysis,
    config: AppConfig,
) -> EditPlan | None:
    """Build an edit plan for one detected fight window (start → end)."""
    start = max(0.0, fight.start_sec)
    end = min(analysis.duration, fight.end_sec)
    if end <= start:
        return None

    hook = fight.start_cue[:72].strip() if fight.start_cue.strip() else ""
    if not hook:
        hook = build_hook_text(analysis, game_profile=config.editing.game_profile)

    segment = EditSegment(start=start, end=end, reason="detected_fight")
    return EditPlan(
        target_duration_sec=end - start,
        segments=[segment],
        effects=[],
        hook_text=hook,
        hashtags=["apexlegends", "gaming", "fyp", "clips"],
        caption_style=config.editing.caption_style,
    )


def plan_from_detected_fights(
    store: DetectedFightStore,
    analysis: ClipAnalysis,
    config: AppConfig,
) -> EditPlan | None:
    """Build a multi-fight montage plan from auto-detected fight windows."""
    usable = [fight for fight in store.fights if fight.use_for_clips]
    if not usable:
        return None

    usable.sort(key=lambda fight: fight.start_sec)
    segments: list[EditSegment] = []
    for fight in usable:
        start = max(0.0, fight.start_sec)
        end = min(analysis.duration, fight.end_sec)
        if end <= start:
            continue
        segments.append(
            EditSegment(start=start, end=end, reason="detected_fight")
        )

    if not segments:
        return None

    hook_source = next(
        (fight.start_cue for fight in usable if fight.start_cue.strip()),
        "",
    )
    hook = (
        hook_source[:72]
        if hook_source
        else build_hook_text(analysis, game_profile=config.editing.game_profile)
    )

    return EditPlan(
        target_duration_sec=sum(seg.end - seg.start for seg in segments),
        segments=segments,
        effects=[],
        hook_text=hook,
        hashtags=["apexlegends", "gaming", "fyp", "clips", "montage"],
        caption_style=config.editing.caption_style,
    )


def scan_vod_fights(
    work_dir: Path,
    vod_id: str,
    analysis: ClipAnalysis,
    config: AppConfig,
) -> DetectedFightStore:
    from twitch_tiktok_bot.labels.matches import load_matches

    matches = load_matches(work_dir)
    if not matches or not matches.matches:
        raise ValueError(
            f"No apex_matches.json in {work_dir}. Run --scan-matches first."
        )

    target = next((m for m in matches.matches if m.use_for_clips), None)
    if target is None:
        target = matches.matches[0]

    fights = detect_fights_in_match(
        analysis, target.start_sec, target.end_sec
    )

    prior = load_detected_fights(work_dir)
    if prior:
        by_range = {
            (round(f.start_sec, 0), round(f.end_sec, 0)): f for f in prior.fights
        }
        for fight in fights:
            key = (round(fight.start_sec, 0), round(fight.end_sec, 0))
            old = by_range.get(key)
            if old:
                fight.id = old.id
                fight.use_for_clips = old.use_for_clips
                fight.notes = old.notes or fight.notes

    store = DetectedFightStore(
        vod_id=vod_id,
        match_id=target.id,
        match_start_sec=target.start_sec,
        match_end_sec=target.end_sec,
        fights=fights,
    )
    save_detected_fights(work_dir, store)
    return store
