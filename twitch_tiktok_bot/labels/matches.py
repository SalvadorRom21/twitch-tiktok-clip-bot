"""Detect full Apex match windows (drop ship → win or elimination) in a VOD."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.models import ClipAnalysis, TranscriptSegment


DROP_CUE = re.compile(
    r"drop there|drop\s+there|where are we going|let'?s go|jump(?:ing)?|"
    r"launch|deploy|skydive|off the ship|dropship",
    re.I,
)
END_CUE = re.compile(
    r"first game|decent first game|decent game|good first game|"
    r"champion|you are the champion|squad eliminated|eliminated|"
    r"return to lobby|summary screen|placement|how do I not get any kills|"
    r"got (?:zero|no) kills",
    re.I,
)
DEATH_CUE = re.compile(
    r"fuck me|teammates left me|out of ammo that whole time|i'?m dead|"
    r"we'?re dead|that'?s game|game over",
    re.I,
)
LOBBY_CUE = re.compile(
    r"rank ladder|push to talk|legend|caustic|lifeline|character select|"
    r"ready up|queue|lobby|settings",
    re.I,
)


@dataclass
class ApexMatch:
    id: str
    start_sec: float
    end_sec: float
    start_cue: str = ""
    end_cue: str = ""
    end_type: str = "unknown"  # elimination | win | unknown
    confidence: str = "low"  # low | medium | high
    notes: str = ""
    use_for_clips: bool = False
    created_at: str = ""

    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ApexMatch:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            start_cue=str(data.get("start_cue", "")),
            end_cue=str(data.get("end_cue", "")),
            end_type=str(data.get("end_type", "unknown")),
            confidence=str(data.get("confidence", "low")),
            notes=str(data.get("notes", "")),
            use_for_clips=bool(data.get("use_for_clips", False)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class ApexMatchStore:
    vod_id: str
    matches: list[ApexMatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "vod_id": self.vod_id,
            "matches": [match.to_dict() for match in self.matches],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ApexMatchStore:
        return cls(
            vod_id=str(data.get("vod_id", "")),
            matches=[ApexMatch.from_dict(item) for item in data.get("matches", [])],
        )


def matches_path(work_dir: Path) -> Path:
    return work_dir / "apex_matches.json"


def load_matches(work_dir: Path) -> ApexMatchStore | None:
    path = matches_path(work_dir)
    if not path.exists():
        return None
    try:
        return ApexMatchStore.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_matches(work_dir: Path, store: ApexMatchStore) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = matches_path(work_dir)
    path.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return path


def _peak_density(
    analysis: ClipAnalysis, start: float, end: float
) -> float:
    if end <= start:
        return 0.0
    count = sum(
        1 for peak in analysis.loud_peaks if start <= peak.time < end
    )
    return count / max(end - start, 1.0)


def _is_ship_drop_cue(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(r"drop there|where are we going|skydive|dropship|off the ship", lower)
        or ("drop" in lower and "let's go" in lower)
    )


def _find_drop_starts(
    transcript: list[TranscriptSegment],
    analysis: ClipAnalysis,
) -> list[tuple[float, str]]:
    """Drop cues after lobby/setup — not mid-fight 'let's go'."""
    candidates: list[tuple[float, str]] = []
    for seg in transcript:
        text = seg.text.strip()
        if not DROP_CUE.search(text):
            continue
        if LOBBY_CUE.search(text) and "drop" not in text.lower():
            continue

        prior_quiet = _peak_density(
            analysis, max(0.0, seg.start - 120), seg.start
        )
        strong_drop = _is_ship_drop_cue(text)
        mid_fight = _peak_density(analysis, max(0.0, seg.start - 90), seg.start) > 0.12

        if mid_fight and not strong_drop:
            continue
        if not strong_drop and prior_quiet > 0.1:
            continue
        if not strong_drop and "let's go" in text.lower() and prior_quiet > 0.05:
            continue

        candidates.append((seg.start, text[:120]))
    return _thin_drop_candidates(candidates)


def _thin_drop_candidates(
    drops: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    if not drops:
        return drops
    drops = sorted(drops, key=lambda item: item[0])
    kept: list[tuple[float, str]] = []
    for drop_time, text in drops:
        if not kept:
            kept.append((drop_time, text))
            continue
        last_time, last_text = kept[-1]
        if drop_time - last_time < 150:
            if _is_ship_drop_cue(text) and not _is_ship_drop_cue(last_text):
                kept[-1] = (drop_time, text)
            elif _is_ship_drop_cue(last_text):
                pass
            elif drop_time - last_time < 90:
                pass
            else:
                kept.append((drop_time, text))
            continue
        if drop_time - last_time < 600 and not _is_ship_drop_cue(text):
            continue
        kept.append((drop_time, text))
    return kept


def _find_match_ends(
    transcript: list[TranscriptSegment],
) -> list[tuple[float, str, str]]:
    """Post-game or death cues."""
    ends: list[tuple[float, str, str]] = []
    for seg in transcript:
        text = seg.text.strip()
        if END_CUE.search(text):
            end_type = "win" if re.search(r"champion|won|win", text, re.I) else "elimination"
            ends.append((seg.end, text[:120], end_type))
        elif DEATH_CUE.search(text):
            ends.append((seg.end, text[:120], "elimination"))
    return ends


def detect_apex_matches(
    analysis: ClipAnalysis,
    *,
    min_match_sec: float = 480.0,
    max_match_sec: float = 1680.0,
) -> list[ApexMatch]:
    """
    Heuristic match boundaries from transcript + audio density.

    Pair each drop cue with the next end/death cue to form one match arc.
    """
    transcript = sorted(analysis.transcript_segments, key=lambda s: s.start)
    drops = _find_drop_starts(transcript, analysis)
    ends = _find_match_ends(transcript)

    if not drops:
        return []

    matches: list[ApexMatch] = []
    used_end = -1

    for drop_time, drop_text in drops:
        end_hit: tuple[float, str, str] | None = None
        for index, (end_time, end_text, end_type) in enumerate(ends):
            if end_time <= drop_time + min_match_sec * 0.35:
                continue
            span = end_time - drop_time
            if span > max_match_sec:
                continue
            if end_hit is None or end_time > end_hit[0]:
                end_hit = (end_time, end_text, end_type)
                used_end = max(used_end, index)

        if end_hit is None:
            continue

        end_time, end_text, end_type = end_hit
        span = end_time - drop_time
        confidence = "medium"
        if span >= min_match_sec and re.search(r"first game|decent", end_text, re.I):
            confidence = "high"
        if re.search(r"drop there|where are we going", drop_text, re.I):
            confidence = "high"

        matches.append(
            ApexMatch(
                id=uuid.uuid4().hex[:10],
                start_sec=round(max(0.0, drop_time - 2.0), 2),
                end_sec=round(min(analysis.duration, end_time + 3.0), 2),
                start_cue=drop_text,
                end_cue=end_text,
                end_type=end_type,
                confidence=confidence,
                notes="Auto-detected from drop + post-game/death transcript cues.",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    return _dedupe_overlapping_matches(matches)


def _overlap_fraction(a: ApexMatch, b: ApexMatch) -> float:
    overlap = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    shorter = max(0.1, min(a.duration_sec(), b.duration_sec()))
    return overlap / shorter


def _dedupe_overlapping_matches(matches: list[ApexMatch]) -> list[ApexMatch]:
    if len(matches) <= 1:
        return matches

    grouped: list[list[ApexMatch]] = []
    for match in sorted(matches, key=lambda m: m.start_sec):
        if not grouped or abs(match.start_sec - grouped[-1][0].start_sec) > 90:
            grouped.append([match])
        else:
            grouped[-1].append(match)

    kept: list[ApexMatch] = []
    for bucket in grouped:
        best = max(
            bucket,
            key=lambda m: (
                {"high": 3, "medium": 2, "low": 1}.get(m.confidence, 0),
                1 if re.search(r"first game|decent", m.end_cue, re.I) else 0,
                m.duration_sec(),
            ),
        )
        if kept and _overlap_fraction(best, kept[-1]) > 0.5:
            if best.duration_sec() > kept[-1].duration_sec():
                kept[-1] = best
        else:
            kept.append(best)
    return kept


def detect_and_save_matches(
    work_dir: Path,
    vod_id: str,
    analysis: ClipAnalysis,
) -> ApexMatchStore:
    existing = load_matches(work_dir)
    detected = detect_apex_matches(analysis)

    if existing:
        by_range = {
            (round(m.start_sec, 1), round(m.end_sec, 1)): m for m in existing.matches
        }
        merged: list[ApexMatch] = []
        for match in detected:
            key = (round(match.start_sec, 1), round(match.end_sec, 1))
            old = by_range.get(key)
            if old:
                match.use_for_clips = old.use_for_clips
                match.notes = old.notes or match.notes
                match.id = old.id
            merged.append(match)
        for old in existing.matches:
            key = (round(old.start_sec, 1), round(old.end_sec, 1))
            if not any(
                round(m.start_sec, 1) == key[0] and round(m.end_sec, 1) == key[1]
                for m in merged
            ):
                merged.append(old)
        merged.sort(key=lambda m: m.start_sec)
        store = ApexMatchStore(vod_id=vod_id, matches=merged)
    else:
        store = ApexMatchStore(vod_id=vod_id, matches=detected)

    save_matches(work_dir, store)
    return store


def fights_in_match(
    work_dir: Path,
    match: ApexMatch,
) -> list[dict]:
    """Return fight labels overlapping this match window."""
    from twitch_tiktok_bot.labels.fights import load_fight_labels

    store = load_fight_labels(work_dir)
    if not store:
        return []
    hits = []
    for fight in store.fights:
        overlap = max(
            0.0,
            min(match.end_sec, fight.end_sec) - max(match.start_sec, fight.start_sec),
        )
        if overlap > 0:
            hits.append(
                {
                    "id": fight.id,
                    "start_sec": fight.start_sec,
                    "end_sec": fight.end_sec,
                    "overlap_sec": round(overlap, 1),
                    "description": fight.description,
                }
            )
    return hits
