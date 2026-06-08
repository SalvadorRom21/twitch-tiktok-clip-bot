"""Detect Super Smash Bros. Ultimate sets (Bo3/Bo5) and individual games in a VOD."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan, EditSegment
from twitch_tiktok_bot.plan.moments import build_hook_text

GAME_START_CUE = re.compile(
    r"game\s*(?:one|1|two|2|three|3|four|4|five|5)\b|"
    r"game count|counter\s*pick|counterpick|"
    r"(?:three|3)[,\s]+(?:two|2)[,\s]+(?:one|1)|"
    r"ready[!.]|here we go",
    re.I,
)
GAME_END_CUE = re.compile(
    r"(?:takes?|wins?) (?:the )?game|"
    r"that'?s (?:the )?game|"
    r"game[!]?\s*(?:to|goes to)|"
    r"game count (?:is|at)|"
    r"(?:two|2|three|3)[\s-]+(?:zero|0|one|1|two|2)|"
    r"stocks?(?:\s+remaining)?",
    re.I,
)
SET_END_CUE = re.compile(
    r"(?:takes?|wins?) (?:the )?set|"
    r"set (?:to|goes to|over)|"
    r"wins? the (?:series|match)",
    re.I,
)
BO3_CUE = re.compile(r"best of (?:three|3)|\bbo\s*3\b", re.I)
BO5_CUE = re.compile(r"best of (?:five|5)|\bbo\s*5\b", re.I)
SCORE_CUE = re.compile(
    r"\b([23])[\s-]+([012])\b|"
    r"(?:two|2)[\s-]+(?:zero|0|one|1|two|2)|"
    r"(?:three|3)[\s-]+(?:zero|0|one|1|two|2)",
    re.I,
)
SMASH_NAME = re.compile(
    r"smash|ssbu|super smash|ultimate|melee|mang\d|genesis|evo\b",
    re.I,
)

MIN_GAME_SEC = 75.0
MAX_GAME_SEC = 720.0
INTER_GAME_GAP_SEC = 150.0
INTER_SET_GAP_SEC = 210.0


@dataclass
class SmashGame:
    id: str
    start_sec: float
    end_sec: float
    game_number: int = 1
    start_cue: str = ""
    end_cue: str = ""
    confidence: str = "medium"
    use_for_clips: bool = True
    created_at: str = ""

    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SmashGame:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            game_number=int(data.get("game_number", 1)),
            start_cue=str(data.get("start_cue", "")),
            end_cue=str(data.get("end_cue", "")),
            confidence=str(data.get("confidence", "medium")),
            use_for_clips=bool(data.get("use_for_clips", True)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class SmashSet:
    id: str
    start_sec: float
    end_sec: float
    format: str = "unknown"  # bo3 | bo5 | unknown
    games_to_win: int = 2
    games: list[SmashGame] = field(default_factory=list)
    start_cue: str = ""
    end_cue: str = ""
    confidence: str = "medium"
    notes: str = ""
    use_for_clips: bool = True
    created_at: str = ""

    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "games": [game.to_dict() for game in self.games],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SmashSet:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            format=str(data.get("format", "unknown")),
            games_to_win=int(data.get("games_to_win", 2)),
            games=[SmashGame.from_dict(item) for item in data.get("games", [])],
            start_cue=str(data.get("start_cue", "")),
            end_cue=str(data.get("end_cue", "")),
            confidence=str(data.get("confidence", "medium")),
            notes=str(data.get("notes", "")),
            use_for_clips=bool(data.get("use_for_clips", True)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class SmashSetStore:
    vod_id: str
    sets: list[SmashSet] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "vod_id": self.vod_id,
            "sets": [item.to_dict() for item in self.sets],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SmashSetStore:
        return cls(
            vod_id=str(data.get("vod_id", "")),
            sets=[SmashSet.from_dict(item) for item in data.get("sets", [])],
        )


def smash_sets_path(work_dir: Path) -> Path:
    return work_dir / "smash_sets.json"


def load_smash_sets(work_dir: Path) -> SmashSetStore | None:
    path = smash_sets_path(work_dir)
    if not path.exists():
        return None
    try:
        return SmashSetStore.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_smash_sets(work_dir: Path, store: SmashSetStore) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = smash_sets_path(work_dir)
    path.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return path


def _peak_density(analysis: ClipAnalysis, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    count = sum(1 for peak in analysis.loud_peaks if start <= peak.time < end)
    return count / max(end - start, 1.0)


def _text_in_range(analysis: ClipAnalysis, start: float, end: float) -> str:
    parts: list[str] = []
    for seg in analysis.transcript_segments:
        if seg.end < start or seg.start > end:
            continue
        parts.append(seg.text)
    return " ".join(parts)


def _cue_text(seg_text: str, limit: int = 80) -> str:
    return seg_text.strip()[:limit]


def _find_transcript_anchors(
    analysis: ClipAnalysis,
) -> tuple[list[tuple[float, str, str]], list[tuple[float, str, str]]]:
    """Return (starts, ends) as (time, cue_type, text)."""
    starts: list[tuple[float, str, str]] = []
    ends: list[tuple[float, str, str]] = []
    for seg in analysis.transcript_segments:
        text = seg.text.strip()
        if not text:
            continue
        if GAME_START_CUE.search(text):
            starts.append((seg.start, "start", _cue_text(text)))
        if GAME_END_CUE.search(text) or SET_END_CUE.search(text):
            ends.append((seg.end, "end", _cue_text(text)))
    return starts, ends


def _games_from_gap_splits(analysis: ClipAnalysis) -> list[tuple[float, float, str]]:
    """Split VOD into game-sized chunks using silence valleys between activity."""
    if not analysis.loud_peaks:
        return []

    peaks = sorted(analysis.loud_peaks, key=lambda peak: peak.time)
    hot_ranges: list[tuple[float, float]] = []
    chunk_start = peaks[0].time
    prev = peaks[0].time

    for peak in peaks[1:]:
        if peak.time - prev > 14.0:
            if prev - chunk_start >= MIN_GAME_SEC * 0.5:
                hot_ranges.append((chunk_start, prev + 4.0))
            chunk_start = peak.time
        prev = peak.time

    if prev - chunk_start >= MIN_GAME_SEC * 0.5:
        hot_ranges.append((chunk_start, prev + 4.0))

    merged: list[tuple[float, float]] = []
    for start, end in hot_ranges:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        combined = end - prev_start
        if 0 < gap <= INTER_GAME_GAP_SEC and combined <= MAX_GAME_SEC:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    candidates: list[tuple[float, float, str]] = []
    for start, end in merged:
        span = end - start
        if span < MIN_GAME_SEC or span > MAX_GAME_SEC:
            continue
        density = _peak_density(analysis, start, end)
        if density < 0.02:
            continue
        candidates.append((max(0.0, start - 8.0), min(analysis.duration, end + 6.0), "gap_split"))
    return candidates


def _games_from_transcript(
    analysis: ClipAnalysis,
    starts: list[tuple[float, str, str]],
    ends: list[tuple[float, str, str]],
) -> list[tuple[float, float, str, str]]:
    games: list[tuple[float, float, str, str]] = []
    end_times = sorted({item[0] for item in ends})

    for end_time in end_times:
        prior_starts = [s for s in starts if s[0] < end_time]
        if prior_starts:
            start_time, _, start_cue = prior_starts[-1]
        else:
            start_time = max(0.0, end_time - 360.0)
            start_cue = ""

        start_time = max(0.0, start_time - 12.0)
        end_time = min(analysis.duration, end_time + 8.0)
        span = end_time - start_time
        if span < MIN_GAME_SEC or span > MAX_GAME_SEC:
            continue

        end_cue = next((text for t, _, text in ends if abs(t - end_time) < 12), "")
        games.append((start_time, end_time, start_cue, end_cue))

    return games


def _merge_game_candidates(
    transcript_games: list[tuple[float, float, str, str]],
    gap_games: list[tuple[float, float, str]],
) -> list[tuple[float, float, str, str, str]]:
    raw: list[tuple[float, float, str, str, str]] = []
    for start, end, start_cue, end_cue in transcript_games:
        raw.append((start, end, start_cue, end_cue, "transcript"))
    for start, end, source in gap_games:
        raw.append((start, end, "", "", source))

    raw.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, str, str, str]] = []
    for item in raw:
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        overlap = max(0.0, min(item[1], prev[1]) - max(item[0], prev[0]))
        shorter = max(0.1, min(item[1] - item[0], prev[1] - prev[0]))
        if overlap / shorter > 0.45:
            if (item[1] - item[0]) > (prev[1] - prev[0]):
                merged[-1] = item
            continue
        if item[0] - prev[1] < 45.0 and item[1] - prev[0] <= MAX_GAME_SEC:
            merged[-1] = (
                prev[0],
                max(prev[1], item[1]),
                prev[2] or item[2],
                item[3] or prev[3],
                prev[4] if prev[4] == "transcript" else item[4],
            )
        else:
            merged.append(item)
    return merged


def _infer_format_from_text(text: str) -> tuple[str, int] | None:
    if BO5_CUE.search(text):
        return "bo5", 3
    if BO3_CUE.search(text):
        return "bo3", 2
    return None


def _infer_set_format(
    games: list[SmashGame],
    set_text: str,
) -> tuple[str, int, str]:
    explicit = _infer_format_from_text(set_text)
    if explicit:
        return explicit[0], explicit[1], "explicit cue"

    game_count = len(games)
    if SCORE_CUE.search(set_text):
        if re.search(r"\b3[\s-]+[012]\b|three[\s-]+", set_text, re.I):
            return "bo5", 3, "score pattern"
        return "bo3", 2, "score pattern"

    if game_count >= 4:
        return "bo5", 3, f"{game_count} games in set"
    if game_count == 3:
        return "bo3", 2, f"{game_count} games in set"
    if game_count <= 2:
        return "bo3", 2, f"{game_count} game(s); default bo3"
    return "unknown", 2, "could not determine"


def _group_games_into_sets(
    games: list[SmashGame],
    analysis: ClipAnalysis,
) -> list[SmashSet]:
    if not games:
        return []

    sets: list[SmashSet] = []
    current: list[SmashGame] = [games[0]]
    now = datetime.now(timezone.utc).isoformat()

    for game in games[1:]:
        gap = game.start_sec - current[-1].end_sec
        if gap <= INTER_GAME_GAP_SEC:
            current.append(game)
            continue

        if gap >= INTER_SET_GAP_SEC or len(current) >= 5:
            sets.append(_build_set(current, analysis, now))
            current = [game]
        elif len(current) >= 3:
            sets.append(_build_set(current, analysis, now))
            current = [game]
        else:
            current.append(game)

    if current:
        sets.append(_build_set(current, analysis, now))

    return sets


def _build_set(
    games: list[SmashGame],
    analysis: ClipAnalysis,
    now: str,
) -> SmashSet:
    for idx, game in enumerate(games, start=1):
        game.game_number = idx

    start = games[0].start_sec
    end = games[-1].end_sec
    set_text = _text_in_range(analysis, max(0.0, start - 30.0), end + 30.0)
    fmt, wins_needed, note = _infer_set_format(games, set_text)

    confidence = "medium"
    if fmt != "unknown":
        confidence = "high" if "explicit" in note else "medium"
    if len(games) == 1:
        confidence = "low"

    return SmashSet(
        id=uuid.uuid4().hex[:10],
        start_sec=round(start, 2),
        end_sec=round(end, 2),
        format=fmt,
        games_to_win=wins_needed,
        games=games,
        start_cue=games[0].start_cue,
        end_cue=games[-1].end_cue,
        confidence=confidence,
        notes=note,
        created_at=now,
    )


def detect_smash_sets(analysis: ClipAnalysis) -> list[SmashSet]:
    """Detect Bo3/Bo5 sets and individual games from transcript + audio gaps."""
    starts, ends = _find_transcript_anchors(analysis)
    transcript_games = _games_from_transcript(analysis, starts, ends)
    gap_games = _games_from_gap_splits(analysis)
    merged = _merge_game_candidates(transcript_games, gap_games)

    now = datetime.now(timezone.utc).isoformat()
    games: list[SmashGame] = []
    for start, end, start_cue, end_cue, source in merged:
        confidence = "high" if source == "transcript" else "medium"
        games.append(
            SmashGame(
                id=uuid.uuid4().hex[:10],
                start_sec=round(start, 2),
                end_sec=round(end, 2),
                start_cue=start_cue,
                end_cue=end_cue,
                confidence=confidence,
                created_at=now,
            )
        )

    games.sort(key=lambda item: item.start_sec)
    return _group_games_into_sets(games, analysis)


def scan_vod_smash_sets(
    work_dir: Path,
    vod_id: str,
    analysis: ClipAnalysis,
) -> SmashSetStore:
    detected = detect_smash_sets(analysis)
    prior = load_smash_sets(work_dir)
    if prior:
        by_start = {round(item.start_sec, 0): item for item in prior.sets}
        for item in detected:
            old = by_start.get(round(item.start_sec, 0))
            if old:
                item.id = old.id
                item.use_for_clips = old.use_for_clips
                item.notes = old.notes or item.notes

    store = SmashSetStore(vod_id=vod_id, sets=detected)
    save_smash_sets(work_dir, store)
    return store


def is_smash_vod(analysis: ClipAnalysis) -> bool:
    text = f"{analysis.game_name} {analysis.clip_title}".strip()
    return bool(SMASH_NAME.search(text)) if text else False


def plan_from_smash_game(
    game: SmashGame,
    analysis: ClipAnalysis,
    config: AppConfig,
) -> EditPlan | None:
    """One full game clip (detected start → end)."""
    start = max(0.0, game.start_sec)
    end = min(analysis.duration, game.end_sec)
    if end <= start:
        return None

    hook = game.start_cue[:72].strip() if game.start_cue.strip() else ""
    if not hook:
        hook = build_hook_text(analysis, game_profile="smash")

    segment = EditSegment(start=start, end=end, reason="smash_game")
    return EditPlan(
        target_duration_sec=end - start,
        segments=[segment],
        effects=[],
        hook_text=hook,
        hashtags=["ssbu", "smashbros", "gaming", "fyp", "clips"],
        caption_style=config.editing.caption_style,
    )
