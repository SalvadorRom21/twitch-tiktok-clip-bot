"""Apex gameplay literacy labels — teach the bot what in-match moments mean."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from twitch_tiktok_bot.labels.fights import load_fight_labels


EVENT_TYPES = [
    ("unknown", "Not sure / other"),
    ("fight_start", "Fight starts (push / contact)"),
    ("fight_end", "Fight ends (disengage)"),
    ("first_contact", "First contact (heard/saw enemy)"),
    ("push", "Push / aggressive peek"),
    ("shield_crack", "Shield crack"),
    ("knock_enemy", "Enemy knocked"),
    ("knock_self", "You knocked"),
    ("knock_ally", "Teammate knocked"),
    ("kill", "Kill / finish"),
    ("team_wipe", "Team wipe"),
    ("death_box", "Death box spawned"),
    ("revive", "Revive (you or ally)"),
    ("heal_loot", "Heal / loot / crafting"),
    ("rotate", "Rotate / reposition (no fight)"),
    ("third_party", "Third party arrives"),
    ("zone_ring", "Ring / zone pressure"),
    ("downtime", "Downtime (cut from clips)"),
    ("clutch", "Clutch / outplay"),
    ("callout", "Important comm / callout"),
]

CLIP_WORTHY_OPTIONS = [
    ("must_include", "Must include in montage"),
    ("nice", "Nice to include"),
    ("skip", "Skip / boring"),
    ("unsure", "Unsure"),
]


@dataclass
class LiteracyMoment:
    id: str
    timestamp_sec: float
    title: str
    bot_question: str
    category: str = "unknown"
    fight_id: str = ""
    fight_label: str = ""
    context_start_sec: float | None = None
    context_end_sec: float | None = None
    event_type: str = "unknown"
    what_happening: str = ""
    visual_cues: str = ""
    audio_cues: str = ""
    clip_worthy: str = "unsure"
    teaches_bot: str = ""
    answered: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiteracyMoment:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            timestamp_sec=float(data["timestamp_sec"]),
            title=str(data.get("title", "")),
            bot_question=str(data.get("bot_question", "")),
            category=str(data.get("category", "unknown")),
            fight_id=str(data.get("fight_id", "")),
            fight_label=str(data.get("fight_label", "")),
            context_start_sec=(
                float(data["context_start_sec"])
                if data.get("context_start_sec") is not None
                else None
            ),
            context_end_sec=(
                float(data["context_end_sec"])
                if data.get("context_end_sec") is not None
                else None
            ),
            event_type=str(data.get("event_type", "unknown")),
            what_happening=str(data.get("what_happening", "")),
            visual_cues=str(data.get("visual_cues", "")),
            audio_cues=str(data.get("audio_cues", "")),
            clip_worthy=str(data.get("clip_worthy", "unsure")),
            teaches_bot=str(data.get("teaches_bot", "")),
            answered=bool(data.get("answered", False)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class GeneralLiteracyAnswer:
    id: str
    prompt: str
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneralLiteracyAnswer:
        return cls(
            id=str(data["id"]),
            prompt=str(data.get("prompt", "")),
            answer=str(data.get("answer", "")),
        )


@dataclass
class LiteracyStore:
    vod_id: str
    moments: list[LiteracyMoment] = field(default_factory=list)
    general: list[GeneralLiteracyAnswer] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vod_id": self.vod_id,
            "moments": [moment.to_dict() for moment in self.moments],
            "general": [item.to_dict() for item in self.general],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiteracyStore:
        return cls(
            vod_id=str(data.get("vod_id", "")),
            moments=[
                LiteracyMoment.from_dict(item) for item in data.get("moments", [])
            ],
            general=[
                GeneralLiteracyAnswer.from_dict(item)
                for item in data.get("general", [])
            ],
        )


def literacy_path(work_dir: Path) -> Path:
    return work_dir / "literacy_moments.json"


def load_literacy_store(work_dir: Path) -> LiteracyStore | None:
    path = literacy_path(work_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LiteracyStore.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_literacy_store(work_dir: Path, store: LiteracyStore) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = literacy_path(work_dir)
    path.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return path


def _offset_in_fight(start: float, end: float, ratio: float) -> float:
    return round(start + (end - start) * ratio, 2)


def _moment(
    *,
    timestamp: float,
    title: str,
    question: str,
    category: str,
    fight_id: str = "",
    fight_label: str = "",
    ctx_start: float | None = None,
    ctx_end: float | None = None,
) -> LiteracyMoment:
    return LiteracyMoment(
        id=uuid.uuid4().hex[:10],
        timestamp_sec=timestamp,
        title=title,
        bot_question=question,
        category=category,
        fight_id=fight_id,
        fight_label=fight_label,
        context_start_sec=ctx_start,
        context_end_sec=ctx_end,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def seed_moments_from_fights(work_dir: Path, vod_id: str) -> list[LiteracyMoment]:
    """Moments the bot cannot identify — anchored to your labeled fights."""
    fights = load_fight_labels(work_dir)
    moments: list[LiteracyMoment] = []

    if fights:
        for fight in fights.fights:
            fid = fight.id
            label = fight.description[:48] or fight.tags[0] if fight.tags else "fight"
            s, e = fight.start_sec, fight.end_sec

            moments.append(
                _moment(
                    timestamp=s,
                    title="Fight start?",
                    question=(
                        "What is happening here? What is the FIRST cue that combat "
                        "started (HUD, audio, movement)?"
                    ),
                    category="fight_start",
                    fight_id=fid,
                    fight_label=label,
                    ctx_start=s,
                    ctx_end=min(e, s + 12),
                )
            )
            moments.append(
                _moment(
                    timestamp=_offset_in_fight(s, e, 0.35),
                    title="Mid-fight beat",
                    question=(
                        "What's going on in this stretch? Is this peak action, "
                        "downtime, healing, or rotate?"
                    ),
                    category="mid_fight",
                    fight_id=fid,
                    fight_label=label,
                    ctx_start=_offset_in_fight(s, e, 0.28),
                    ctx_end=_offset_in_fight(s, e, 0.42),
                )
            )
            moments.append(
                _moment(
                    timestamp=e - 4.0,
                    title="Fight end?",
                    question=(
                        "Why does the fight end here? What on screen confirms it's "
                        "over (wipe, knock, zone, disengage)?"
                    ),
                    category="fight_end",
                    fight_id=fid,
                    fight_label=label,
                    ctx_start=max(s, e - 10),
                    ctx_end=e,
                )
            )

            if "revive" in fight.tags or "revive" in fight.description.lower():
                moments.append(
                    _moment(
                        timestamp=s + 2.0,
                        title="Post-revive moment",
                        question=(
                            "How can you tell someone was just revived? What makes "
                            "this moment dangerous?"
                        ),
                        category="revive",
                        fight_id=fid,
                        fight_label=label,
                        ctx_start=s,
                        ctx_end=min(e, s + 15),
                    )
                )
            if "teamwipe" in fight.tags or "wipe" in fight.description.lower():
                moments.append(
                    _moment(
                        timestamp=e - 2.0,
                        title="Wipe / death box?",
                        question=(
                            "Is this the team wipe? Where is the death box / kill "
                            "feed and what should the bot look for?"
                        ),
                        category="team_wipe",
                        fight_id=fid,
                        fight_label=label,
                        ctx_start=max(s, e - 8),
                        ctx_end=e,
                    )
                )
            if "zone" in fight.description.lower():
                moments.append(
                    _moment(
                        timestamp=e - 6.0,
                        title="Zone / ring pressure",
                        question=(
                            "What UI or audio says the ring is forcing you to stop "
                            "fighting and run?"
                        ),
                        category="zone_ring",
                        fight_id=fid,
                        fight_label=label,
                        ctx_start=max(s, e - 20),
                        ctx_end=e,
                    )
                )

    # Fight 2 specific beats (bot's main training clip)
    fight2 = next((f for f in (fights.fights if fights else []) if f.id == "edd9aa091a"), None)
    if fight2:
        s, e = fight2.start_sec, fight2.end_sec
        moments.append(
            _moment(
                timestamp=_offset_in_fight(s, e, 0.45),
                title="You knocked?",
                question=(
                    "Are you knocked here? What does knocked look/sound like on "
                    "your POV — include or cut from montage?"
                ),
                category="knock_self",
                fight_id=fight2.id,
                fight_label="teamwipe fight",
                ctx_start=_offset_in_fight(s, e, 0.38),
                ctx_end=_offset_in_fight(s, e, 0.52),
            )
        )
        moments.append(
            _moment(
                timestamp=_offset_in_fight(s, e, 0.15),
                title="Enemy push after rez",
                question=(
                    "What signals the enemy team is pushing you right now?"
                ),
                category="push",
                fight_id=fight2.id,
                fight_label="teamwipe fight",
                ctx_start=s,
                ctx_end=_offset_in_fight(s, e, 0.25),
            )
        )

    # Dedupe by timestamp + title
    seen: set[tuple[float, str]] = set()
    unique: list[LiteracyMoment] = []
    for moment in sorted(moments, key=lambda m: m.timestamp_sec):
        key = (round(moment.timestamp_sec, 1), moment.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(moment)
    return unique


def load_general_prompts(project_root: Path) -> list[GeneralLiteracyAnswer]:
    path = project_root / "data" / "reference" / "apex_game_literacy.yaml"
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []

    items: list[GeneralLiteracyAnswer] = []
    for entry in data.get("general_questions", []):
        items.append(
            GeneralLiteracyAnswer(
                id=str(entry.get("id", "")),
                prompt=str(entry.get("prompt", "")),
                answer=str(entry.get("answer", "")),
            )
        )
    return items


def ensure_literacy_store(work_dir: Path, vod_id: str, project_root: Path) -> LiteracyStore:
    existing = load_literacy_store(work_dir)
    general = load_general_prompts(project_root)

    if existing:
        by_id = {item.id: item for item in general}
        merged_general = []
        for item in general:
            old = next((g for g in existing.general if g.id == item.id), None)
            if old and old.answer.strip():
                merged_general.append(old)
            else:
                merged_general.append(item)
        for g in existing.general:
            if g.id not in by_id:
                merged_general.append(g)
        existing.general = merged_general
        return existing

    return LiteracyStore(
        vod_id=vod_id,
        moments=seed_moments_from_fights(work_dir, vod_id),
        general=general,
    )


def sync_literacy_to_reference(
    store: LiteracyStore,
    output_path: Path,
    project_root: Path,
) -> Path:
    """Merge labeled answers into apex_game_literacy.yaml for the bot."""
    path = project_root / "data" / "reference" / "apex_game_literacy.yaml"
    if path.exists():
        try:
            base = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            base = {}
    else:
        base = {}

    base["meta"] = {
        "vod_id": store.vod_id,
        "status": "curator_labeled",
        "updated": datetime.now(timezone.utc).isoformat(),
        "moments_answered": sum(1 for m in store.moments if m.answered),
        "moments_total": len(store.moments),
    }

    for item in store.general:
        for entry in base.get("general_questions", []):
            if entry.get("id") == item.id:
                entry["answer"] = item.answer

    learned: dict[str, list[str]] = {
        "fight_start": [],
        "fight_end": [],
        "action_beats": [],
        "downtime": [],
        "hud_cues": [],
    }
    for moment in store.moments:
        if not moment.answered:
            continue
        rule = moment.teaches_bot.strip() or moment.what_happening.strip()
        if not rule:
            continue
        if moment.event_type in ("fight_start", "first_contact", "push"):
            learned["fight_start"].append(rule)
        elif moment.event_type in ("fight_end", "rotate", "zone_ring"):
            learned["fight_end"].append(rule)
        elif moment.event_type in ("downtime", "heal_loot"):
            learned["downtime"].append(rule)
        elif moment.clip_worthy == "must_include":
            learned["action_beats"].append(rule)
        if moment.visual_cues.strip():
            learned["hud_cues"].append(moment.visual_cues.strip())

    base["learned_patterns"] = learned
    base["labeled_moments"] = [
        {
            "id": m.id,
            "time_sec": m.timestamp_sec,
            "title": m.title,
            "event_type": m.event_type,
            "what_happening": m.what_happening,
            "visual_cues": m.visual_cues,
            "audio_cues": m.audio_cues,
            "clip_worthy": m.clip_worthy,
            "teaches_bot": m.teaches_bot,
            "fight_id": m.fight_id,
        }
        for m in store.moments
        if m.answered
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path
