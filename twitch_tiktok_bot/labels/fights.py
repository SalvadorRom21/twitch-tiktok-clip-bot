"""Save and apply manually labeled Apex fight windows."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan, EditSegment
from twitch_tiktok_bot.plan.duration import target_clip_duration_sec
from twitch_tiktok_bot.plan.moments import build_hook_text


@dataclass
class FightLabel:
    """
    Human-marked fight on a POV recording.

    start_sec / end_sec — full fight arc (contact → wipe, before rez/loot).
    clip_start_sec / clip_end_sec — optional tighter export window inside the fight.
    """

    id: str
    start_sec: float
    end_sec: float
    description: str = ""
    tags: list[str] = field(default_factory=list)
    quality: str = "good"
    use_for_clips: bool = True
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    notes: str = ""
    created_at: str = ""

    def export_start(self) -> float:
        return self.clip_start_sec if self.clip_start_sec is not None else self.start_sec

    def export_end(self) -> float:
        return self.clip_end_sec if self.clip_end_sec is not None else self.end_sec

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FightLabel:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            description=str(data.get("description", "")),
            tags=list(data.get("tags") or []),
            quality=str(data.get("quality", "good")),
            use_for_clips=bool(data.get("use_for_clips", True)),
            clip_start_sec=(
                float(data["clip_start_sec"])
                if data.get("clip_start_sec") is not None
                else None
            ),
            clip_end_sec=(
                float(data["clip_end_sec"])
                if data.get("clip_end_sec") is not None
                else None
            ),
            notes=str(data.get("notes", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class FightLabelStore:
    vod_id: str
    fights: list[FightLabel] = field(default_factory=list)
    guide: str = ""

    def to_dict(self) -> dict:
        return {
            "vod_id": self.vod_id,
            "guide": self.guide,
            "fights": [fight.to_dict() for fight in self.fights],
        }

    @classmethod
    def from_dict(cls, data: dict) -> FightLabelStore:
        return cls(
            vod_id=str(data.get("vod_id", "")),
            guide=str(data.get("guide", "")),
            fights=[FightLabel.from_dict(item) for item in data.get("fights", [])],
        )


def fight_labels_path(work_dir: Path) -> Path:
    return work_dir / "fight_labels.json"


def load_fight_labels(work_dir: Path) -> FightLabelStore | None:
    path = fight_labels_path(work_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FightLabelStore.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_fight_labels(work_dir: Path, store: FightLabelStore) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = fight_labels_path(work_dir)
    path.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return path


def find_cached_vods(data_dir: Path) -> list[dict]:
    vods: list[dict] = []
    if not data_dir.exists():
        return vods
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        videos = sorted(child.glob("*.mp4"))
        if not videos:
            continue
        labels = load_fight_labels(child)
        vods.append(
            {
                "vod_id": child.name,
                "video": videos[0].name,
                "label_count": len(labels.fights) if labels else 0,
            }
        )
    return vods


def resolve_vod_video(config: AppConfig, vod_id: str) -> tuple[Path, Path]:
    from twitch_tiktok_bot.ingest.download import find_cached_video

    work_dir = config.resolve_path(config.paths.data_dir) / vod_id
    if not work_dir.exists():
        raise FileNotFoundError(
            f"No cached data folder for {vod_id} at {work_dir}. "
            f"Download the VOD first or run from the project root."
        )
    video = find_cached_video(work_dir, vod_id)
    if video is None:
        videos = sorted(work_dir.glob("*.mp4"))
        video = videos[0] if videos else None
    if video is None:
        raise FileNotFoundError(
            f"No video file in {work_dir}. "
            f"Expected data/{vod_id}/*.mp4 — re-run a VOD download first."
        )
    return work_dir, video


def plan_from_fight_labels(
    store: FightLabelStore,
    analysis: ClipAnalysis,
    config: AppConfig,
) -> EditPlan | None:
    """Build an edit plan from curator labels (overrides auto detection)."""
    usable = [
        fight
        for fight in store.fights
        if fight.use_for_clips and fight.quality != "bad"
    ]
    if not usable:
        return None

    usable.sort(key=lambda fight: fight.start_sec)
    editing = config.editing
    max_segments = editing.max_montage_segments
    target = target_clip_duration_sec(editing)

    segments: list[EditSegment] = []
    total = 0.0
    for fight in usable:
        if len(segments) >= max_segments:
            break
        start = max(0.0, fight.export_start())
        end = min(analysis.duration, fight.export_end())
        if end <= start:
            continue
        length = end - start
        if target < 1e8 and total + length > target + 1.0 and segments:
            continue
        segments.append(
            EditSegment(start=start, end=end, reason="labeled_fight")
        )
        total += length

    if not segments:
        return None

    primary = usable[0]
    hook_source = primary.description or primary.notes or (
        primary.tags[0] if primary.tags else ""
    )
    hook = (
        hook_source[:72]
        if hook_source
        else build_hook_text(analysis, game_profile=editing.game_profile)
    )
    tag_hashtags = [
        tag.lstrip("#").replace(" ", "")
        for tag in primary.tags
        if tag.strip()
    ]
    hashtags = tag_hashtags or ["apexlegends", "gaming", "fyp", "clips"]

    return EditPlan(
        target_duration_sec=sum(seg.end - seg.start for seg in segments),
        segments=segments,
        effects=[],
        hook_text=hook,
        hashtags=hashtags,
        caption_style=editing.caption_style,
    )


def new_fight_label(
    *,
    start_sec: float,
    end_sec: float,
    description: str = "",
    tags: list[str] | None = None,
    quality: str = "good",
    use_for_clips: bool = True,
    notes: str = "",
) -> FightLabel:
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return FightLabel(
        id=uuid.uuid4().hex[:10],
        start_sec=round(start_sec, 2),
        end_sec=round(end_sec, 2),
        description=description.strip(),
        tags=tags or [],
        quality=quality,
        use_for_clips=use_for_clips,
        notes=notes.strip(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
