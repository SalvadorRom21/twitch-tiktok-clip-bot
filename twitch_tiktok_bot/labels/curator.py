"""Learn fight timing from human-labeled reference fights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from twitch_tiktok_bot.labels.fights import FightLabel, FightLabelStore, load_fight_labels
from twitch_tiktok_bot.models import ClipAnalysis
from twitch_tiktok_bot.plan.action import (
    expand_gunfight_window,
    fight_arc_quality,
    segment_action_score,
)


@dataclass(frozen=True)
class CuratorFightRef:
    vod_id: str
    fight: FightLabel
    guide: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.fight.end_sec - self.fight.start_sec)


def load_curator_references(
    data_dir: Path,
    *,
    vod_id: str | None = None,
) -> list[CuratorFightRef]:
    """Load good/reference fight labels (not bad examples)."""
    refs: list[CuratorFightRef] = []
    if not data_dir.exists():
        return refs

    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if vod_id and child.name != vod_id:
            continue
        store = load_fight_labels(child)
        if not store:
            continue
        for fight in store.fights:
            if fight.quality == "bad":
                continue
            refs.append(
                CuratorFightRef(
                    vod_id=store.vod_id or child.name,
                    fight=fight,
                    guide=store.guide,
                )
            )
    refs.sort(key=lambda ref: ref.fight.start_sec)
    return refs


def load_curator_references_for_work_dir(work_dir: Path) -> list[CuratorFightRef]:
    data_dir = work_dir.parent
    return load_curator_references(data_dir, vod_id=work_dir.name)


def sync_curator_reference_yaml(
    data_dir: Path,
    output_path: Path,
) -> Path | None:
    """Export all curator labels to data/reference for tuning docs."""
    refs = load_curator_references(data_dir)
    if not refs:
        return None

    by_vod: dict[str, dict] = {}
    for ref in refs:
        bucket = by_vod.setdefault(
            ref.vod_id,
            {"vod_id": ref.vod_id, "guide": ref.guide, "fights": []},
        )
        if ref.guide and not bucket["guide"]:
            bucket["guide"] = ref.guide
        bucket["fights"].append(
            {
                "id": ref.fight.id,
                "start_sec": ref.fight.start_sec,
                "end_sec": ref.fight.end_sec,
                "duration_sec": round(ref.duration, 1),
                "description": ref.fight.description,
                "tags": ref.fight.tags,
                "quality": ref.fight.quality,
                "use_for_clips": ref.fight.use_for_clips,
                "notes": ref.fight.notes,
            }
        )

    payload = {
        "source": "fight_labels.json (curator UI)",
        "patterns": {
            "fight_start": "push into fight / first contact (not audio peak tail)",
            "fight_end": "wipe or disengage before rez-loot; not post-fight revive",
            "clip_from_fight": "anchor at labeled start; avoid tail-only windows",
        },
        "vods": list(by_vod.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _overlap_ratio(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if overlap <= 0:
        return 0.0
    shorter = max(0.1, min(a_end - a_start, b_end - b_start))
    return overlap / shorter


def find_curator_ref_for_window(
    refs: list[CuratorFightRef],
    start: float,
    end: float,
    *,
    min_overlap: float = 0.2,
) -> CuratorFightRef | None:
    best: CuratorFightRef | None = None
    best_ratio = 0.0
    mid = (start + end) / 2
    for ref in refs:
        ratio = _overlap_ratio(
            start, end, ref.fight.start_sec, ref.fight.end_sec
        )
        inside = ref.fight.start_sec <= mid <= ref.fight.end_sec
        score = ratio + (0.35 if inside else 0.0)
        if score >= min_overlap and score > best_ratio:
            best = ref
            best_ratio = score
    return best


def align_window_to_curator_fight(
    start: float,
    end: float,
    ref: CuratorFightRef,
) -> tuple[float, float]:
    """
    Re-anchor late detections (audio peak tail) to the curator fight start.

    Example: auto picked 597–611 inside labeled 529–620 fight → snap start to 529.
    """
    fight_start = ref.fight.start_sec
    fight_end = ref.fight.end_sec
    span = max(0.1, fight_end - fight_start)

    if _overlap_ratio(start, end, fight_start, fight_end) < 0.12:
        return start, end

    late_start = start > fight_start + span * 0.35
    if late_start:
        start = fight_start

    end = min(end, fight_end)
    if end <= start:
        end = min(fight_end, start + 12.0)
    return start, end


def best_clip_in_curator_fight(
    analysis: ClipAnalysis,
    ref: CuratorFightRef,
    *,
    max_duration: float = 22.0,
    min_duration: float = 10.0,
) -> tuple[float, float, float] | None:
    """Pick clip window inside a labeled fight; uses full span when max_duration is uncapped."""
    fight_start = ref.fight.export_start()
    fight_end = min(analysis.duration, ref.fight.export_end())
    span = fight_end - fight_start
    if span < min_duration * 0.5:
        return None

    uncapped = max_duration <= 0 or max_duration >= span - 0.5
    if uncapped:
        arc = fight_arc_quality(analysis, fight_start, fight_end)
        action = segment_action_score(
            analysis, fight_start, fight_end, require_combat_for_apex=True
        )
        return fight_start, fight_end, arc * 2.5 + action + 30.0

    best: tuple[float, float, float] | None = None
    step = 2.0
    latest_start = max(fight_start, fight_end - max_duration)
    t = fight_start
    while t <= latest_start + 0.01:
        end = min(fight_end, t + max_duration)
        if end - t < min_duration:
            t += step
            continue

        start, end = expand_gunfight_window(
            analysis,
            t,
            end,
            max_duration=max_duration,
            max_prelude=min(span, max(20.0, span * 0.85)),
            min_duration=min_duration,
        )
        start, end = align_window_to_curator_fight(start, end, ref)
        start = max(fight_start, start)
        end = min(fight_end, end)
        if end - start < min_duration:
            t += step
            continue
        if max_duration > 0 and end - start > max_duration:
            end = start + max_duration

        arc = fight_arc_quality(analysis, start, end)
        action = segment_action_score(
            analysis, start, end, require_combat_for_apex=True
        )
        # Reward early anchor (push-in), punish tail-only windows.
        rel_start = (start - fight_start) / max(span, 0.1)
        anchor_bonus = max(0.0, 12.0 - rel_start * 18.0)
        tail_penalty = 14.0 if rel_start > 0.45 else 0.0

        score = arc * 2.5 + action + anchor_bonus - tail_penalty
        if best is None or score > best[2]:
            best = (start, end, score)
        t += step

    return best


def curator_gunfight_candidates(
    analysis: ClipAnalysis,
    refs: list[CuratorFightRef],
    *,
    max_duration: float = 22.0,
    min_duration: float = 10.0,
) -> list[tuple[float, float, float, str]]:
    """(start, end, score, quote) tuples from curator-labeled fights."""
    out: list[tuple[float, float, float, str]] = []
    for ref in refs:
        window = best_clip_in_curator_fight(
            analysis,
            ref,
            max_duration=max_duration,
            min_duration=min_duration,
        )
        if window is None:
            continue
        start, end, score = window
        quote = ref.fight.description or ref.fight.notes or "curator fight"
        out.append((start, end, score + 25.0, quote))
    out.sort(key=lambda item: item[2], reverse=True)
    return out


def apply_curator_to_gunfight_window(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    refs: list[CuratorFightRef],
    *,
    max_duration: float = 22.0,
    min_duration: float = 10.0,
) -> tuple[float, float]:
    """Snap auto-detected gunfights to curator timing when they overlap."""
    ref = find_curator_ref_for_window(refs, start, end)
    if ref is None:
        return start, end

    start, end = align_window_to_curator_fight(start, end, ref)
    span = ref.duration
    start, end = expand_gunfight_window(
        analysis,
        start,
        end,
        max_duration=max_duration,
        max_prelude=min(span, max(25.0, span * 0.9)),
        min_duration=min_duration,
    )
    start, end = align_window_to_curator_fight(start, end, ref)
    start = max(ref.fight.start_sec, start)
    end = min(ref.fight.end_sec, end)
    if max_duration > 0 and end - start > max_duration:
        end = start + max_duration
    return start, end
