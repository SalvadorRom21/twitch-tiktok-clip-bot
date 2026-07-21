"""Incremental Elden Ring detector tuning from guidance and lightweight feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.labels.elden_boss_detect import EldenTuning, load_elden_tuning, save_elden_tuning


def feedback_path(data_dir: Path) -> Path:
    return data_dir / "reference" / "elden_ring_feedback.json"


@dataclass
class FeedbackStore:
    guide: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "guide": self.guide,
            "updated_at": self.updated_at,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackStore:
        return cls(
            guide=str(data.get("guide", "")),
            updated_at=str(data.get("updated_at", "")),
            events=list(data.get("events", [])),
        )


def load_feedback_store(data_dir: Path) -> FeedbackStore:
    path = feedback_path(data_dir)
    if not path.exists():
        return FeedbackStore()
    try:
        return FeedbackStore.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError):
        return FeedbackStore()


def save_feedback_store(data_dir: Path, store: FeedbackStore) -> Path:
    path = feedback_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.updated_at = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return path


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_guide_to_tuning(tuning: EldenTuning, guide: str) -> tuple[EldenTuning, list[str]]:
    """Parse standing instructions into detector parameters."""
    changes: list[str] = []
    text = (guide or "").lower()
    if not text.strip():
        return tuning, changes

    if any(k in text for k in ("cutscene", "cinematic", "fire giant")):
        if tuning.cutscene_preroll_sec < 45.0:
            tuning.cutscene_preroll_sec = 45.0
            changes.append("Include longer pre-fight cutscenes before the boss bar")
    if any(k in text for k in ("full hp", "full health", "full bar", "start of attempt")):
        if tuning.full_bar_fill_min > 0.65:
            tuning.full_bar_fill_min = 0.65
            changes.append("Start clips when the boss bar is at full HP")
    if any(k in text for k in ("enemy felled", "you died", "death screen", "victory screen")):
        tuning.end_on_death = True
        tuning.trim_at_victory = True
        changes.append("End clips on YOU DIED or ENEMY FELLED")
    if any(
        k in text
        for k in (
            "each attempt",
            "per attempt",
            "every attempt",
            "split attempts",
            "one clip per attempt",
        )
    ):
        tuning.split_per_attempt = True
        changes.append("One clip per attempt — death or kill ends the clip")
    if any(k in text for k in ("one clip per boss", "single clip", "don't split", "do not split", "merge attempts")):
        tuning.split_per_attempt = False
        changes.append("Keep one clip per boss encounter")
    if "boss name" in text or "white name" in text:
        tuning.require_name_with_bar = True
        changes.append("Require boss name text with the health bar")
    if "no max" in text or "unlimited" in text or "no cap" in text:
        tuning.max_clip_sec = 0.0
        changes.append("No maximum clip length")

    return tuning, changes


def apply_feedback_event(
    tuning: EldenTuning,
    *,
    verdict: str,
    wrong_reason: str = "",
    notes: str = "",
    guide: str = "",
    timing_adjusted: bool = False,
) -> tuple[EldenTuning, list[str]]:
    """Nudge detector thresholds from one thumbs-up/down (no labeled examples)."""
    changes: list[str] = []
    notes_l = (notes or "").lower()

    tuning, guide_changes = apply_guide_to_tuning(tuning, guide)
    changes.extend(guide_changes)

    if verdict == "missed":
        tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score - 0.02, 0.08, 0.5)
        tuning.min_boss_name_score = _clamp(tuning.min_boss_name_score - 0.015, 0.08, 0.45)
        tuning.start_bar_min_score = _clamp(tuning.start_bar_min_score - 0.02, 0.15, 0.45)
        changes.append("Lowered detection thresholds — look for more boss fights")
        return tuning, changes

    if verdict == "correct":
        # Light positive signal — slightly trust current thresholds.
        tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score - 0.005, 0.08, 0.35)
        tuning.min_boss_name_score = _clamp(tuning.min_boss_name_score - 0.005, 0.08, 0.45)
        changes.append("Confirmed detection — thresholds relaxed slightly")
        return tuning, changes

    if verdict != "wrong":
        return tuning, changes

    reason = (wrong_reason or "").strip()
    if reason == "no_boss_bar":
        tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score + 0.025, 0.08, 0.5)
        tuning.bar_without_name_reject = _clamp(tuning.bar_without_name_reject + 0.03, 0.35, 0.65)
        changes.append("Raised boss bar threshold — fewer bar-less false positives")
    elif reason == "regular_mobs":
        tuning.min_boss_name_score = _clamp(tuning.min_boss_name_score + 0.02, 0.08, 0.55)
        tuning.min_hud_samples = min(6, tuning.min_hud_samples + 1)
        changes.append("Require stronger boss name signal — ignore trash mobs")
    elif reason == "exploring":
        tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score + 0.02, 0.08, 0.5)
        tuning.min_hud_samples = min(6, tuning.min_hud_samples + 1)
        changes.append("Tightened HUD requirements — ignore exploration")
    elif reason == "cutscene":
        # Cutscene-only FPs: shrink preroll and demand real boss HUD + terminal.
        tuning.cutscene_preroll_sec = _clamp(tuning.cutscene_preroll_sec - 8.0, 8.0, 40.0)
        tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score + 0.02, 0.08, 0.5)
        tuning.min_boss_name_score = _clamp(tuning.min_boss_name_score + 0.02, 0.08, 0.55)
        tuning.min_hud_samples = min(6, tuning.min_hud_samples + 1)
        tuning.require_name_with_bar = True
        tuning.require_terminal = True
        changes.append("Reject cutscene-only clips — need named boss HUD + death/victory")
    elif reason == "invader":
        tuning.require_name_with_bar = True
        tuning.min_boss_name_score = _clamp(tuning.min_boss_name_score + 0.02, 0.08, 0.55)
        changes.append("Require named boss HUD — ignore PvP invaders")
    elif reason == "audio_only":
        tuning.require_name_with_bar = True
        tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score + 0.03, 0.08, 0.5)
        changes.append("Ignore loud audio without boss bar + name")
    elif reason == "wrong_timing":
        if any(k in notes_l for k in ("late", "middle", "second half", "too long", "after")):
            tuning.start_bar_lookback_sec = _clamp(
                tuning.start_bar_lookback_sec + 15.0, 60.0, 240.0
            )
            tuning.onset_quiet_sec = _clamp(tuning.onset_quiet_sec + 2.0, 6.0, 25.0)
            changes.append("Look further back for fight start (full HP bar onset)")
        if any(k in notes_l for k in ("early", "before", "cutscene", "first half")):
            tuning.cutscene_preroll_sec = _clamp(
                tuning.cutscene_preroll_sec + 8.0, 20.0, 90.0
            )
            changes.append("Allow more time before the boss bar for cutscenes")
        if any(k in notes_l for k in ("end", "long", "enemy felled", "died", "ran")):
            tuning.end_offset_sec = _clamp(tuning.end_offset_sec + 2.0, -15.0, 25.0)
            tuning.victory_hold_sec = _clamp(tuning.victory_hold_sec + 0.5, 2.0, 8.0)
            changes.append("Extend clip end toward death/victory screens")
        if not changes:
            tuning.start_offset_sec = _clamp(tuning.start_offset_sec - 1.0, -25.0, 25.0)
            changes.append("Adjusted default start timing from feedback")
    elif reason == "split_fight":
        tuning.split_per_attempt = False
        tuning.merge_gap_sec = _clamp(tuning.merge_gap_sec + 8.0, 15.0, 60.0)
        changes.append("Merge nearby attempts into one clip")
    elif reason == "merged_fights":
        tuning.merge_gap_sec = _clamp(tuning.merge_gap_sec - 8.0, 8.0, 60.0)
        tuning.split_per_attempt = True
        changes.append("Split separate boss fights apart")
    elif reason == "other" and notes_l:
        if "bar" in notes_l and "name" in notes_l:
            tuning.require_name_with_bar = True
            changes.append("Require boss bar and name together")
        elif "bar" in notes_l:
            tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score + 0.02, 0.08, 0.5)
            changes.append("Raised boss bar threshold from notes")

    if timing_adjusted and "Adjusted" not in " ".join(changes):
        tuning.start_offset_sec = _clamp(tuning.start_offset_sec - 0.5, -25.0, 25.0)
        changes.append("Recorded manual timing correction")

    if not changes:
        tuning.min_boss_bar_score = _clamp(tuning.min_boss_bar_score + 0.01, 0.08, 0.5)
        changes.append("General false-positive tightening")

    return tuning, changes


def record_feedback(
    data_dir: Path,
    *,
    verdict: str,
    wrong_reason: str = "",
    notes: str = "",
    vod_id: str = "",
    candidate_id: str = "",
    timing_adjusted: bool = False,
) -> dict:
    """Log review feedback for QA. ML detector is trained via frame labels, not threshold nudges."""
    store = load_feedback_store(data_dir)
    changes = [
        "Logged for QA — retrain the ML model after labeling more frames "
        "(cutscenes / other as hard negatives)."
    ]
    store.events.append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "vod_id": vod_id,
            "candidate_id": candidate_id,
            "verdict": verdict,
            "wrong_reason": wrong_reason,
            "notes": notes,
            "timing_adjusted": timing_adjusted,
            "changes": changes,
        }
    )
    if len(store.events) > 500:
        store.events = store.events[-500:]
    save_feedback_store(data_dir, store)
    return {"changes": changes, "mode": "ml"}


def save_guide(data_dir: Path, guide: str) -> dict:
    """Update standing notes (informational only under ML mode)."""
    store = load_feedback_store(data_dir)
    store.guide = guide.strip()
    save_feedback_store(data_dir, store)
    return {
        "guide": store.guide,
        "changes": ["Saved notes (ML detector uses frame labels, not guide thresholds)"],
    }


def feedback_stats(data_dir: Path) -> dict:
    store = load_feedback_store(data_dir)
    events = store.events
    wrong = [e for e in events if e.get("verdict") == "wrong"]
    correct = [e for e in events if e.get("verdict") == "correct"]
    missed = [e for e in events if e.get("verdict") == "missed"]
    by_reason: dict[str, int] = {}
    for e in wrong:
        key = e.get("wrong_reason") or "unknown"
        by_reason[key] = by_reason.get(key, 0) + 1
    recent = events[-1].get("changes", []) if events else []
    return {
        "feedback_count": len(events),
        "correct": len(correct),
        "wrong": len(wrong),
        "missed": len(missed),
        "wrong_by_reason": by_reason,
        "guide": store.guide,
        "recent_changes": recent,
        "hints": recent[:3],
    }


def clear_feedback(data_dir: Path) -> None:
    path = feedback_path(data_dir)
    if path.exists():
        path.unlink()
