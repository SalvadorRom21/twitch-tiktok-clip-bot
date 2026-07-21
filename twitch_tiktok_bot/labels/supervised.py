"""Supervised learning store — auto-detections + human verdicts for Elden Ring."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from twitch_tiktok_bot.labels.elden_boss_detect import BossFightCandidate, BossScanResult


VERDICT_CORRECT = "correct"
VERDICT_WRONG = "wrong"
VERDICT_MISSED = "missed"  # user-added fight the bot missed
VERDICT_PENDING = "pending"

WRONG_REASONS = [
    {"id": "no_boss_bar", "label": "No boss health bar visible"},
    {"id": "regular_mobs", "label": "Regular mobs / trash, not a boss"},
    {"id": "exploring", "label": "Exploring / walking, not fighting"},
    {"id": "cutscene", "label": "Cutscene or NPC dialogue"},
    {"id": "invader", "label": "PvP invader, not a boss"},
    {"id": "audio_only", "label": "Loud moment but no boss bar"},
    {"id": "wrong_timing", "label": "Right fight, wrong start/end times"},
    {"id": "split_fight", "label": "Same fight split into multiple clips"},
    {"id": "merged_fights", "label": "Two separate fights merged together"},
    {"id": "other", "label": "Other (see notes)"},
]

CORRECT_LABELS = [
    {"id": "main_boss", "label": "Main / story boss"},
    {"id": "mini_boss", "label": "Mini-boss (named bar)"},
    {"id": "attempt_chain", "label": "Multi-attempt chain (deaths + win)"},
]


def _opt_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass
class CandidateReview:
    candidate_id: str
    verdict: str = VERDICT_PENDING
    wrong_reason: str = ""
    boss_type: str = ""
    notes: str = ""
    reviewed_at: str = ""
    # Human-corrected clip bounds (override the detector's start/end for training).
    start_sec: float | None = None
    end_sec: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CandidateReview:
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            verdict=str(data.get("verdict", VERDICT_PENDING)),
            wrong_reason=str(data.get("wrong_reason", "")),
            boss_type=str(data.get("boss_type", "")),
            notes=str(data.get("notes", "")),
            reviewed_at=str(data.get("reviewed_at", "")),
            start_sec=_opt_float(data.get("start_sec")),
            end_sec=_opt_float(data.get("end_sec")),
        )

    def has_timing_override(self) -> bool:
        return self.start_sec is not None or self.end_sec is not None


@dataclass
class MissedFight:
    """False negative — boss fight the bot did not propose."""

    id: str
    start_sec: float
    end_sec: float
    boss_type: str = ""
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> MissedFight:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            boss_type=str(data.get("boss_type", "")),
            notes=str(data.get("notes", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class SupervisedTrainingStore:
    vod_id: str
    title: str = ""
    source: str = "upload"
    scan_status: str = "idle"  # idle | scanning | done | failed
    scan_error: str = ""
    reviews: list[CandidateReview] = field(default_factory=list)
    missed_fights: list[MissedFight] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "vod_id": self.vod_id,
            "title": self.title,
            "source": self.source,
            "scan_status": self.scan_status,
            "scan_error": self.scan_error,
            "reviews": [r.to_dict() for r in self.reviews],
            "missed_fights": [m.to_dict() for m in self.missed_fights],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SupervisedTrainingStore:
        return cls(
            vod_id=str(data.get("vod_id", "")),
            title=str(data.get("title", "")),
            source=str(data.get("source", "upload")),
            scan_status=str(data.get("scan_status", "idle")),
            scan_error=str(data.get("scan_error", "")),
            reviews=[CandidateReview.from_dict(r) for r in data.get("reviews", [])],
            missed_fights=[
                MissedFight.from_dict(m) for m in data.get("missed_fights", [])
            ],
        )


def supervised_path(work_dir: Path) -> Path:
    return work_dir / "supervised_training.json"


def load_supervised_store(work_dir: Path) -> SupervisedTrainingStore | None:
    path = supervised_path(work_dir)
    if not path.exists():
        return None
    try:
        return SupervisedTrainingStore.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_supervised_store(work_dir: Path, store: SupervisedTrainingStore) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = supervised_path(work_dir)
    path.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return path


def ensure_supervised_store(work_dir: Path, vod_id: str, title: str = "") -> SupervisedTrainingStore:
    store = load_supervised_store(work_dir)
    if store is None:
        store = SupervisedTrainingStore(vod_id=vod_id, title=title)
        save_supervised_store(work_dir, store)
    return store


def review_for(store: SupervisedTrainingStore, candidate_id: str) -> CandidateReview:
    for review in store.reviews:
        if review.candidate_id == candidate_id:
            return review
    review = CandidateReview(candidate_id=candidate_id)
    store.reviews.append(review)
    return review


def export_training_dataset(data_dir: Path, out_path: Path | None = None) -> Path:
    """Aggregate reviewed VODs; merge into cumulative training archive."""
    out_path = out_path or data_dir / "reference" / "elden_ring_supervised.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            for ex in prior.get("examples", []):
                key = f"{ex.get('vod_id')}:{ex.get('candidate_id')}:{ex.get('start_sec')}"
                existing[key] = ex
        except (json.JSONDecodeError, TypeError):
            pass

    examples: list[dict] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        store = load_supervised_store(child)
        scan = _load_scan(child)
        if store is None or scan is None:
            continue
        cand_by_id = {c.id: c for c in scan.candidates}
        for review in store.reviews:
            if review.verdict == VERDICT_PENDING:
                continue
            cand = cand_by_id.get(review.candidate_id)
            if cand is None:
                continue
            final_start = (
                review.start_sec if review.start_sec is not None else cand.start_sec
            )
            final_end = review.end_sec if review.end_sec is not None else cand.end_sec
            ex = {
                "vod_id": store.vod_id,
                "candidate_id": review.candidate_id,
                "start_sec": round(float(final_start), 2),
                "end_sec": round(float(final_end), 2),
                "detected_start_sec": cand.start_sec,
                "detected_end_sec": cand.end_sec,
                "adjusted": review.has_timing_override(),
                "verdict": review.verdict,
                "wrong_reason": review.wrong_reason,
                "boss_type": review.boss_type,
                "notes": review.notes,
                "detector_confidence": cand.confidence,
                "detector_reason": cand.reason,
                "signals": cand.signals,
            }
            key = f"{ex['vod_id']}:{ex['candidate_id']}:{ex['start_sec']}"
            existing[key] = ex
        for missed in store.missed_fights:
            ex = {
                "vod_id": store.vod_id,
                "candidate_id": missed.id,
                "start_sec": missed.start_sec,
                "end_sec": missed.end_sec,
                "detected_start_sec": None,
                "detected_end_sec": None,
                "adjusted": True,
                "verdict": VERDICT_MISSED,
                "wrong_reason": "",
                "boss_type": missed.boss_type,
                "notes": missed.notes,
                "detector_confidence": 0.0,
                "detector_reason": "not detected",
                "signals": {},
            }
            key = f"{ex['vod_id']}:{ex['candidate_id']}:{ex['start_sec']}"
            existing[key] = ex

    examples = sorted(existing.values(), key=lambda e: (e["vod_id"], e["start_sec"]))
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "example_count": len(examples),
        "correct": sum(1 for e in examples if e["verdict"] == VERDICT_CORRECT),
        "wrong": sum(1 for e in examples if e["verdict"] == VERDICT_WRONG),
        "missed": sum(1 for e in examples if e["verdict"] == VERDICT_MISSED),
        "examples": examples,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _load_scan(work_dir: Path) -> BossScanResult | None:
    from twitch_tiktok_bot.labels.elden_boss_detect import load_scan_result

    return load_scan_result(work_dir)


def tuning_hints_from_export(data_dir: Path) -> dict:
    """Simple supervised feedback — suggest threshold tweaks from wrong reasons."""
    export_path = data_dir / "reference" / "elden_ring_supervised.json"
    if not export_path.exists():
        export_path = export_training_dataset(data_dir)
    data = json.loads(export_path.read_text(encoding="utf-8"))
    examples = data.get("examples", [])
    wrong = [e for e in examples if e["verdict"] == VERDICT_WRONG]
    hints: list[str] = []
    no_bar = sum(1 for e in wrong if e.get("wrong_reason") == "no_boss_bar")
    audio = sum(1 for e in wrong if e.get("wrong_reason") == "audio_only")
    if no_bar >= 2:
        hints.append("Raise boss_bar threshold — many false positives had no bar.")
    if audio >= 2:
        hints.append("Require boss_bar AND boss_name together, not audio peaks alone.")
    return {
        "total_reviewed": len(examples),
        "hints": hints,
        "wrong_by_reason": _count_by(wrong, "wrong_reason"),
    }


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        val = item.get(key) or "unknown"
        out[val] = out.get(val, 0) + 1
    return out


def learn_elden_tuning(data_dir: Path) -> tuple["EldenTuning", dict]:
    """Derive detector thresholds from human-labeled examples."""
    from twitch_tiktok_bot.labels.elden_boss_detect import (
        EldenTuning,
        save_elden_tuning,
    )

    export_path = data_dir / "reference" / "elden_ring_supervised.json"
    if not export_path.exists():
        export_training_dataset(data_dir)
    data = json.loads(export_path.read_text(encoding="utf-8"))
    examples = data.get("examples", [])
    correct = [e for e in examples if e["verdict"] == VERDICT_CORRECT]
    wrong = [e for e in examples if e["verdict"] == VERDICT_WRONG]

    correct_names = [
        float(e.get("signals", {}).get("peak_boss_name_score", 0))
        for e in correct
        if float(e.get("signals", {}).get("peak_boss_name_score", 0)) >= 0.5
    ]
    min_name_correct = min(correct_names) if correct_names else 0.12
    learned_min_name = round(max(0.1, min_name_correct * 0.85), 3)
    if len(correct) < 5:
        learned_min_name = min(learned_min_name, 0.55)

    timing_notes = [
        e for e in examples
        if any(
            kw in (e.get("notes") or "").lower()
            for kw in (
                "wrong at the start",
                "towards the end",
                "late start",
                "too late",
                "bar appears",
                "bar pop",
                "cut",
                "second half",
                "first half",
                "middle",
                "not at the start",
                "nothing useful",
                "useless",
                "no boss",
            )
        )
    ]
    long_end_notes = [
        e for e in examples
        if any(
            kw in (e.get("notes") or "").lower()
            for kw in (
                "enemy felled",
                "lasted way longer",
                "kept running",
                "after boss",
                "after the boss",
                "runs too long",
                "nothing useful",
                "useless",
                "middle",
            )
        )
        or e.get("wrong_reason") == "wrong_timing"
    ]
    late_start_notes = [
        e for e in examples
        if e.get("wrong_reason") == "wrong_timing"
        or any(
            kw in (e.get("notes") or "").lower()
            for kw in (
                "middle",
                "second half",
                "late start",
                "too late",
                "not at the start",
                "starts at the very end",
                "very end of",
            )
        )
    ]
    cutscene_notes = [
        e for e in examples
        if any(
            kw in (e.get("notes") or "").lower()
            for kw in ("cutscene", "fire giant", "cinematic")
        )
    ]
    merged_wrong = sum(1 for e in wrong if e.get("wrong_reason") == "merged_fights")
    split_wrong = sum(1 for e in wrong if e.get("wrong_reason") == "split_fight")

    bleed_fps = [
        e
        for e in wrong
        if e.get("wrong_reason") == "no_boss_bar"
        and float(e.get("signals", {}).get("peak_boss_bar_score", 0)) >= 0.5
        and float(e.get("signals", {}).get("peak_boss_name_score", 0)) < 0.55
    ]
    weak_name_fps = [
        e
        for e in wrong
        if e.get("wrong_reason") == "no_boss_bar"
        and float(e.get("signals", {}).get("peak_boss_name_score", 0)) < 0.55
    ]

    start_offset, end_offset, offset_samples = _learn_timing_offsets(examples)

    tuning = EldenTuning(
        min_boss_name_score=learned_min_name,
        min_boss_bar_score=0.18 if weak_name_fps else 0.15,
        require_name_with_bar=True,
        bar_without_name_reject=0.45 if len(bleed_fps) >= 8 else 0.5,
        min_name_when_bar_high=0.18 if bleed_fps else 0.12,
        frame_hit_threshold=0.18,
        min_frame_hits=2,
        trim_at_victory=True,
        victory_hold_sec=3.0,
        max_post_victory_sec=8.0,
        pre_buffer_sec=1.0,
        start_name_threshold=0.85 if timing_notes else 0.75,
        retry_bridge_sec=0.0,
        merge_gap_sec=25.0,
        split_per_attempt=True,
        merge_while_bar_visible=True,
        bar_gone_sec=8.0,
        hud_bridge_sec=25.0,
        min_hud_samples=3 if len(bleed_fps) >= 5 else 2,
        death_hold_sec=4.0,
        onset_quiet_sec=12.0 if late_start_notes else 10.0,
        onset_quiet_name_max=0.4,
        start_bar_lookback_sec=150.0 if len(late_start_notes) >= 8 else 120.0,
        start_bar_min_score=0.28,
        full_bar_min_score=0.45,
        full_bar_fill_min=0.68,
        cutscene_preroll_sec=50.0 if cutscene_notes else 40.0,
        end_on_death=True,
        min_fight_sec=8.0,
        min_clip_sec=10.0,
        min_keep_sec=6.0,
        max_clip_sec=0.0,
        start_offset_sec=start_offset,
        end_offset_sec=end_offset,
        learned_from=len(examples),
    )

    meta = {
        "learned_at": datetime.now(timezone.utc).isoformat(),
        "from_correct": len(correct),
        "from_wrong": len(wrong),
        "timing_adjustments": len(timing_notes),
        "late_start_adjustments": len(late_start_notes),
        "cutscene_adjustments": len(cutscene_notes),
        "timing_offset_samples": offset_samples,
        "learned_start_offset_sec": start_offset,
        "learned_end_offset_sec": end_offset,
        "rules": [
            "Reject high red bar without white boss name (bleeding bar / UI noise)",
            "Require boss name AND red health bar together for boss HUD",
            "One clip per attempt — start at full HP bar (+ cutscene)",
            "End each attempt on YOU DIED or ENEMY FELLED",
            "No max clip length — clips run until the attempt ends",
            f"Learned start offset {start_offset:+.1f}s, end offset {end_offset:+.1f}s from {offset_samples} corrections",
        ],
    }
    save_elden_tuning(data_dir, tuning, meta=meta)
    return tuning, meta


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _learn_timing_offsets(
    examples: list[dict],
    *,
    clamp_sec: float = 25.0,
    min_samples: int = 3,
) -> tuple[float, float, int]:
    """Median start/end shift between the detector's bounds and your corrections."""
    start_deltas: list[float] = []
    end_deltas: list[float] = []
    for e in examples:
        if not e.get("adjusted"):
            continue
        ds = e.get("detected_start_sec")
        de = e.get("detected_end_sec")
        if ds is not None:
            try:
                start_deltas.append(float(e["start_sec"]) - float(ds))
            except (KeyError, TypeError, ValueError):
                pass
        if de is not None:
            try:
                end_deltas.append(float(e["end_sec"]) - float(de))
            except (KeyError, TypeError, ValueError):
                pass

    samples = max(len(start_deltas), len(end_deltas))
    start_offset = _median(start_deltas) if len(start_deltas) >= min_samples else 0.0
    end_offset = _median(end_deltas) if len(end_deltas) >= min_samples else 0.0
    start_offset = round(max(-clamp_sec, min(clamp_sec, start_offset)), 2)
    end_offset = round(max(-clamp_sec, min(clamp_sec, end_offset)), 2)
    return start_offset, end_offset, samples


def _segment_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return overlap / union if union > 0 else 0.0


def _remap_reviews_after_rescan(
    store: SupervisedTrainingStore,
    prior: BossScanResult,
    result: BossScanResult,
) -> int:
    """Match old reviews to new candidates by time overlap."""
    old_by_id = {c.id: c for c in prior.candidates}
    remapped: list[CandidateReview] = []
    for review in store.reviews:
        if review.verdict == VERDICT_PENDING:
            continue
        old = old_by_id.get(review.candidate_id)
        if old is None:
            continue
        best: BossFightCandidate | None = None
        best_iou = 0.0
        for nc in result.candidates:
            iou = _segment_iou(old.start_sec, old.end_sec, nc.start_sec, nc.end_sec)
            if iou > best_iou:
                best_iou = iou
                best = nc
        if best is not None and best_iou >= 0.25:
            remapped.append(
                CandidateReview(
                    candidate_id=best.id,
                    verdict=review.verdict,
                    wrong_reason=review.wrong_reason,
                    boss_type=review.boss_type,
                    notes=review.notes,
                    reviewed_at=review.reviewed_at,
                    start_sec=review.start_sec,
                    end_sec=review.end_sec,
                )
            )
    store.reviews = remapped
    return len(remapped)


def retune_all_vods(
    data_dir: Path,
    *,
    clear_reviews: bool = True,
    on_progress: object | None = None,
) -> list[dict]:
    """Re-score cached frames and re-cluster with terminal-first hybrid."""
    from twitch_tiktok_bot.labels.elden_boss_detect import (
        load_scan_result,
        save_scan_result,
    )
    from twitch_tiktok_bot.labels.elden_ml.hybrid_scan import rescan_hybrid_from_cached

    def _progress(fields: dict) -> None:
        if callable(on_progress):
            on_progress(fields)

    children = [
        child
        for child in sorted(data_dir.iterdir())
        if child.is_dir()
        and child.name != "reference"
        and (
            load_scan_result(child) is not None
            or (child / "boss_scan_frames").is_dir()
        )
    ]
    reports: list[dict] = []
    total_vods = max(1, len(children))
    for vod_i, child in enumerate(children):
        prior = load_scan_result(child)
        vod_id = prior.vod_id if prior else child.name
        duration = prior.duration_sec if prior else 0.0
        before = len(prior.candidates) if prior else 0
        base_pct = (vod_i / total_vods) * 100.0
        span = 100.0 / total_vods
        _progress(
            {
                "phase": "reanalyze",
                "message": f"Hybrid re-scan {vod_id[:8]}… ({vod_i + 1}/{total_vods})",
                "current": vod_i,
                "total": total_vods,
                "pct": round(base_pct + 2, 1),
            }
        )

        def _infer_progress(fields: dict, _base=base_pct, _span=span, _vid=vod_id) -> None:
            local = float(fields.get("pct") or 0) / 100.0
            _progress(
                {
                    **fields,
                    "message": f"{_vid[:8]}… {fields.get('message', '')}",
                    "pct": round(_base + local * _span * 0.95, 1),
                }
            )

        result = rescan_hybrid_from_cached(
            child,
            vod_id,
            data_dir,
            duration_sec=duration,
            on_progress=_infer_progress,
        )
        save_scan_result(child, result)
        store = load_supervised_store(child)
        remapped = 0
        if store is not None:
            if clear_reviews:
                store.reviews = []
            elif prior is not None:
                remapped = _remap_reviews_after_rescan(store, prior, result)
            store.scan_status = "done"
            store.scan_error = ""
            save_supervised_store(child, store)
        reports.append(
            {
                "vod_id": vod_id,
                "before": before,
                "after": len(result.candidates),
                "reviews_kept": remapped,
                "reviews_cleared": clear_reviews,
            }
        )
        _progress(
            {
                "phase": "reanalyze",
                "message": f"Finished {vod_id[:8]} ({vod_i + 1}/{total_vods})",
                "current": vod_i + 1,
                "total": total_vods,
                "pct": round(((vod_i + 1) / total_vods) * 100.0, 1),
            }
        )
    return reports


_ELDEN_SCAN_CACHE_DIRS = (
    "boss_scan_frames",
    "edge_probes",
    "end_probe",
    "review_frames",
)


def reset_elden_training(
    data_dir: Path,
    *,
    clear_scan_cache: bool = True,
) -> dict:
    """Wipe ML labels/model, feedback, reviews, and scans. Uploaded videos are kept."""
    from twitch_tiktok_bot.labels.elden_feedback import clear_feedback
    from twitch_tiktok_bot.labels.elden_ml.config import MLConfig, save_ml_config
    from twitch_tiktok_bot.labels.elden_ml.dataset import clear_ml_dataset
    from twitch_tiktok_bot.labels.elden_ml.infer import clear_model_cache

    ref = data_dir / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    clear_feedback(data_dir)
    ml_cleared = clear_ml_dataset(data_dir, keep_config=False)
    save_ml_config(
        data_dir,
        MLConfig.defaults(),
        meta={
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "mode": "ml",
            "rules": [
                "Label frames: other / boss_hud / you_died / enemy_felled",
                "Train model, then scan — one attempt = HUD → death or felled",
            ],
        },
    )
    clear_model_cache()

    # Remove legacy heuristic tuning file if present.
    legacy = ref / "elden_ring_tuning.json"
    if legacy.exists():
        legacy.unlink()

    vods_reset: list[dict] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir() or child.name == "reference":
            continue
        scan_path = child / "boss_scan.json"
        store = load_supervised_store(child)
        if store is None and not scan_path.exists():
            continue

        removed_scan = False
        if scan_path.exists():
            scan_path.unlink()
            removed_scan = True

        cache_dirs: list[str] = []
        if clear_scan_cache:
            for dirname in _ELDEN_SCAN_CACHE_DIRS:
                cache_path = child / dirname
                if cache_path.is_dir():
                    shutil.rmtree(cache_path, ignore_errors=True)
                    cache_dirs.append(dirname)

        if store is not None:
            store.reviews = []
            store.missed_fights = []
            store.scan_status = "idle"
            store.scan_error = ""
            save_supervised_store(child, store)

        vods_reset.append(
            {
                "vod_id": child.name,
                "scan_cleared": removed_scan,
                "cache_dirs_removed": cache_dirs,
            }
        )

    return {
        "vods_reset": vods_reset,
        "ml_cleared": ml_cleared,
        "mode": "ml",
        "feedback_cleared": True,
        "vod_count": len(vods_reset),
    }


def _safe_vod_id(vod_id: str) -> bool:
    import re

    return bool(re.fullmatch(r"[\w-]{4,64}", vod_id))


def is_train_vod(work_dir: Path) -> bool:
    return load_supervised_store(work_dir) is not None


def delete_train_vod(data_dir: Path, vod_id: str) -> dict:
    """Delete one uploaded training VOD folder (video + scans + reviews)."""
    if not _safe_vod_id(vod_id):
        raise ValueError("Invalid VOD id")
    work_dir = data_dir / vod_id
    if not work_dir.is_dir():
        raise FileNotFoundError(vod_id)
    if not is_train_vod(work_dir):
        raise ValueError("Not a training upload")
    shutil.rmtree(work_dir, ignore_errors=True)
    return {"vod_id": vod_id, "deleted": True}


def delete_all_train_vods(data_dir: Path) -> dict:
    """Remove every Elden trainer upload under data/."""
    deleted: list[str] = []
    if not data_dir.exists():
        return {"deleted": deleted, "count": 0}
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir() or child.name == "reference":
            continue
        if not is_train_vod(child):
            continue
        shutil.rmtree(child, ignore_errors=True)
        deleted.append(child.name)
    return {"deleted": deleted, "count": len(deleted)}

