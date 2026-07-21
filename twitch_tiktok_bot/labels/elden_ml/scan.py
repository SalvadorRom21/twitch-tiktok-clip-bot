"""ML scan pipeline: sample frames → classify → segment attempts."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from twitch_tiktok_bot.labels.elden_boss_detect import (
    BossFightCandidate,
    BossScanResult,
    extract_scan_frame_grid,
)
from twitch_tiktok_bot.labels.elden_ml.config import MLConfig, load_ml_config
from twitch_tiktok_bot.labels.elden_ml.infer import (
    FramePrediction,
    model_is_ready,
    predict_scan_frames,
)
from twitch_tiktok_bot.labels.elden_ml.segment import AttemptSegment, segment_attempts


def _candidate_id(vod_id: str, start: float, end: float) -> str:
    key = f"{vod_id}:{start:.1f}:{end:.1f}"
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def sample_scan_frames(
    video_path: Path,
    work_dir: Path,
    *,
    interval_sec: float = 2.0,
    ffmpeg: str = "ffmpeg",
    on_progress: Callable[[dict], None] | None = None,
) -> float:
    """Extract JPEG grid into boss_scan_frames/. Returns duration."""
    _frames_dir, duration, _times = extract_scan_frame_grid(
        video_path,
        work_dir,
        interval_sec=interval_sec,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    return duration


def _preds_to_frame_samples(preds: list[FramePrediction]) -> list:
    """Map ML preds into legacy FrameSignals-shaped dicts for boss_scan.json."""
    from twitch_tiktok_bot.labels.elden_boss_detect import FrameSignals

    samples: list[FrameSignals] = []
    for p in preds:
        samples.append(
            FrameSignals(
                time_sec=p.time_sec,
                boss_bar_score=p.probs.get("boss_hud", 0.0),
                boss_bar_fill=0.0,
                boss_name_score=p.probs.get("boss_hud", 0.0),
                death_screen_score=p.probs.get("you_died", 0.0),
                victory_screen_score=p.probs.get("enemy_felled", 0.0),
            )
        )
    return samples


def attempts_to_candidates(
    vod_id: str,
    attempts: list[AttemptSegment],
) -> list[BossFightCandidate]:
    cands: list[BossFightCandidate] = []
    for a in attempts:
        reason_parts = [f"ML {a.end_kind}", f"{a.hud_frames} HUD frames"]
        if a.end_kind == "you_died":
            reason_parts.append("YOU DIED")
        elif a.end_kind == "enemy_felled":
            reason_parts.append("ENEMY FELLED")
        elif a.end_kind == "incomplete":
            reason_parts.append("in-progress (no terminal yet)")
        cands.append(
            BossFightCandidate(
                id=_candidate_id(vod_id, a.start_sec, a.end_sec),
                start_sec=a.start_sec,
                end_sec=a.end_sec,
                confidence=a.confidence,
                reason="; ".join(reason_parts),
                signals={
                    "peak_boss_bar_score": a.peak_hud,
                    "peak_boss_name_score": a.peak_hud,
                    "had_death_screen": a.end_kind == "you_died",
                    "had_victory_screen": a.end_kind == "enemy_felled",
                    "frame_hits": a.hud_frames,
                    "avg_boss_score": a.peak_hud,
                    "ml_end_kind": a.end_kind,
                    "peak_terminal": a.peak_terminal,
                },
                peak_boss_bar_score=a.peak_hud,
                had_death_screen=a.end_kind == "you_died",
                had_victory_screen=a.end_kind == "enemy_felled",
                source="ml",
            )
        )
    return cands


def scan_boss_fights_ml(
    video_path: Path,
    work_dir: Path,
    vod_id: str,
    data_dir: Path,
    *,
    interval_sec: float | None = None,
    ffmpeg: str = "ffmpeg",
    cfg: MLConfig | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> BossScanResult:
    """Full ML scan: sample → classify → segment attempts."""
    cfg = cfg or load_ml_config(data_dir)
    interval = interval_sec if interval_sec is not None else cfg.sample_interval_sec

    def _progress(fields: dict) -> None:
        if on_progress:
            on_progress(fields)

    if not model_is_ready(data_dir):
        raise FileNotFoundError(
            "No trained ML model yet. Label frames (boss HUD / YOU DIED / ENEMY FELLED / other) "
            "then click Train model before scanning."
        )

    _progress({"phase": "starting", "message": "Sampling video frames…", "pct": 1.0})
    duration = sample_scan_frames(
        video_path,
        work_dir,
        interval_sec=interval,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )

    def _infer_progress(fields: dict) -> None:
        # Map infer 0-100 into 40-95.
        local = float(fields.get("pct") or 0) / 100.0
        _progress(
            {
                **fields,
                "pct": round(40.0 + local * 55.0, 1),
            }
        )

    preds = predict_scan_frames(
        work_dir,
        data_dir,
        interval_sec=interval,
        cfg=cfg,
        on_progress=_infer_progress,
    )
    _progress(
        {
            "phase": "segment",
            "message": "Building attempt clips…",
            "pct": 96.0,
            "current": 1,
            "total": 1,
        }
    )
    attempts = segment_attempts(preds, cfg)
    candidates = attempts_to_candidates(vod_id, attempts)
    return BossScanResult(
        vod_id=vod_id,
        duration_sec=duration,
        candidates=candidates,
        frame_samples=_preds_to_frame_samples(preds),
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )


def rescan_from_cached_frames_ml(
    work_dir: Path,
    vod_id: str,
    data_dir: Path,
    *,
    duration_sec: float = 0.0,
    interval_sec: float | None = None,
    cfg: MLConfig | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> BossScanResult:
    """Re-infer + segment using existing boss_scan_frames (no ffmpeg)."""
    cfg = cfg or load_ml_config(data_dir)
    interval = interval_sec if interval_sec is not None else cfg.sample_interval_sec
    if not model_is_ready(data_dir):
        raise FileNotFoundError("No trained ML model yet.")
    preds = predict_scan_frames(
        work_dir,
        data_dir,
        interval_sec=interval,
        cfg=cfg,
        on_progress=on_progress,
    )
    if duration_sec <= 0 and preds:
        duration_sec = preds[-1].time_sec + interval
    attempts = segment_attempts(preds, cfg)
    return BossScanResult(
        vod_id=vod_id,
        duration_sec=duration_sec,
        candidates=attempts_to_candidates(vod_id, attempts),
        frame_samples=_preds_to_frame_samples(preds),
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )
