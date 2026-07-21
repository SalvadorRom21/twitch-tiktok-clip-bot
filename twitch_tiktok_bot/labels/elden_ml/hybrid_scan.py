"""Terminal-first boss attempt scan.

Primary: find YOU DIED / ENEMY FELLED (OpenCV color screens), walk back to HUD onset.
Optional: blend trained ML scores onto the same frame grid when model.pt exists.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2

from twitch_tiktok_bot.labels.elden_boss_detect import (
    BossFightCandidate,
    BossScanResult,
    EldenTuning,
    FrameSignals,
    _build_reason,
    _candidate_id,
    apply_fight_anchors,
    cluster_attempts_terminal_first,
    load_elden_tuning,
    load_fight_anchors,
    sample_frame_signals,
    supplement_end_probes,
    supplement_start_probes,
    supplement_terminal_probes,
)
from twitch_tiktok_bot.labels.elden_layout import (
    estimate_layout_profile,
    get_active_layout,
    set_active_layout,
)
from twitch_tiktok_bot.labels.elden_ml.infer import model_is_ready, predict_scan_frames


def _progress(cb: Callable[[dict], None] | None, fields: dict) -> None:
    if callable(cb):
        cb(fields)


def blend_ml_into_signals(
    signals: list[FrameSignals],
    work_dir: Path,
    data_dir: Path,
    *,
    on_progress: Callable[[dict], None] | None = None,
) -> list[FrameSignals]:
    """ML assist + onset/terminal verifier on OpenCV scores.

    - Never invent HUD from ML alone (Merica explore false starts).
    - Boost weak OpenCV HUD / near-miss terminals when ML agrees.
    - Veto OpenCV death/victory / weak HUD when ML is confident ``other``
      (Leyndell gold wash, map UI, outdoor false FELLED).
    """
    if not signals or not model_is_ready(data_dir):
        return signals

    _progress(
        on_progress,
        {
            "phase": "ml_blend",
            "message": "ML assist + onset/terminal verifier…",
            "current": 0,
            "total": 1,
            "phase_frac": 0.0,
        },
    )
    try:
        preds = predict_scan_frames(
            work_dir,
            data_dir,
            on_progress=lambda f: _progress(
                on_progress,
                {
                    "phase": "ml_blend",
                    "message": f.get("message") or "Classifying frames…",
                    "current": f.get("current") or 0,
                    "total": f.get("total") or 1,
                    "phase_frac": (
                        float(f.get("current") or 0) / max(1.0, float(f.get("total") or 1))
                    ),
                },
            ),
        )
    except Exception:  # noqa: BLE001 — OpenCV path still works without ML
        return signals

    by_t = {round(p.time_sec, 1): p for p in preds}
    out: list[FrameSignals] = []
    for s in signals:
        p = by_t.get(round(s.time_sec, 1))
        if p is None:
            out.append(s)
            continue
        hud = p.probs.get("boss_hud", 0.0)
        died = p.probs.get("you_died", 0.0)
        felled = p.probs.get("enemy_felled", 0.0)
        other = p.probs.get("other", 0.0)
        death = s.death_screen_score
        victory = s.victory_screen_score
        # Soft ML confirmation for weak OpenCV terminals only (near miss).
        if 0.22 <= death < 0.45 and died >= 0.65:
            death = max(death, 0.48)
        if 0.22 <= victory < 0.50 and felled >= 0.65:
            victory = max(victory, 0.52)
        # Terminal veto — OpenCV gold/fire wash without ML support.
        if victory >= 0.45 and felled < 0.35 and other >= 0.50:
            victory = min(victory, 0.32)
        # Don't veto strong OpenCV YOU DIED (zoomed layouts ML never saw).
        if 0.40 <= death < 0.85 and died < 0.35 and other >= 0.50:
            death = min(death, 0.32)
        dual = float(getattr(s, "dual_boss_bar_score", 0.0) or 0.0)
        opencv_hud = (
            s.boss_bar_score >= 0.22
            or s.boss_name_score >= 0.12
            or dual >= 0.45
        )
        bar = s.boss_bar_score
        name = s.boss_name_score
        # Onset veto — exploration / map mistaken for HUD.
        if opencv_hud and other >= 0.60 and hud < 0.35 and dual < 0.45:
            bar = min(bar, 0.20)
            name = min(name, 0.08)
        elif opencv_hud and hud >= 0.40:
            bar = max(bar, min(1.0, hud * 0.92))
            name = max(name, min(1.0, hud * 0.88))
        out.append(
            FrameSignals(
                time_sec=s.time_sec,
                boss_bar_score=round(bar, 3),
                boss_bar_fill=s.boss_bar_fill,
                boss_name_score=round(name, 3),
                death_screen_score=round(death, 3),
                victory_screen_score=round(victory, 3),
                dual_boss_bar_score=dual,
            )
        )
    return out


def _signals_from_cached_frames(
    work_dir: Path,
    *,
    interval_sec: float = 2.0,
) -> tuple[list[FrameSignals], float]:
    """Re-score boss_scan_frames/*.jpg with OpenCV (no ffmpeg)."""
    frames_dir = work_dir / "boss_scan_frames"
    if not frames_dir.is_dir():
        return [], 0.0
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    signals: list[FrameSignals] = []
    from twitch_tiktok_bot.labels.elden_boss_detect import analyze_frame

    # Bootstrap adaptive layout from a spread of cached frames.
    probe_imgs: list = []
    if paths:
        step = max(1, len(paths) // 8)
        for path in paths[::step][:10]:
            img = cv2.imread(str(path))
            if img is not None:
                probe_imgs.append(img)
        set_active_layout(estimate_layout_profile(probe_imgs))

    for idx, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        t = idx * interval_sec
        signals.append(analyze_frame(img, t))
    duration = signals[-1].time_sec + interval_sec if signals else 0.0
    return signals, duration


def _segments_to_result(
    segments: list[tuple[float, float, dict]],
    *,
    vod_id: str,
    duration: float,
    frame_samples: list[FrameSignals],
    data_dir: Path | None,
) -> BossScanResult:
    candidates: list[BossFightCandidate] = []
    for start, end, meta in segments:
        confidence = min(
            0.95,
            meta.get("avg_boss_score", 0) * 0.2
            + meta.get("peak_boss_bar_score", 0) * 0.15
            + meta.get("peak_boss_name_score", 0) * 0.25
            + meta.get("terminal_score", 0.5) * 0.35
            + (0.1 if meta.get("frame_hits", 0) >= 4 else 0.0),
        )
        reason = _build_reason(meta)
        if meta.get("detector") == "terminal_first":
            reason = f"Terminal-first · {reason}"
        candidates.append(
            BossFightCandidate(
                id=_candidate_id(vod_id, start, end),
                start_sec=round(start, 2),
                end_sec=round(end, 2),
                confidence=round(confidence, 3),
                reason=reason,
                signals=meta,
                peak_boss_bar_score=meta.get("peak_boss_bar_score", 0.0),
                had_death_screen=bool(meta.get("had_death_screen")),
                had_victory_screen=bool(meta.get("had_victory_screen")),
            )
        )
    anchors = load_fight_anchors(data_dir, vod_id)
    candidates = apply_fight_anchors(candidates, anchors, vod_id=vod_id)
    return BossScanResult(
        vod_id=vod_id,
        duration_sec=duration,
        candidates=candidates,
        frame_samples=frame_samples,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )


def scan_boss_fights_hybrid(
    video_path: Path,
    work_dir: Path,
    vod_id: str,
    data_dir: Path,
    *,
    interval_sec: float = 2.0,
    ffmpeg: str = "ffmpeg",
    tuning: EldenTuning | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> BossScanResult:
    """Sample → find terminals → densify → walk back to HUD → emit attempts."""
    tuning = (tuning or load_elden_tuning(data_dir)).apply_builtin_attempt_policy()

    _progress(
        on_progress,
        {
            "phase": "starting",
            "message": "Terminal-first scan — reading video…",
            "current": 0,
            "total": 1,
            "phase_frac": 0.5,
        },
    )
    signals, duration = sample_frame_signals(
        video_path,
        work_dir,
        interval_sec=interval_sec,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    probe_dir = work_dir / "edge_probes"
    signals = supplement_terminal_probes(
        signals,
        video_path,
        probe_dir=probe_dir,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    signals = supplement_start_probes(
        signals,
        video_path,
        tuning,
        probe_dir=probe_dir,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    signals = supplement_end_probes(
        signals,
        video_path,
        tuning,
        probe_dir=probe_dir,
        ffmpeg=ffmpeg,
        on_progress=on_progress,
    )
    signals = blend_ml_into_signals(
        signals, work_dir, data_dir, on_progress=on_progress
    )

    _progress(
        on_progress,
        {
            "phase": "clustering",
            "message": "Walking back from YOU DIED / FELLED to HUD…",
            "current": 0,
            "total": 1,
            "phase_frac": 0.2,
        },
    )
    segments = cluster_attempts_terminal_first(signals, tuning=tuning)
    _progress(
        on_progress,
        {
            "phase": "clustering",
            "message": "Walk-back complete",
            "current": 1,
            "total": 1,
            "phase_frac": 1.0,
        },
    )
    return _segments_to_result(
        segments,
        vod_id=vod_id,
        duration=duration,
        frame_samples=signals,
        data_dir=data_dir,
    )


def rescan_hybrid_from_cached(
    work_dir: Path,
    vod_id: str,
    data_dir: Path,
    *,
    duration_sec: float = 0.0,
    interval_sec: float = 2.0,
    tuning: EldenTuning | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> BossScanResult:
    """Re-score cached 2s grid frames + terminal-first cluster.

    Full scans still run end/terminal probes via ``scan_boss_fights_hybrid``.
    """
    tuning = (tuning or load_elden_tuning(data_dir)).apply_builtin_attempt_policy()
    _progress(
        on_progress,
        {
            "phase": "reanalyze",
            "message": "Re-scoring cached frames…",
            "current": 0,
            "total": 1,
            "phase_frac": 0.0,
        },
    )
    signals, duration = _signals_from_cached_frames(work_dir, interval_sec=interval_sec)
    if duration_sec > 0:
        duration = duration_sec
    _progress(
        on_progress,
        {
            "phase": "reanalyze",
            "message": "Cached frames scored",
            "current": 1,
            "total": 1,
            "phase_frac": 1.0,
        },
    )
    signals = blend_ml_into_signals(
        signals, work_dir, data_dir, on_progress=on_progress
    )
    _progress(
        on_progress,
        {
            "phase": "clustering",
            "message": "Terminal-first re-cluster…",
            "current": 0,
            "total": 1,
            "phase_frac": 0.2,
        },
    )
    segments = cluster_attempts_terminal_first(signals, tuning=tuning)
    _progress(
        on_progress,
        {
            "phase": "clustering",
            "message": "Re-cluster complete",
            "current": 1,
            "total": 1,
            "phase_frac": 1.0,
        },
    )
    return _segments_to_result(
        segments,
        vod_id=vod_id,
        duration=duration,
        frame_samples=signals,
        data_dir=data_dir,
    )
