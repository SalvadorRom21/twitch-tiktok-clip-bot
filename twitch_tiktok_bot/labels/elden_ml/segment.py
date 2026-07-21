"""Temporal attempt segmenter: boss_hud onset → YOU DIED / ENEMY FELLED."""

from __future__ import annotations

from dataclasses import dataclass

from twitch_tiktok_bot.labels.elden_ml.config import MLConfig
from twitch_tiktok_bot.labels.elden_ml.infer import FramePrediction


@dataclass
class AttemptSegment:
    start_sec: float
    end_sec: float
    end_kind: str  # you_died | enemy_felled | incomplete
    hud_frames: int
    peak_hud: float
    peak_terminal: float
    confidence: float


def segment_attempts(
    preds: list[FramePrediction],
    cfg: MLConfig | None = None,
) -> list[AttemptSegment]:
    """Build one clip per attempt from a time-ordered prediction series.

    Start = first sustained boss_hud (min_hud_frames).
    End = first you_died / enemy_felled after onset (+ terminal_hold_sec).
    Terminals without a prior HUD window are ignored (cuts cinematic FPs).
    """
    cfg = cfg or MLConfig.defaults()
    if not preds:
        return []

    ordered = sorted(preds, key=lambda p: p.time_sec)
    segments: list[AttemptSegment] = []
    n = len(ordered)
    i = 0

    while i < n:
        if ordered[i].label != "boss_hud":
            i += 1
            continue

        run_start = i
        while i < n and ordered[i].label == "boss_hud":
            i += 1
        if i - run_start < cfg.min_hud_frames:
            continue

        onset = ordered[run_start].time_sec
        hud_frames = i - run_start
        peak_hud = max(
            ordered[k].probs.get("boss_hud", ordered[k].confidence)
            for k in range(run_start, i)
        )
        peak_terminal = 0.0
        end_kind = "incomplete"
        end_t = ordered[i - 1].time_sec
        j = i

        while j < n:
            lab = ordered[j].label
            conf = ordered[j].confidence
            if lab == "boss_hud":
                hud_frames += 1
                peak_hud = max(peak_hud, ordered[j].probs.get("boss_hud", conf))
                end_t = ordered[j].time_sec
                j += 1
                continue
            if lab in ("you_died", "enemy_felled"):
                end_kind = lab
                peak_terminal = ordered[j].probs.get(lab, conf)
                end_t = ordered[j].time_sec + cfg.terminal_hold_sec
                j += 1
                break
            # other — allow brief flicker; stop if HUD gone without terminal
            gap_t0 = ordered[j].time_sec
            k = j
            revived = False
            terminal_hit = False
            while k < n and ordered[k].time_sec - gap_t0 <= 25.0:
                if ordered[k].label == "boss_hud":
                    j = k
                    revived = True
                    break
                if ordered[k].label in ("you_died", "enemy_felled"):
                    end_kind = ordered[k].label
                    peak_terminal = ordered[k].probs.get(end_kind, ordered[k].confidence)
                    end_t = ordered[k].time_sec + cfg.terminal_hold_sec
                    j = k + 1
                    terminal_hit = True
                    break
                k += 1
            if terminal_hit:
                break
            if revived:
                continue
            end_t = ordered[max(0, j - 1)].time_sec
            end_kind = "incomplete"
            break

        i = max(i, j)
        duration = end_t - onset
        if duration < cfg.min_attempt_sec:
            continue
        if end_kind == "incomplete" and hud_frames < max(cfg.min_hud_frames + 2, 4):
            continue

        conf = min(
            0.95,
            0.35 * peak_hud
            + 0.45 * peak_terminal
            + 0.15 * min(1.0, hud_frames / 8.0)
            + (0.1 if end_kind in ("you_died", "enemy_felled") else 0.0),
        )
        if end_kind == "incomplete":
            conf = min(conf, 0.55)

        segments.append(
            AttemptSegment(
                start_sec=round(max(0.0, onset), 2),
                end_sec=round(max(onset + cfg.min_attempt_sec, end_t), 2),
                end_kind=end_kind,
                hud_frames=hud_frames,
                peak_hud=round(peak_hud, 3),
                peak_terminal=round(peak_terminal, 3),
                confidence=round(conf, 3),
            )
        )

    return _dedupe(segments)


def _dedupe(segments: list[AttemptSegment], iou: float = 0.5) -> list[AttemptSegment]:
    if len(segments) < 2:
        return segments
    ordered = sorted(segments, key=lambda s: s.start_sec)
    kept: list[AttemptSegment] = [ordered[0]]
    for seg in ordered[1:]:
        prev = kept[-1]
        overlap = max(0.0, min(prev.end_sec, seg.end_sec) - max(prev.start_sec, seg.start_sec))
        union = max(prev.end_sec, seg.end_sec) - min(prev.start_sec, seg.start_sec)
        if union > 0 and overlap / union >= iou:
            if seg.confidence > prev.confidence or (
                seg.end_kind != "incomplete" and prev.end_kind == "incomplete"
            ):
                kept[-1] = seg
            continue
        kept.append(seg)
    return kept
