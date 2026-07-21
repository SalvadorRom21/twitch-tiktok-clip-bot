"""Find true dual-boss HUD onset vs false early bars on Merica first fight."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from twitch_tiktok_bot.labels.elden_boss_detect import (
    FrameSignals,
    _boss_bar_search_band,
    _score_boss_bar,
    analyze_frame,
    cluster_attempts_terminal_first,
    load_elden_tuning,
)

ROOT = Path(__file__).resolve().parents[1]
VOD = ROOT / "data" / "76cd3cf1089f"
OUT = VOD / "analysis_first_clip"
OUT.mkdir(parents=True, exist_ok=True)


def dual_bar_score(frame: np.ndarray) -> float:
    """Two stacked red boss bars → high score (Crystalian-style)."""
    h, w = frame.shape[:2]
    x0, x1 = int(w * 0.12), int(w * 0.88)
    # Search band region around typical boss bar Y
    y0, y1 = int(h * 0.55), int(h * 0.78)
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)),
        cv2.inRange(hsv, (168, 80, 60), (180, 255, 255)),
    )
    # Per-row coverage; dual bars = two separated peaks of wide red
    coverages = []
    for row in red:
        coverages.append(float(np.count_nonzero(row)) / row.shape[0])
    peaks = [i for i, c in enumerate(coverages) if c >= 0.22]
    if len(peaks) < 2:
        return 0.0
    # Cluster into bands
    bands = [[peaks[0]]]
    for i in peaks[1:]:
        if i - bands[-1][-1] <= 3:
            bands[-1].append(i)
        else:
            bands.append([i])
    wide = [b for b in bands if len(b) >= 1 and max(coverages[j] for j in b) >= 0.22]
    if len(wide) < 2:
        return 0.0
    # Vertical separation between first two bands
    gap = wide[1][0] - wide[0][-1]
    if gap < 2 or gap > 40:
        return 0.0
    return min(1.0, 0.55 + 0.1 * len(wide) + max(coverages) * 0.4)


def main() -> None:
    # Cached scan early samples
    scan = json.loads((VOD / "boss_scan.json").read_text(encoding="utf-8"))
    frames = scan.get("frame_samples") or []
    print("=== cached early samples (bar>=0.25 or death/vic) ===")
    for f in frames:
        if f["time_sec"] > 700:
            break
        if (
            f.get("boss_bar_score", 0) >= 0.25
            or f.get("death_screen_score", 0) >= 0.4
            or f.get("victory_screen_score", 0) >= 0.4
            or f.get("boss_name_score", 0) >= 0.3
        ):
            print(
                f"t={f['time_sec']:6.1f} bar={f['boss_bar_score']:.2f} "
                f"fill={f['boss_bar_fill']:.2f} name={f['boss_name_score']:.2f} "
                f"d={f['death_screen_score']:.2f} v={f['victory_screen_score']:.2f}"
            )

    vid = next(VOD.glob("*.mp4"))
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print("=== live 1s scan 0-700s ===")
    first_strong = None
    first_dual = None
    first_name = None
    death_t = None
    for t in range(0, 701):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        s = analyze_frame(frame, float(t))
        dual = dual_bar_score(frame)
        if first_strong is None and s.boss_bar_score >= 0.34:
            first_strong = (t, s, dual)
        if first_dual is None and dual >= 0.55:
            first_dual = (t, s, dual)
        if first_name is None and s.boss_name_score >= 0.35:
            first_name = (t, s, dual)
        if death_t is None and s.death_screen_score >= 0.5:
            death_t = (t, s)
        if (
            s.boss_bar_score >= 0.34
            or dual >= 0.5
            or s.boss_name_score >= 0.35
            or s.death_screen_score >= 0.45
            or (60 <= t <= 200 and t % 10 == 0)
            or (600 <= t <= 670 and t % 5 == 0)
        ):
            if s.boss_bar_score >= 0.3 or dual >= 0.45 or s.death_screen_score >= 0.4 or s.boss_name_score >= 0.3:
                print(
                    f"t={t:4d} bar={s.boss_bar_score:.2f} fill={s.boss_bar_fill:.2f} "
                    f"name={s.boss_name_score:.2f} d={s.death_screen_score:.2f} dual={dual:.2f}"
                )
                if dual >= 0.5 or (first_strong and abs(t - first_strong[0]) < 3) or (death_t and abs(t - death_t[0]) < 3):
                    cv2.imwrite(str(OUT / f"t_{t:04d}.jpg"), frame)
                    _, y0, y1 = _boss_bar_search_band(frame)
                    band = frame[y0:y1, int(frame.shape[1] * 0.12) : int(frame.shape[1] * 0.88)]
                    cv2.imwrite(str(OUT / f"band_{t:04d}.jpg"), band)

    print("first_strong", first_strong[0] if first_strong else None)
    print("first_dual", first_dual[0] if first_dual else None)
    print("first_name", first_name[0] if first_name else None)
    print("death", death_t[0] if death_t else None)

    # Ideal start = max(0, hud - 30) .. hud-5
    if first_dual or first_strong:
        hud = (first_dual or first_strong)[0]
        print(f"ideal start window: {max(0, hud - 30)} .. {max(0, hud - 5)} (hud~{hud})")

    cap.release()
    print("done", OUT)


if __name__ == "__main__":
    main()
