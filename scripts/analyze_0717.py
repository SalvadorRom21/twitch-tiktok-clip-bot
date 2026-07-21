"""Deep timeline analysis of golden example clip 0717."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from twitch_tiktok_bot.labels.elden_boss_detect import (
    _boss_hud_present,
    analyze_frame,
    cluster_attempts_terminal_first,
    collect_terminal_events,
    load_elden_tuning,
)

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "abd1650cdef7" / "abd1650cdef7.mp4"
OUT = ROOT / "data" / "abd1650cdef7" / "analysis_0717"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tuning = load_elden_tuning(ROOT / "data")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if fps else 0.0
    print(f"fps={fps:.3f} frames={n} dur={dur:.2f}")

    signals = []
    step = 0.25
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            signals.append(analyze_frame(frame, t))
        t += step
    cap.release()

    rows = []
    for s in signals:
        rows.append(
            {
                "t": s.time_sec,
                "bar": s.boss_bar_score,
                "fill": s.boss_bar_fill,
                "name": s.boss_name_score,
                "death": s.death_screen_score,
                "victory": s.victory_screen_score,
                "hud": bool(_boss_hud_present(s, tuning)),
            }
        )
    (OUT / "timeline.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def first(pred):
        for r in rows:
            if pred(r):
                return r
        return None

    hud_on = first(lambda r: r["bar"] >= 0.34 or r["hud"])
    full_bar = first(lambda r: r["fill"] >= 0.7 and r["bar"] >= 0.34)
    emptyish = None
    if hud_on:
        for r in rows:
            if r["t"] > hud_on["t"] and r["fill"] <= 0.15 and r["bar"] >= 0.2:
                emptyish = r
                break
    victory = first(lambda r: r["victory"] >= 0.45)
    death = first(lambda r: r["death"] >= 0.45)

    print("--- key events ---")
    print("hud_on", hud_on)
    print("full_bar", full_bar)
    print("emptyish_bar", emptyish)
    print("victory", victory)
    print("death", death)

    terms = collect_terminal_events(
        signals, death_threshold=0.45, victory_threshold=0.45
    )
    print("terminals", terms)
    segs = cluster_attempts_terminal_first(signals, tuning=tuning)
    print(
        "segments",
        [
            (a, b, m.get("ml_end_kind"), m.get("terminal_score"), m.get("frame_hits"))
            for a, b, m in segs
        ],
    )

    # Print condensed score track around transitions
    print("--- timeline highlights (bar/fill/name/death/victory) ---")
    for r in rows:
        interesting = (
            r["bar"] >= 0.25
            or r["name"] >= 0.2
            or r["death"] >= 0.25
            or r["victory"] >= 0.25
            or (hud_on and abs(r["t"] - hud_on["t"]) < 1.0)
        )
        if not interesting:
            continue
        print(
            f"{r['t']:6.2f}  bar={r['bar']:.3f} fill={r['fill']:.3f} "
            f"name={r['name']:.3f} d={r['death']:.3f} v={r['victory']:.3f} "
            f"hud={r['hud']}"
        )

    # Key frames for visual review
    cap = cv2.VideoCapture(str(VIDEO))
    picks: list[tuple[str, float]] = [("t00", 0.0)]
    if hud_on:
        picks.append(("pre_hud", max(0.0, hud_on["t"] - 2.0)))
        picks.append(("hud_on", hud_on["t"]))
    if full_bar:
        picks.append(("full_bar", full_bar["t"]))
    if emptyish:
        picks.append(("empty_bar", emptyish["t"]))
    if victory:
        picks.append(("victory", victory["t"]))
    if death:
        picks.append(("death", death["t"]))
    picks.append(("end", max(0.0, dur - 0.5)))

    for label, tsec in picks:
        nearest = min(rows, key=lambda x: abs(x["t"] - tsec))
        cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        path = OUT / f"{label}_{tsec:06.2f}.jpg"
        cv2.imwrite(str(path), frame)
        print(
            f"saved {path.name} bar={nearest['bar']} fill={nearest['fill']} "
            f"name={nearest['name']} d={nearest['death']} v={nearest['victory']}"
        )
    cap.release()
    print("done", OUT)


if __name__ == "__main__":
    main()
