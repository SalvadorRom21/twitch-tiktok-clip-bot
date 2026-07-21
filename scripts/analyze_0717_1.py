"""Deep timeline analysis of golden death example clip 0717(1)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from twitch_tiktok_bot.labels.elden_boss_detect import (
    _boss_hud_present,
    _mist_door_score,
    _score_player_hp_depleted,
    analyze_frame,
    cluster_attempts_terminal_first,
    collect_terminal_events,
    load_elden_tuning,
)

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "516dd3ff9165" / "516dd3ff9165.mp4"
OUT = ROOT / "data" / "516dd3ff9165" / "analysis_0717_1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tuning = load_elden_tuning(ROOT / "data")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if fps else 0.0
    print(f"fps={fps:.3f} frames={n} dur={dur:.2f}")

    signals = []
    mist = []
    step = 0.25
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            sig = analyze_frame(frame, t)
            signals.append(sig)
            mist.append((t, round(_mist_door_score(frame), 3), round(_score_player_hp_depleted(frame), 3)))
        t += step
    cap.release()

    rows = []
    for s, (mt, mscore, hp) in zip(signals, mist):
        rows.append(
            {
                "t": s.time_sec,
                "bar": float(s.boss_bar_score),
                "fill": float(s.boss_bar_fill),
                "name": float(s.boss_name_score),
                "death": float(s.death_screen_score),
                "victory": float(s.victory_screen_score),
                "hud": bool(_boss_hud_present(s, tuning)),
                "mist": mscore,
                "hp_empty": hp,
            }
        )
    (OUT / "timeline.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def first(pred):
        for r in rows:
            if pred(r):
                return r
        return None

    mist_on = first(lambda r: r["mist"] >= 0.35)
    hud_on = first(lambda r: r["bar"] >= 0.34 or r["hud"])
    # HUD gap during fight (phase cinematic): bar was on, then off >=2s, then on again
    phase_gap = None
    saw_hud = False
    gap_start = None
    for r in rows:
        if r["bar"] >= 0.34 or r["hud"]:
            if saw_hud and gap_start is not None and r["t"] - gap_start >= 2.0:
                phase_gap = {"gap_start": gap_start, "gap_end": r["t"], "dur": round(r["t"] - gap_start, 2)}
                break
            saw_hud = True
            gap_start = None
        elif saw_hud and gap_start is None:
            gap_start = r["t"]

    death = first(lambda r: r["death"] >= 0.45)
    victory = first(lambda r: r["victory"] >= 0.45 and r["bar"] < 0.28)

    print("--- key events ---")
    print("mist_door", mist_on)
    print("hud_on", hud_on)
    print("phase_cinematic_gap", phase_gap)
    print("death", death)
    print("victory", victory)

    terms = collect_terminal_events(signals)
    print("terminals", terms[:12], "count", len(terms))
    segs = cluster_attempts_terminal_first(signals, tuning=tuning)
    print(
        "segments",
        [
            (a, b, m.get("ml_end_kind"), m.get("terminal_score"), m.get("hud_onset_sec"), m.get("detector"))
            for a, b, m in segs
        ],
    )

    print("--- highlights ---")
    for r in rows:
        interesting = (
            r["mist"] >= 0.3
            or r["bar"] >= 0.3
            or r["hud"]
            or r["death"] >= 0.35
            or r["victory"] >= 0.35
            or (hud_on and abs(r["t"] - hud_on["t"]) < 1)
            or (death and abs(r["t"] - death["t"]) < 2)
            or (phase_gap and phase_gap["gap_start"] - 1 <= r["t"] <= phase_gap["gap_end"] + 1)
        )
        if not interesting:
            continue
        print(
            f"{r['t']:6.2f} bar={r['bar']:.3f} fill={r['fill']:.3f} name={r['name']:.3f} "
            f"d={r['death']:.3f} v={r['victory']:.3f} mist={r['mist']:.3f} hpE={r['hp_empty']:.2f} hud={r['hud']}"
        )

    # Storyboard
    cap = cv2.VideoCapture(str(VIDEO))
    picks = [("t00", 0.0)]
    if mist_on:
        picks.append(("mist", mist_on["t"]))
    if hud_on:
        picks.append(("pre_hud", max(0.0, hud_on["t"] - 2.0)))
        picks.append(("hud_on", hud_on["t"]))
    if phase_gap:
        picks.append(("phase_gap", phase_gap["gap_start"] + 0.5))
        picks.append(("phase_return", phase_gap["gap_end"]))
    if death:
        picks.append(("death", death["t"]))
        picks.append(("pre_death", max(0.0, death["t"] - 2.0)))
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
            f"d={nearest['death']} v={nearest['victory']} mist={nearest['mist']}"
        )
    # dense story every 3s
    for tsec in range(0, int(dur) + 1, 3):
        cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000.0)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(OUT / f"story_{tsec:03d}.jpg"), frame)
    cap.release()
    print("done", OUT)


if __name__ == "__main__":
    main()
