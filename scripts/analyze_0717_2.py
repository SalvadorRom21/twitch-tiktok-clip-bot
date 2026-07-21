"""Deep timeline analysis of golden clip 0717(2) — missed by scanner."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from twitch_tiktok_bot.labels.elden_boss_detect import (
    _boss_hud_present,
    _find_hud_runs,
    _mist_door_score,
    _resolve_attempt_terminal,
    _score_player_hp_depleted,
    analyze_frame,
    cluster_attempts_terminal_first,
    collect_terminal_events,
    load_elden_tuning,
)

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "5f8921012661" / "5f8921012661.mp4"
OUT = ROOT / "data" / "5f8921012661" / "analysis_0717_2"


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
            mist.append(
                (
                    t,
                    round(_mist_door_score(frame), 3),
                    round(_score_player_hp_depleted(frame), 3),
                )
            )
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

    print("--- interesting rows ---")
    for r in rows:
        if (
            r["mist"] >= 0.3
            or r["bar"] >= 0.25
            or r["hud"]
            or r["death"] >= 0.3
            or r["victory"] >= 0.3
            or r["name"] >= 0.3
        ):
            print(
                f"{r['t']:6.2f} bar={r['bar']:.3f} fill={r['fill']:.3f} name={r['name']:.3f} "
                f"d={r['death']:.3f} v={r['victory']:.3f} mist={r['mist']:.3f} "
                f"hpE={r['hp_empty']:.2f} hud={r['hud']}"
            )

    terms = collect_terminal_events(signals)
    print("terminals", terms)
    ordered = sorted(signals, key=lambda s: s.time_sec)
    runs = _find_hud_runs(ordered, tuning)
    print("hud_runs", [(a, b, round(b - a, 2)) for a, b, _ in runs])
    for a, b, _ in runs:
        print("  resolve", a, b, _resolve_attempt_terminal(ordered, a, b))
    segs = cluster_attempts_terminal_first(signals, tuning=tuning)
    print(
        "segments",
        [
            (a, b, m.get("ml_end_kind"), m.get("terminal_score"), m.get("hud_onset_sec"), m.get("detector"))
            for a, b, m in segs
        ],
    )

    # Storyboard key times
    cap = cv2.VideoCapture(str(VIDEO))
    picks = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, max(0.0, dur - 0.5)]
    # also peaks
    for key, pred in [
        ("mist", lambda r: r["mist"] >= 0.35),
        ("hud", lambda r: r["bar"] >= 0.34 or r["hud"]),
        ("death", lambda r: r["death"] >= 0.45),
        ("victory", lambda r: r["victory"] >= 0.45),
        ("name", lambda r: r["name"] >= 0.4),
    ]:
        hit = next((r for r in rows if pred(r)), None)
        if hit:
            picks.append(hit["t"])
            print(f"first_{key}", hit)

    for tsec in sorted(set(round(x, 2) for x in picks)):
        if tsec > dur:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000.0)
        ok, frame = cap.read()
        if ok:
            path = OUT / f"key_{tsec:06.2f}.jpg"
            cv2.imwrite(str(path), frame)
    for tsec in range(0, int(dur) + 1, 2):
        cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000.0)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(OUT / f"story_{tsec:03d}.jpg"), frame)
    cap.release()
    print("done", OUT)


if __name__ == "__main__":
    main()
