"""Deep analysis of first Merica VOD detection (dual boss bars + early start)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from twitch_tiktok_bot.labels.elden_boss_detect import (
    FrameSignals,
    _boss_hud_present,
    _find_hud_runs,
    _mist_door_score,
    _resolve_attempt_terminal,
    analyze_frame,
    cluster_attempts_terminal_first,
    collect_terminal_events,
    load_elden_tuning,
)

ROOT = Path(__file__).resolve().parents[1]
VOD = ROOT / "data" / "76cd3cf1089f"
OUT = VOD / "analysis_first_clip"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scan = json.loads((VOD / "boss_scan.json").read_text(encoding="utf-8"))
    cands = scan.get("candidates") or []
    print("dur", scan.get("duration_sec"), "cands", len(cands), "at", scan.get("scanned_at"))
    for i, c in enumerate(cands[:8]):
        sig = c.get("signals") or {}
        print(
            f"#{i} {c['start_sec']:.1f}-{c['end_sec']:.1f} "
            f"({c['end_sec'] - c['start_sec']:.1f}s) kind={sig.get('ml_end_kind')} "
            f"det={sig.get('detector')} hud={sig.get('hud_onset_sec')} "
            f"term={sig.get('terminal_score')} conf={c.get('confidence')}"
        )

    first = cands[0]
    start = float(first["start_sec"])
    end = float(first["end_sec"])
    hud_meta = (first.get("signals") or {}).get("hud_onset_sec")
    print("FIRST", start, end, "hud_meta", hud_meta)

    # Dense window: preroll through first ~90s after start, plus around hud/end
    vid = next(VOD.glob("*.mp4"))
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tuning = load_elden_tuning(ROOT / "data")

    window_start = max(0.0, start - 5.0)
    window_end = min(end + 5.0, start + 180.0)
    # Also sample a bit past claimed start if start is 0
    if start < 5:
        window_end = max(window_end, 120.0)

    rows = []
    t = window_start
    step = 0.5
    while t <= window_end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if ok:
            s = analyze_frame(frame, t)
            mist = _mist_door_score(frame)
            rows.append(
                {
                    "t": float(s.time_sec),
                    "bar": float(s.boss_bar_score),
                    "fill": float(s.boss_bar_fill),
                    "name": float(s.boss_name_score),
                    "death": float(s.death_screen_score),
                    "victory": float(s.victory_screen_score),
                    "mist": float(round(mist, 3)),
                    "hud": bool(_boss_hud_present(s, tuning)),
                }
            )
        t += step

    (OUT / "timeline_first.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # First sustained HUD
    first_hud = next((r for r in rows if r["bar"] >= 0.34 or r["hud"]), None)
    first_mist = next((r for r in rows if r["mist"] >= 0.35), None)
    first_death = next((r for r in rows if r["death"] >= 0.45), None)
    first_vic = next(
        (r for r in rows if r["victory"] >= 0.45 and r["bar"] < 0.12), None
    )
    print("first_mist", first_mist)
    print("first_hud", first_hud)
    print("first_death", first_death)
    print("first_vic", first_vic)

    print("--- interesting ---")
    for r in rows:
        if (
            r["mist"] >= 0.3
            or r["bar"] >= 0.28
            or r["hud"]
            or r["death"] >= 0.35
            or r["victory"] >= 0.35
            or r["name"] >= 0.35
        ):
            print(
                f"{r['t']:7.1f} bar={r['bar']:.3f} fill={r['fill']:.3f} name={r['name']:.3f} "
                f"d={r['death']:.3f} v={r['victory']:.3f} mist={r['mist']:.3f} hud={r['hud']}"
            )

    # Storyboard around claimed start, true HUD, terminal
    picks = [start, start + 5, start + 15, start + 30]
    if first_hud:
        picks += [max(0, first_hud["t"] - 20), max(0, first_hud["t"] - 10), first_hud["t"] - 5, first_hud["t"], first_hud["t"] + 2]
    if first_mist:
        picks.append(first_mist["t"])
    if first_death:
        picks.append(first_death["t"])
    if first_vic:
        picks.append(first_vic["t"])
    picks.append(end - 2)
    for tsec in sorted(set(round(x, 1) for x in picks if x is not None and x >= 0)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tsec * fps)))
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(OUT / f"key_{tsec:07.1f}.jpg"), frame)

    # Full-clip recluster at 1s for context on early segment only (first 8 min)
    signals = []
    t = 0.0
    while t < min(480.0, float(scan.get("duration_sec") or 480)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if ok:
            signals.append(analyze_frame(frame, t))
        t += 1.0
    cap.release()

    ordered = sorted(signals, key=lambda s: s.time_sec)
    runs = _find_hud_runs(ordered, tuning)
    print("early hud_runs", [(a, b, round(b - a, 1)) for a, b, _ in runs[:6]])
    for a, b, _ in runs[:3]:
        print(" resolve", a, b, _resolve_attempt_terminal(ordered, a, b))
    segs = cluster_attempts_terminal_first(signals, tuning=tuning)
    print(
        "early segments",
        [
            (a, b, m.get("ml_end_kind"), m.get("hud_onset_sec"), m.get("detector"))
            for a, b, m in segs[:5]
        ],
    )
    print("terminals", collect_terminal_events(signals)[:12])
    print("done", OUT)


if __name__ == "__main__":
    main()
