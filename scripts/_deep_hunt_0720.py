"""Dense HUD hunt on 0720 VOD to tighten fight windows after deep review."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twitch_tiktok_bot.labels.elden_boss_detect import analyze_frame  # noqa: E402

VOD = ROOT / "data" / "1fbc9fdef6a0" / "1fbc9fdef6a0.mp4"
OUT = ROOT / "data" / "reference" / "clip_reviews" / "1fbc9fdef6a0_deep"
OUT.mkdir(parents=True, exist_ok=True)

# Broad windows to re-hunt after review found loose/wrong prior bounds
RANGES = [
    ("godrick_area", 480, 900, 2.0),
    ("cavalry", 2050, 2300, 2.0),
    ("golem", 3950, 4320, 2.0),
    ("shade_area", 6300, 6700, 2.0),
    ("deathbird", 7920, 8040, 2.0),
    ("runebear", 8160, 8380, 2.0),
    ("malenia1", 8400, 8660, 2.0),
    ("malenia2", 8640, 8820, 2.0),
    # Extra: maybe Soldier of Godrick earlier in tutorial
    ("early_cave", 200, 560, 4.0),
]


def main() -> None:
    if not VOD.exists():
        # fallback glob
        found = list((ROOT / "data" / "1fbc9fdef6a0").glob("*.mp4"))
        video = found[0] if found else None
        if video is None:
            raise SystemExit(f"missing video {VOD}")
    else:
        video = VOD

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    summary: dict = {"video": str(video), "ranges": {}}

    for name, a, b, step in RANGES:
        hits = []
        t = float(a)
        while t <= b:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                t += step
                continue
            s = analyze_frame(frame, float(t))
            keep = (
                s.boss_name_score >= 0.35
                or s.boss_bar_score >= 0.55
                or s.death_screen_score >= 0.35
                or s.victory_screen_score >= 0.35
                or s.dual_boss_bar_score >= 0.45
            )
            if keep:
                row = {
                    "t": round(t, 1),
                    "bar": round(s.boss_bar_score, 3),
                    "fill": round(s.boss_bar_fill, 3),
                    "name": round(s.boss_name_score, 3),
                    "dual": round(s.dual_boss_bar_score, 3),
                    "death": round(s.death_screen_score, 3),
                    "vic": round(s.victory_screen_score, 3),
                }
                hits.append(row)
                # save notable frames
                if (
                    s.boss_name_score >= 0.45
                    or s.death_screen_score >= 0.4
                    or s.victory_screen_score >= 0.4
                ):
                    path = OUT / f"{name}_{int(t):05d}.jpg"
                    cv2.imwrite(str(path), frame)
            t += step
        summary["ranges"][name] = hits
        print(f"\n=== {name} ({a}-{b}) hits={len(hits)} ===")
        for h in hits[:40]:
            print(
                f"  t={h['t']:7.1f} bar={h['bar']:.2f} fill={h['fill']:.2f} "
                f"name={h['name']:.2f} dual={h['dual']:.2f} d={h['death']:.2f} v={h['vic']:.2f}"
            )
        if len(hits) > 40:
            print(f"  ... +{len(hits)-40} more")

    (OUT / "dense_hunt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'dense_hunt.json'}")
    cap.release()


if __name__ == "__main__":
    main()
