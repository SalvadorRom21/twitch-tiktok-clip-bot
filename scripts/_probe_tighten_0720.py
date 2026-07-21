"""Extract targeted review frames for tightened 0720 windows."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twitch_tiktok_bot.labels.elden_boss_detect import analyze_frame  # noqa: E402

VOD_DIR = ROOT / "data" / "1fbc9fdef6a0"
video = next(VOD_DIR.glob("*.mp4"))
OUT = ROOT / "data" / "reference" / "clip_reviews" / "1fbc9fdef6a0_deep"
OUT.mkdir(parents=True, exist_ok=True)

# Fine probes around confirmed moments
PROBES = [
    # cavalry: find last bar / felled
    *[(f"cav_{t}", t) for t in range(2070, 2140, 2)],
    # golem: find first named bar before FELLED
    *[(f"gol_{t}", t) for t in (4200, 4220, 4240, 4250, 4260, 4265, 4270, 4272, 4275, 4280, 4285, 4287, 4290)],
    # shade: around confirmed Cemetery Shade
    *[(f"shd_{t}", t) for t in range(6475, 6535, 2)],
    # deathbird tight
    *[(f"db_{t}", t) for t in range(7960, 8010, 2)],
    # runebear tight
    *[(f"rb_{t}", t) for t in range(8230, 8350, 4)],
    # malenia goddess phase
    *[(f"mal2_{t}", t) for t in range(8680, 8780, 2)],
    # soldier of godrick hunt in cave of knowledge style
    *[(f"sog_{t}", t) for t in range(300, 700, 5)],
]


def main() -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for label, t in PROBES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        s = analyze_frame(frame, float(t))
        notable = (
            s.boss_name_score >= 0.4
            or s.death_screen_score >= 0.35
            or s.victory_screen_score >= 0.35
            or label.startswith(("gol_427", "gol_428", "cav_207", "cav_208", "cav_209", "shd_649", "mal2_87"))
        )
        if notable:
            path = OUT / f"probe_{label}.jpg"
            cv2.imwrite(str(path), frame)
            print(
                f"{label} t={t} name={s.boss_name_score:.2f} bar={s.boss_bar_score:.2f} "
                f"fill={s.boss_bar_fill:.2f} d={s.death_screen_score:.2f} v={s.victory_screen_score:.2f}"
            )
    cap.release()


if __name__ == "__main__":
    main()
