"""Extract labeled frames from a training VOD for human review."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "elden_boss_detect",
    ROOT / "twitch_tiktok_bot" / "labels" / "elden_boss_detect.py",
)
ebd = importlib.util.module_from_spec(spec)
sys.modules["elden_boss_detect"] = ebd
assert spec.loader is not None
spec.loader.exec_module(ebd)
_boss_bar_band = ebd._boss_bar_band
_boss_name_band = ebd._boss_name_band
_gameplay_roi = ebd._gameplay_roi

VOD = ROOT / "data" / "be951f9be832" / "be951f9be832.mp4"
OUT = ROOT / "data" / "be951f9be832" / "review_frames"

# (time_sec, label, bot_guess)
REVIEW_POINTS = [
    (10670, "ghost_bird_bar_start", "maybe_boss — name+bar detected"),
    (10692, "ghost_bird_bar_again", "maybe_boss — paired HUD"),
    (10730, "ghost_bird_name_only", "maybe_boss — name only, no bar"),
    (10826, "ghost_bird_victory", "victory banner score 0.73"),
    (1876, "clip1_fight_start", "correct — boss bar appears"),
    (1964, "clip1_enemy_felled", "correct — ENEMY FELLED"),
    (1969, "clip1_clip_end", "current clip end (+5s after victory)"),
    (4234, "fp_bleed_bar", "wrong — bot flagged, no boss bar"),
    (5394, "fp_victory_noise", "wrong — victory + weak bar"),
    (5480, "fp_bar_no_name", "wrong — red bar, low name score"),
]


def extract_frame(time_sec: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        str(VOD),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0 and out_path.exists()


def main() -> None:
    import cv2

    manifest = []
    for time_sec, label, guess in REVIEW_POINTS:
        base = OUT / f"{int(time_sec):05d}_{label}"
        full_path = base.with_name(base.name + "_full.jpg")
        if not extract_frame(time_sec, full_path):
            print("failed", time_sec, label)
            continue
        img = cv2.imread(str(full_path))
        if img is None:
            continue
        x, y, w, h = _gameplay_roi(img)
        gameplay = img[y : y + h, x : x + w]
        cv2.imwrite(str(base.with_name(base.name + "_gameplay.jpg")), gameplay)
        cv2.imwrite(str(base.with_name(base.name + "_boss_bar.jpg")), _boss_bar_band(img))
        cv2.imwrite(str(base.with_name(base.name + "_boss_name.jpg")), _boss_name_band(img))
        manifest.append(
            {
                "time_sec": time_sec,
                "label": label,
                "bot_guess": guess,
                "files": {
                    "full": str(full_path.relative_to(ROOT)).replace("\\", "/"),
                    "gameplay": str(
                        base.with_name(base.name + "_gameplay.jpg").relative_to(ROOT)
                    ).replace("\\", "/"),
                },
            }
        )
        print(f"ok {time_sec}s {label}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
