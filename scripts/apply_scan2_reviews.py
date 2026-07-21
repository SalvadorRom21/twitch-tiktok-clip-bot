"""Import scan2 agent verdicts into ML labels and optionally retrain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twitch_tiktok_bot.labels.elden_ml.clips import read_clips, save_manual_clip  # noqa: E402
from twitch_tiktok_bot.labels.elden_ml.dataset import label_counts  # noqa: E402
from twitch_tiktok_bot.labels.elden_ml.infer import clear_model_cache  # noqa: E402
from twitch_tiktok_bot.labels.elden_ml.train import train_model  # noqa: E402

MERICA = "76cd3cf1089f"
TAG = "scan2-review"


def _already(data_dir: Path, note_tag: str) -> bool:
    return any(note_tag in str(c.get("notes") or "") for c in read_clips(data_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=ROOT / "data" / "reference" / "clip_reviews" / "agent_verdicts_scan2.json",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    scan = json.loads((args.data_dir / MERICA / "boss_scan.json").read_text(encoding="utf-8"))
    cands = scan.get("candidates") or []

    added = 0
    frames = 0
    for row in verdicts.get("merica") or []:
        idx = int(row["index"])
        if idx >= len(cands):
            continue
        kind = row.get("train_as") or "not_fight"
        note_tag = f"{TAG} Merica#{idx}"
        if _already(args.data_dir, note_tag):
            print(f"skip {note_tag}")
            continue
        cand = cands[idx]
        start = float(cand.get("start_sec") or 0)
        end = float(cand.get("end_sec") or start + 1)
        if kind == "not_fight" and end - start > 90.0:
            mid = (start + end) / 2.0
            start, end = mid - 45.0, mid + 45.0
        notes = f"{note_tag}: {row.get('verdict')} - {row.get('notes')}"
        print(f"add {note_tag} {kind} {start:.1f}-{end:.1f}")
        result = save_manual_clip(
            args.data_dir,
            vod_id=MERICA,
            start_sec=start,
            end_sec=end,
            kind=kind,
            notes=notes,
            extract_video=False,
            expand_frames=True,
        )
        added += 1
        frames += int(result.get("frame_labels_added") or 0)

    # Reinforce goldens if not already tagged from this pass.
    for g in verdicts.get("goldens") or []:
        vod = g["vod_id"]
        note_tag = f"{TAG} golden-{vod}"
        if _already(args.data_dir, note_tag):
            print(f"skip {note_tag}")
            continue
        gscan = args.data_dir / vod / "boss_scan.json"
        if not gscan.exists():
            continue
        gcands = json.loads(gscan.read_text(encoding="utf-8")).get("candidates") or []
        if not gcands:
            continue
        c = gcands[0]
        kind = g.get("train_as") or "you_died"
        result = save_manual_clip(
            args.data_dir,
            vod_id=vod,
            start_sec=float(c.get("start_sec") or 0),
            end_sec=float(c.get("end_sec") or 1),
            kind=kind,
            notes=f"{note_tag}: correct golden",
            extract_video=False,
            expand_frames=True,
        )
        added += 1
        frames += int(result.get("frame_labels_added") or 0)
        print(f"add {note_tag} {kind}")

    print({"clips_added": added, "frames_added": frames, "counts": label_counts(args.data_dir)})
    if args.train and added:
        metrics = train_model(args.data_dir, epochs=args.epochs)
        clear_model_cache()
        print("train", metrics)


if __name__ == "__main__":
    main()
