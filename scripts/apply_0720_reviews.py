"""Import 0720 zoomed-VOD agent verdicts into ML labels and retrain."""

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

VOD = "1fbc9fdef6a0"
TAG = "0720-review"


def _already(data_dir: Path, note_tag: str) -> bool:
    return any(note_tag in str(c.get("notes") or "") for c in read_clips(data_dir))


def _add(
    data_dir: Path,
    *,
    start: float,
    end: float,
    kind: str,
    note_tag: str,
    notes: str,
) -> tuple[int, int]:
    if _already(data_dir, note_tag):
        print(f"skip {note_tag}")
        return 0, 0
    if end - start > 120.0 and kind == "not_fight":
        mid = (start + end) / 2.0
        start, end = mid - 55.0, mid + 55.0
    print(f"add {note_tag} {kind} {start:.1f}-{end:.1f}")
    result = save_manual_clip(
        data_dir,
        vod_id=VOD,
        start_sec=start,
        end_sec=end,
        kind=kind,
        notes=f"{note_tag}: {notes}",
        extract_video=False,
        expand_frames=True,
    )
    return 1, int(result.get("frame_labels_added") or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=ROOT / "data" / "reference" / "clip_reviews" / "agent_verdicts_0720.json",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    added = 0
    frames = 0

    for i, row in enumerate(verdicts.get("auto_candidates") or []):
        note_tag = f"{TAG} auto#{row.get('index', i)}"
        a, f = _add(
            args.data_dir,
            start=float(row["start_sec"]),
            end=float(row["end_sec"]),
            kind=str(row["train_as"]),
            note_tag=note_tag,
            notes=str(row.get("notes") or row.get("verdict") or ""),
        )
        added += a
        frames += f

    for i, row in enumerate(verdicts.get("missed_fights") or []):
        boss = str(row.get("boss") or f"fight{i}").replace(" ", "_")[:40]
        note_tag = f"{TAG} missed-{boss}"
        a, f = _add(
            args.data_dir,
            start=float(row["start_sec"]),
            end=float(row["end_sec"]),
            kind=str(row["train_as"]),
            note_tag=note_tag,
            notes=str(row.get("notes") or boss),
        )
        added += a
        frames += f

    for i, row in enumerate(verdicts.get("hard_negatives") or []):
        note_tag = f"{TAG} hardneg#{i}"
        a, f = _add(
            args.data_dir,
            start=float(row["start_sec"]),
            end=float(row["end_sec"]),
            kind="not_fight",
            note_tag=note_tag,
            notes=str(row.get("notes") or "hard negative"),
        )
        added += a
        frames += f

    print({"clips_added": added, "frames_added": frames, "counts": label_counts(args.data_dir)})
    if args.train and added:
        metrics = train_model(args.data_dir, epochs=args.epochs)
        clear_model_cache()
        print("train", metrics)


if __name__ == "__main__":
    main()
