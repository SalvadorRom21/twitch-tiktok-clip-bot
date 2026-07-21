"""Apply deep-review 0720 verdicts: tighten fights, hard-neg FPs, retrain."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twitch_tiktok_bot.labels.elden_boss_detect import (  # noqa: E402
    BossFightCandidate,
    BossScanResult,
    save_scan_result,
)
from twitch_tiktok_bot.labels.elden_ml.clips import read_clips, save_manual_clip  # noqa: E402
from twitch_tiktok_bot.labels.elden_ml.dataset import label_counts  # noqa: E402
from twitch_tiktok_bot.labels.elden_ml.infer import clear_model_cache  # noqa: E402
from twitch_tiktok_bot.labels.elden_ml.train import train_model  # noqa: E402
from twitch_tiktok_bot.labels.supervised import (  # noqa: E402
    CandidateReview,
    SupervisedTrainingStore,
    export_training_dataset,
    save_supervised_store,
)

VOD = "1fbc9fdef6a0"
TAG = "0720-deep2"


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


def _update_scan(data_dir: Path, rows: list[dict]) -> None:
    work = data_dir / VOD
    cands: list[BossFightCandidate] = []
    for i, row in enumerate(rows):
        if row.get("verdict") == "false_positive":
            continue
        kind = str(row["train_as"])
        boss = {
            1: "Night's Cavalry",
            2: "Guardian Golem",
            3: "Cemetery Shade",
            4: "Deathbird",
            5: "Runebear",
            6: "Malenia, Blade of Miquella",
            7: "Malenia, Goddess of Rot",
        }.get(int(row["index"]), "")
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        cands.append(
            BossFightCandidate(
                id=f"deep2_{i:02d}",
                start_sec=start,
                end_sec=end,
                confidence=0.9 if kind != "incomplete" else 0.8,
                reason=f"Deep review · {boss} · {row.get('notes', '')[:80]}",
                signals={
                    "peak_boss_bar_score": 1.0,
                    "peak_boss_name_score": 1.0,
                    "peak_boss_bar_fill": 0.9,
                    "had_death_screen": kind == "you_died",
                    "had_victory_screen": kind == "enemy_felled",
                    "ml_end_kind": kind,
                    "detector": "agent_0720_deep2",
                    "hud_onset_sec": start + 2.0,
                    "boss_name": boss,
                    "agent_notes": str(row.get("notes") or ""),
                },
                peak_boss_bar_score=1.0,
                had_death_screen=kind == "you_died",
                had_victory_screen=kind == "enemy_felled",
                source="manual",
            )
        )
    result = BossScanResult(
        vod_id=VOD,
        duration_sec=8828.36,
        candidates=cands,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )
    save_scan_result(work, result)
    print(f"scan candidates -> {len(cands)}")

    store = SupervisedTrainingStore(
        vod_id=VOD, title="0720", source="tiktok_upload", scan_status="done"
    )
    now = datetime.now(timezone.utc).isoformat()
    for c in cands:
        boss = (c.signals or {}).get("boss_name") or ""
        store.reviews.append(
            CandidateReview(
                candidate_id=c.id,
                verdict="correct",
                boss_type=str(boss),
                notes=str((c.signals or {}).get("agent_notes") or "deep2"),
                reviewed_at=now,
                start_sec=float(c.start_sec),
                end_sec=float(c.end_sec),
            )
        )
    save_supervised_store(work, store)
    export_training_dataset(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=ROOT
        / "data"
        / "reference"
        / "clip_reviews"
        / "agent_verdicts_0720_deep.json",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    added = 0
    frames = 0

    for row in verdicts.get("auto_candidates") or []:
        note_tag = f"{TAG} cand#{row['index']}"
        a, f = _add(
            args.data_dir,
            start=float(row["start_sec"]),
            end=float(row["end_sec"]),
            kind=str(row["train_as"]),
            note_tag=note_tag,
            notes=str(row.get("notes") or ""),
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

    _update_scan(args.data_dir, list(verdicts.get("auto_candidates") or []))
    print({"clips_added": added, "frames_added": frames, "counts": label_counts(args.data_dir)})
    if args.train and added:
        metrics = train_model(args.data_dir, epochs=args.epochs)
        clear_model_cache()
        print("train", metrics)


if __name__ == "__main__":
    main()
