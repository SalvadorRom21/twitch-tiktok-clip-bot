"""Inventory + deep-review auto-detected Elden clips without human uploads.

For each boss_scan.json candidate:
  - extract key frames (preroll, onset, mid, end/terminal)
  - score with OpenCV analyze_frame
  - write a review JSON the agent (or human) can use to label/fix

Usage:
  python scripts/review_scan_candidates.py
  python scripts/review_scan_candidates.py --vod 76cd3cf1089f
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twitch_tiktok_bot.labels.elden_boss_detect import analyze_frame  # noqa: E402


def _find_video(vod_dir: Path, vod_id: str) -> Path | None:
    for pat in (f"{vod_id}.mp4", f"{vod_id}.mkv", "*.mp4", "*.mkv", "*.webm"):
        found = sorted(vod_dir.glob(pat))
        if found:
            return found[0]
    return None


def _grab(cap: cv2.VideoCapture, fps: float, t: float) -> object | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(max(0.0, t) * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def review_candidate(
    vod_id: str,
    video: Path,
    cand: dict,
    index: int,
    out_dir: Path,
) -> dict:
    start = float(cand.get("start_sec") or 0.0)
    end = float(cand.get("end_sec") or start)
    sig = cand.get("signals") or {}
    onset = float(sig.get("hud_onset_sec") or start)
    mid = (start + end) / 2.0
    times = {
        "start": start,
        "onset": onset,
        "mid": mid,
        "end_minus_3": max(start, end - 3.0),
        "end": end,
    }
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_meta: dict[str, dict] = {}
    cand_dir = out_dir / f"cand_{index:02d}_{int(start)}_{int(end)}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for name, t in times.items():
        frame = _grab(cap, fps, t)
        if frame is None:
            frames_meta[name] = {"time_sec": t, "ok": False}
            continue
        path = cand_dir / f"{name}_{t:.1f}.jpg"
        cv2.imwrite(str(path), frame)
        s = analyze_frame(frame, float(t))
        frames_meta[name] = {
            "time_sec": t,
            "ok": True,
            "path": str(path).replace("\\", "/"),
            "boss_bar_score": s.boss_bar_score,
            "boss_bar_fill": s.boss_bar_fill,
            "boss_name_score": s.boss_name_score,
            "dual_boss_bar_score": s.dual_boss_bar_score,
            "death_screen_score": s.death_screen_score,
            "victory_screen_score": s.victory_screen_score,
        }
    cap.release()
    return {
        "vod_id": vod_id,
        "index": index,
        "start_sec": start,
        "end_sec": end,
        "confidence": cand.get("confidence"),
        "source": cand.get("source"),
        "reason": cand.get("reason"),
        "had_death_screen": cand.get("had_death_screen"),
        "had_victory_screen": cand.get("had_victory_screen"),
        "signals": sig,
        "frames": frames_meta,
        # Filled by agent review pass:
        "verdict": None,  # correct | wrong_start | wrong_end | false_positive | missed_split
        "notes": "",
        "suggested_start_sec": None,
        "suggested_end_sec": None,
        "suggested_kind": None,
    }


def inventory(data_dir: Path, vod_filter: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for vod in sorted(data_dir.iterdir()):
        if not vod.is_dir():
            continue
        if vod_filter and vod.name != vod_filter:
            continue
        scan = vod / "boss_scan.json"
        if not scan.exists():
            continue
        try:
            data = json.loads(scan.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cands = data.get("candidates") or []
        rows.append(
            {
                "vod_id": vod.name,
                "n_candidates": len(cands),
                "duration_sec": data.get("duration_sec"),
                "scanned_at": data.get("scanned_at"),
                "candidates": cands,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--vod", type=str, default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "reference" / "clip_reviews",
    )
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    vods = inventory(args.data_dir, args.vod or None)
    print(f"VODs with scans: {len(vods)}")
    for v in vods:
        print(
            f"  {v['vod_id']}: {v['n_candidates']} candidates "
            f"dur={v['duration_sec']} scanned={str(v['scanned_at'] or '')[:19]}"
        )
        for i, c in enumerate(v["candidates"][:30]):
            sig = c.get("signals") or {}
            print(
                f"    #{i} {c.get('start_sec')}->{c.get('end_sec')} "
                f"kind={sig.get('ml_end_kind')} onset={sig.get('hud_onset_sec')} "
                f"death={c.get('had_death_screen')} vic={c.get('had_victory_screen')} "
                f"conf={c.get('confidence')}"
            )
    if args.inventory_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    all_reviews: list[dict] = []
    for v in vods:
        vod_id = v["vod_id"]
        vod_dir = args.data_dir / vod_id
        video = _find_video(vod_dir, vod_id)
        if video is None:
            print(f"SKIP {vod_id}: no video")
            continue
        out_dir = args.out / vod_id
        out_dir.mkdir(parents=True, exist_ok=True)
        reviews = []
        for i, cand in enumerate(v["candidates"]):
            print(f"Review extract {vod_id} #{i}…")
            row = review_candidate(vod_id, video, cand, i, out_dir)
            reviews.append(row)
            all_reviews.append(row)
        (out_dir / "reviews.json").write_text(
            json.dumps(reviews, indent=2), encoding="utf-8"
        )
    summary_path = args.out / "all_reviews.json"
    summary_path.write_text(json.dumps(all_reviews, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_reviews)} reviews -> {summary_path}")


if __name__ == "__main__":
    main()
