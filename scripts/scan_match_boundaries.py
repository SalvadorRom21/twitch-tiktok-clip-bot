"""Scan VOD analysis for Apex match start/end boundaries."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

START_KW = re.compile(
    r"drop|jump|launch|deploy|skydive|ship|legend|ready|"
    r"drop\s*ship|dropship|jumping|here we go|let'?s go",
    re.I,
)
END_KW = re.compile(
    r"champion|eliminated|elimination|you are the|squad.?elim|"
    r"game over|lobby|requeue|queue up|placement|"
    r"second place|third place|\bwin\b|\bwon\b|champ|"
    r"gg|good game|summary|return to",
    re.I,
)
MENU_KW = re.compile(
    r"lobby|menu|queue|loading|legend select|ready up|character select",
    re.I,
)


def main() -> int:
    vod_id = sys.argv[1] if len(sys.argv) > 1 else "2788855626"
    path = Path("data") / vod_id / "analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    duration = data["duration"]
    transcript = data.get("transcript_segments", [])
    peaks = [(p["t"], p["score"]) for p in data.get("loud_peaks", [])]
    silence = data.get("silence_ranges", [])

    print(f"VOD {vod_id}: {duration/60:.1f} min analyzed")
    print(f"  transcript={len(transcript)} peaks={len(peaks)} silence={len(silence)}")

    print("\n--- Possible MATCH START cues ---")
    for seg in transcript:
        if START_KW.search(seg["text"]):
            print(f"  {seg['start']:7.1f}s  {seg['text'][:100]}")

    print("\n--- Possible MATCH END cues ---")
    for seg in transcript:
        if END_KW.search(seg["text"]):
            print(f"  {seg['start']:7.1f}s  {seg['text'][:100]}")

    print("\n--- Long silence gaps (>=40s) ---")
    for r in silence:
        span = r["end"] - r["start"]
        if span >= 40:
            print(f"  {r['start']:7.1f}-{r['end']:7.1f}s  ({span:.0f}s)")

    print("\n--- Peak density per 5 min ---")
    for block in range(int(duration // 300) + 1):
        a, b = block * 300, min(duration, (block + 1) * 300)
        n = sum(1 for t, _ in peaks if a <= t < b)
        if n:
            print(f"  {a/60:5.0f}-{b/60:5.0f} min: {n} peaks")

    # Heuristic: match = gameplay block between silences/menu
    print("\n--- Heuristic match candidates ---")
    fights_path = Path("data") / vod_id / "fight_labels.json"
    if fights_path.exists():
        fights = json.loads(fights_path.read_text(encoding="utf-8")).get("fights", [])
        if fights:
            f0 = min(f["start_sec"] for f in fights)
            f1 = max(f["end_sec"] for f in fights)
            print(f"  Labeled fights span: {f0:.0f}s - {f1:.0f}s ({(f1-f0)/60:.1f} min)")

    windows = [
        (0, 360, "first 6 min"),
        (220, 340, "drop + fight1 start"),
        (850, 1020, "fight3 + after"),
        (980, 1400, "post fight3 through 23min"),
        (1400, 1800, "20-30 min"),
        (1180, 1280, "match1 end debrief"),
        (1380, 1460, "match2 start"),
    ]
    print("\n--- Transcript windows ---")
    for lo, hi, label in windows:
        print(f"\n=== {label} ({lo}-{hi}s) ===")
        for seg in transcript:
            if lo <= seg["start"] < hi:
                print(f"  {seg['start']:7.1f}s  {seg['text'][:110]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
