"""Quick probe of fight detection in match window."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twitch_tiktok_bot.models import ClipAnalysis, LoudPeak, TimeRange, TranscriptSegment
from twitch_tiktok_bot.plan.game_profiles import apex_fight_windows
from twitch_tiktok_bot.plan.action import expand_gunfight_window, segment_action_score


def load_analysis(path: Path) -> ClipAnalysis:
    d = json.loads(path.read_text(encoding="utf-8"))
    return ClipAnalysis(
        duration=d["duration"],
        transcript_segments=[TranscriptSegment(**s) for s in d["transcript_segments"]],
        loud_peaks=[LoudPeak(time=p["t"], score=p["score"]) for p in d["loud_peaks"]],
        silence_ranges=[TimeRange(**r) for r in d["silence_ranges"]],
    )


def main() -> None:
    work = Path("data/2788855626")
    analysis = load_analysis(work / "analysis.json")
    mstart, mend = 246.48, 1284.23

    # Filter peaks to match
    peaks = [p for p in analysis.loud_peaks if mstart <= p.time <= mend]
    sub = ClipAnalysis(
        duration=mend - mstart,
        transcript_segments=[
            TranscriptSegment(s.start - mstart, s.end - mstart, s.text)
            for s in analysis.transcript_segments
            if s.end >= mstart and s.start <= mend
        ],
        loud_peaks=[LoudPeak(p.time - mstart, p.score) for p in peaks],
        silence_ranges=[
            TimeRange(max(0, r.start - mstart), r.end - mstart)
            for r in analysis.silence_ranges
            if r.end >= mstart and r.start <= mend
        ],
    )

    print("apex_fight_windows (default):")
    for s, e, sc in apex_fight_windows(sub)[:15]:
        print(f"  {mstart+s:7.1f}-{mstart+e:7.1f}s  score={sc:.1f}")

    print("\nexpanded uncapped clusters:")
    raw = apex_fight_windows(sub, window_sec=18.0, min_peaks=4)
    for s, e, sc in raw:
        es, ee = expand_gunfight_window(
            sub, s, e, max_duration=0, max_prelude=35.0, min_duration=8.0
        )
        print(f"  {mstart+es:7.1f}-{mstart+ee:7.1f}s  ({ee-es:.0f}s) score={sc:.1f}")

    print("\n732-800 window score:", segment_action_score(analysis, 732, 800, require_combat_for_apex=True))

    print("\npeak blocks 11-14 min (absolute):")
    for block in range(660, 851, 30):
        n = sum(1 for p in analysis.loud_peaks if block <= p.time < block + 30)
        if n:
            print(f"  {block//60}:{block%60:02d}  {n} peaks")

    print("\ntranscript 720-820:")
    for seg in analysis.transcript_segments:
        if 720 <= seg.start < 820:
            print(f"  {seg.start:7.1f}  {seg.text[:90]}")


    from twitch_tiktok_bot.labels.detect_fights import _expand_fight_arc, _window_action_score

    es, ee = _expand_fight_arc(analysis, 726, 820, match_start=246, match_end=1284)
    print(f"\nexpand 726-820: {es:.1f}-{ee:.1f} ({ee-es:.0f}s) score={_window_action_score(analysis, es, ee)}")

    print("\npeak blocks 18-21 min:")
    for block in range(1080, 1290, 30):
        n = sum(1 for p in analysis.loud_peaks if block <= p.time < block + 30)
        if n:
            print(f"  {block//60}:{block%60:02d}  {n} peaks")
    print("transcript 1080-1290:")
    for seg in analysis.transcript_segments:
        if 1080 <= seg.start < 1290:
            print(f"  {seg.start:7.1f}  {seg.text[:90]}")
    print("score 1113-1284:", segment_action_score(analysis, 1113, 1284, require_combat_for_apex=True))


if __name__ == "__main__":
    main()
