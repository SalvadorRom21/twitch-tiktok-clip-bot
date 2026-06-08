#!/usr/bin/env python3
"""CLI for Twitch-to-TikTok clip automation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from twitch_tiktok_bot.config import apply_game_profile, load_config
from twitch_tiktok_bot.plan.game_profiles import GAME_PROFILE_OPTIONS
from twitch_tiktok_bot.ingest.twitch import fetch_recent_clips, fetch_recent_vods
from twitch_tiktok_bot.pipeline import (
    process_cached_clip,
    process_clip_url,
    process_media_url,
    process_twitch_clip,
    process_twitch_vod,
    process_vod_url,
    load_cached_analysis,
    render_fight_clips,
    render_match_montage,
    render_smash_game_clips,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Twitch clips/VODs, analyze them, and render TikTok shorts.",
    )
    parser.add_argument(
        "--clip-url",
        help="Process a single Twitch clip URL",
    )
    parser.add_argument(
        "--vod-url",
        help="Process a full Twitch VOD/stream recording URL",
    )
    parser.add_argument(
        "--media-url",
        help="Auto-detect clip or VOD URL and process",
    )
    parser.add_argument(
        "--clip-id",
        help="Optional ID for output naming (also the data/<id>/ cache folder)",
    )
    parser.add_argument(
        "--from-cache",
        metavar="ID",
        help="Reprocess a previously downloaded clip from data/<ID>/ (no download)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="With --clip-url/--vod-url: use cached video in data/<clip-id>/ if present",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force a fresh yt-dlp download even if a cached video exists",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Download/analyze VOD only; skip rendering clips",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title metadata for hooks/hashtags",
    )
    parser.add_argument(
        "--max-shorts",
        type=int,
        default=None,
        help="Max TikTok shorts to generate from a VOD (default from config)",
    )
    parser.add_argument(
        "--fetch-clips",
        action="store_true",
        help="Fetch recent clips from Twitch API and process each one",
    )
    parser.add_argument(
        "--fetch-vods",
        action="store_true",
        help="Fetch recent stream VODs from Twitch API and process each one",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days back to fetch clips/VODs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config.local.yaml or config.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root for resolving relative paths",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the preview web UI server",
    )
    parser.add_argument(
        "--label",
        metavar="VOD_ID",
        nargs="?",
        const="",
        help="Open fight labeler UI (optional VOD id, e.g. 2788855626)",
    )
    parser.add_argument(
        "--literacy",
        metavar="VOD_ID",
        nargs="?",
        const="",
        help="Open Apex literacy trainer UI (optional VOD id)",
    )
    parser.add_argument(
        "--scan-matches",
        metavar="VOD_ID",
        help="Detect Apex match boundaries (drop ship → elimination/win) in cached VOD",
    )
    parser.add_argument(
        "--scan-fights",
        metavar="VOD_ID",
        help="Auto-detect fights inside the active match (requires --scan-matches first)",
    )
    parser.add_argument(
        "--match-montage",
        metavar="VOD_ID",
        help="Render action-cut montage of all detected fights (requires --scan-fights first)",
    )
    parser.add_argument(
        "--render-fights",
        metavar="VOD_ID",
        help="Render one action-cut video per detected fight (requires --scan-fights first)",
    )
    parser.add_argument(
        "--scan-sets",
        metavar="VOD_ID",
        help="Detect Smash Bros sets/games and Bo3/Bo5 format (use --game-profile smash)",
    )
    parser.add_argument(
        "--render-smash-games",
        metavar="VOD_ID",
        help="Render one video per Smash game (requires --scan-sets first)",
    )
    parser.add_argument(
        "--youtube-url",
        help="Download and analyze a YouTube VOD (same as --media-url for YouTube links)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Web server host (with --web, default from config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Web server port (with --web, default from config)",
    )
    profile_ids = [opt["id"] for opt in GAME_PROFILE_OPTIONS]
    parser.add_argument(
        "--game-profile",
        choices=profile_ids,
        default=None,
        metavar="PROFILE",
        help=(
            "Highlight detection profile for this run "
            f"({', '.join(profile_ids)}). Overrides config.local.yaml."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(config_path=args.config, project_root=args.project_root)
    apply_game_profile(config, args.game_profile)

    download_kwargs = {
        "skip_download": args.skip_download,
        "redownload": args.redownload,
        "analyze_only": args.analyze_only,
    }

    if args.scan_sets:
        from twitch_tiktok_bot.labels.smash_sets import scan_vod_smash_sets

        work_dir = config.resolve_path(config.paths.data_dir) / args.scan_sets
        if not (work_dir / "analysis.json").exists():
            print(f"No analysis at {work_dir / 'analysis.json'}. Download/analyze first.")
            return 1
        analysis = load_cached_analysis(work_dir)
        store = scan_vod_smash_sets(work_dir, args.scan_sets, analysis)
        game_count = sum(len(item.games) for item in store.sets)
        print(f"Detected {len(store.sets)} set(s), {game_count} game(s)\n")
        for set_idx, smash_set in enumerate(store.sets, start=1):
            fmt = smash_set.format.upper() if smash_set.format != "unknown" else "?"
            m1, s1 = int(smash_set.start_sec // 60), int(smash_set.start_sec % 60)
            m2, s2 = int(smash_set.end_sec // 60), int(smash_set.end_sec % 60)
            print(
                f"  Set {set_idx}  {m1}:{s1:02d}–{m2}:{s2:02d}  "
                f"[{fmt}] {len(smash_set.games)} game(s)  "
                f"(first to {smash_set.games_to_win})  [{smash_set.confidence}]"
            )
            if smash_set.notes:
                print(f"    format: {smash_set.notes}")
            for game in smash_set.games:
                gm1, gs1 = int(game.start_sec // 60), int(game.start_sec % 60)
                gm2, gs2 = int(game.end_sec // 60), int(game.end_sec % 60)
                print(
                    f"      Game {game.game_number}  {gm1}:{gs1:02d}–{gm2}:{gs2:02d}  "
                    f"({game.duration_sec():.0f}s)  [{game.confidence}]"
                )
                if game.start_cue:
                    print(f"        start: {game.start_cue[:70]}")
        print(f"\nSaved -> {work_dir / 'smash_sets.json'}")
        return 0

    if args.render_smash_games:
        try:
            outputs = render_smash_game_clips(args.render_smash_games, config)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return 1
        print(f"\nRendered {len(outputs)} Smash game video(s):")
        for path in outputs:
            print(f"  {path}")
        return 0

    if args.render_fights:
        try:
            outputs = render_fight_clips(args.render_fights, config)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return 1
        print(f"\nRendered {len(outputs)} fight video(s):")
        for path in outputs:
            print(f"  {path}")
        return 0

    if args.match_montage:
        try:
            out_path = render_match_montage(args.match_montage, config)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return 1
        print(f"\nMatch montage ready → {out_path}")
        return 0

    if args.scan_fights:
        from twitch_tiktok_bot.labels.detect_fights import scan_vod_fights

        def _load_cached_analysis(work_dir: Path):
            from twitch_tiktok_bot.models import (
                ClipAnalysis,
                LoudPeak,
                TimeRange,
                TranscriptSegment,
            )
            import json

            data = json.loads((work_dir / "analysis.json").read_text(encoding="utf-8"))
            return ClipAnalysis(
                duration=data["duration"],
                transcript_segments=[
                    TranscriptSegment(**s) for s in data.get("transcript_segments", [])
                ],
                loud_peaks=[
                    LoudPeak(time=p["t"], score=p["score"])
                    for p in data.get("loud_peaks", [])
                ],
                silence_ranges=[
                    TimeRange(**r) for r in data.get("silence_ranges", [])
                ],
            )

        work_dir = config.resolve_path(config.paths.data_dir) / args.scan_fights
        if not (work_dir / "analysis.json").exists():
            print(f"No analysis at {work_dir / 'analysis.json'}. Run --from-cache first.")
            return 1
        analysis = _load_cached_analysis(work_dir)
        store = scan_vod_fights(work_dir, args.scan_fights, analysis, config)
        print(
            f"Detected {len(store.fights)} fight(s) in match "
            f"{store.match_start_sec/60:.1f}–{store.match_end_sec/60:.1f} min\n"
        )
        for idx, fight in enumerate(store.fights, start=1):
            m1 = int(fight.start_sec // 60)
            s1 = int(fight.start_sec % 60)
            m2 = int(fight.end_sec // 60)
            s2 = int(fight.end_sec % 60)
            print(
                f"  Fight {idx}  {m1}:{s1:02d}–{m2}:{s2:02d} "
                f"({fight.duration_sec():.0f}s)  [{fight.confidence}] score={fight.score}"
            )
            if fight.start_cue:
                print(f"    start: {fight.start_cue[:75]}")
            if fight.end_cue:
                print(f"    end:   {fight.end_cue[:75]}")
        from twitch_tiktok_bot.labels.fights import load_fight_labels

        labels = load_fight_labels(work_dir)
        if labels and labels.fights:
            print("\nYour training labels (reference):")
            for fight in labels.fights:
                m1, s1 = int(fight.start_sec // 60), int(fight.start_sec % 60)
                m2, s2 = int(fight.end_sec // 60), int(fight.end_sec % 60)
                print(f"  {m1}:{s1:02d}–{m2}:{s2:02d}  {fight.description[:55]}")

        print(f"\nSaved → {work_dir / 'detected_fights.json'}")
        return 0

    if args.scan_matches:
        from twitch_tiktok_bot.labels.matches import (
            ApexMatchStore,
            detect_apex_matches,
            fights_in_match,
            load_matches,
            save_matches,
        )
        from twitch_tiktok_bot.models import ClipAnalysis, LoudPeak, TimeRange, TranscriptSegment
        import json

        work_dir = config.resolve_path(config.paths.data_dir) / args.scan_matches
        analysis_path = work_dir / "analysis.json"
        if not analysis_path.exists():
            print(f"No analysis at {analysis_path}. Run --from-cache first.")
            return 1
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis = ClipAnalysis(
            duration=data["duration"],
            transcript_segments=[
                TranscriptSegment(**s) for s in data.get("transcript_segments", [])
            ],
            loud_peaks=[
                LoudPeak(time=p["t"], score=p["score"])
                for p in data.get("loud_peaks", [])
            ],
            silence_ranges=[
                TimeRange(**r) for r in data.get("silence_ranges", [])
            ],
        )
        prior = load_matches(work_dir)
        detected = detect_apex_matches(analysis)
        if prior:
            prior_by_start = {round(m.start_sec, 0): m for m in prior.matches}
            for match in detected:
                old = prior_by_start.get(round(match.start_sec, 0))
                if old:
                    match.id = old.id
                    match.use_for_clips = old.use_for_clips
                    match.notes = old.notes or match.notes
        store = ApexMatchStore(vod_id=args.scan_matches, matches=detected)
        save_matches(work_dir, store)
        print(f"Detected {len(store.matches)} match(es) in VOD {args.scan_matches}\n")
        for idx, match in enumerate(store.matches, start=1):
            print(
                f"  Match {idx}  {match.start_sec/60:.1f}–{match.end_sec/60:.1f} min "
                f"({match.duration_sec()/60:.1f} min)  [{match.confidence}] {match.end_type}"
            )
            print(f"    start: {match.start_cue[:80]}")
            print(f"    end:   {match.end_cue[:80]}")
            fights = fights_in_match(work_dir, match)
            if fights:
                print(f"    fights in window: {len(fights)} labeled")
                for fight in fights:
                    print(
                        f"      - {fight['start_sec']:.0f}s–{fight['end_sec']:.0f}s "
                        f"{fight['description'][:50]}"
                    )
        out = work_dir / "apex_matches.json"
        print(f"\nSaved → {out}")
        return 0

    if args.from_cache:
        process_cached_clip(
            cache_id=args.from_cache,
            config=config,
            clip_title=args.title,
        )
        return 0

    if args.web or args.label is not None or args.literacy is not None:
        if args.host:
            config.web.host = args.host
        if args.port:
            config.web.port = args.port
        from twitch_tiktok_bot.web.app import run_server

        base = f"http://{config.web.host}:{config.web.port}"
        if args.literacy is not None:
            suffix = (
                f"/literacy?vod={args.literacy}" if args.literacy else "/literacy"
            )
            print(f"Starting Apex literacy trainer at {base}{suffix}")
        elif args.label is not None:
            suffix = f"/label?vod={args.label}" if args.label else "/label"
            print(f"Starting fight labeler at {base}{suffix}")
        else:
            print(f"Starting preview UI at {base}")
        run_server(config)
        return 0

    if args.youtube_url:
        outputs = process_media_url(
            url=args.youtube_url,
            config=config,
            media_id=args.clip_id,
            title=args.title,
            max_shorts=args.max_shorts,
            **download_kwargs,
        )
        print(f"\nDownloaded and analyzed. Cache id: {args.clip_id or '(auto from URL)'}")
        print(f"Next: python main.py --scan-sets <ID> --game-profile smash")
        if outputs:
            print(f"Outputs: {len(outputs)} file(s)")
        return 0

    if args.vod_url:
        outputs = process_vod_url(
            url=args.vod_url,
            config=config,
            vod_id=args.clip_id,
            vod_title=args.title,
            max_shorts=args.max_shorts,
            **download_kwargs,
        )
        print(f"\nGenerated {len(outputs)} short(s).")
        return 0

    if args.media_url:
        outputs = process_media_url(
            url=args.media_url,
            config=config,
            media_id=args.clip_id,
            title=args.title,
            max_shorts=args.max_shorts,
            **download_kwargs,
        )
        print(f"\nGenerated {len(outputs)} short(s).")
        return 0

    if args.clip_url:
        process_clip_url(
            url=args.clip_url,
            config=config,
            clip_id=args.clip_id,
            clip_title=args.title,
            **download_kwargs,
        )
        return 0

    if args.fetch_clips:
        clips = fetch_recent_clips(config, days_back=args.days)
        if not clips:
            print("No clips found.")
            return 0
        print(f"Found {len(clips)} clip(s). Processing...")
        for clip in clips:
            print(f"\n=== {clip.title} ({clip.id}) ===")
            process_twitch_clip(clip, config)
        return 0

    if args.fetch_vods:
        vods = fetch_recent_vods(config, days_back=args.days)
        if not vods:
            print("No VODs found.")
            return 0
        print(f"Found {len(vods)} VOD(s). Processing...")
        for vod in vods:
            print(f"\n=== {vod.title} ({vod.id}) — {vod.duration_sec/60:.0f} min ===")
            process_twitch_vod(vod, config, max_shorts=args.max_shorts)
        return 0

    parser.print_help()
    print("\nExamples:")
    print("  python main.py --clip-url https://clips.twitch.tv/SomeClipSlug")
    print("  python main.py --vod-url https://www.twitch.tv/videos/1234567890")
    print("  python main.py --media-url https://www.twitch.tv/videos/1234567890")
    print("  python main.py --fetch-vods --max-shorts 3")
    print("  python main.py --fetch-clips --game-profile apex")
    print("  python main.py --from-cache clip")
    print("  python main.py --clip-url <url>   # reuses data/<clip-id>/ if cached")
    print("  python main.py --web --game-profile apex")
    print("  python main.py --label 2788855626")
    print("  python main.py --literacy 2788855626")
    print("  python main.py --youtube-url https://www.youtube.com/watch?v=... --clip-id smashvod --game-profile smash")
    print("  python main.py --scan-sets smashvod --game-profile smash")
    print("  python main.py --render-smash-games smashvod")
    return 1


if __name__ == "__main__":
    sys.exit(main())
