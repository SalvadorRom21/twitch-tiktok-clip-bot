"""End-to-end clip and VOD processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from twitch_tiktok_bot.analyze.duration import get_video_duration
from twitch_tiktok_bot.analyze.pipeline import analyze_clip
from twitch_tiktok_bot.analyze.vod import analyze_vod
from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.ingest.download import find_cached_video, resolve_video_path
from twitch_tiktok_bot.ingest.media import is_longform_url, parse_media_id
from twitch_tiktok_bot.ingest.vod import is_vod_url, parse_vod_id
from twitch_tiktok_bot.models import (
    ClipAnalysis,
    LoudPeak,
    TimeRange,
    TranscriptSegment,
    TwitchClip,
    TwitchVod,
)
from twitch_tiktok_bot.plan.editor_llm import create_edit_plan
from twitch_tiktok_bot.plan.vod_planner import create_vod_short_plans
from twitch_tiktok_bot.render.ffmpeg_build import render_short
from twitch_tiktok_bot.progress import step
from twitch_tiktok_bot.status import ClipJob, ClipStatus, save_job


def _print_montage_plan(plan, source_duration: float) -> None:
    print(
        f"       POV {source_duration:.1f}s -> montage {plan.target_duration_sec:.1f}s, "
        f"segments={len(plan.segments)}, hook=\"{plan.hook_text}\""
    )
    for idx, seg in enumerate(plan.segments, start=1):
        print(
            f"         [{idx}] {seg.start:.1f}–{seg.end:.1f}s "
            f"({seg.end - seg.start:.1f}s) {seg.reason}"
        )


def _save_job_summary(
    work_dir: Path,
    safe_id: str,
    source_url: str,
    outputs: list[Path],
    analysis_path: Path,
) -> None:
    summary = {
        "clip_id": safe_id,
        "source_url": source_url,
        "outputs": [str(p) for p in outputs],
        "output_video": str(outputs[0]) if outputs else "",
        "caption_file": str(outputs[0].with_suffix(".txt")) if outputs else "",
        "analysis": str(analysis_path),
    }
    (work_dir / "job_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def process_vod_url(
    url: str,
    config: AppConfig,
    vod_id: str | None = None,
    vod_title: str = "",
    game_name: str = "",
    max_shorts: int | None = None,
    *,
    skip_download: bool = False,
    redownload: bool = False,
    analyze_only: bool = False,
) -> list[Path]:
    data_dir = config.resolve_path(config.paths.data_dir)
    output_dir = config.resolve_path(config.paths.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = vod_id or parse_media_id(url) or parse_vod_id(url) or "vod"
    work_dir = data_dir / safe_id
    work_dir.mkdir(parents=True, exist_ok=True)

    end_sec = config.vod.max_download_sec if config.vod.max_download_sec > 0 else None

    print(f"[1/4] Resolving VOD video...")
    video_path, used_cache = resolve_video_path(
        url,
        work_dir,
        video_id=safe_id,
        skip_download=skip_download,
        redownload=redownload,
        end_sec=end_sec,
    )
    print(f"       -> {video_path}" + (" (cached)" if used_cache else ""))

    print(f"[2/4] Analyzing full VOD (chunked scan)...")
    analysis = analyze_vod(
        video_path,
        config,
        work_dir,
        vod_title=vod_title,
        game_name=game_name,
    )
    print(
        f"       duration={analysis.duration/60:.1f} min, "
        f"transcript={len(analysis.transcript_segments)} segments, "
        f"peaks={len(analysis.loud_peaks)}"
    )

    if analyze_only:
        _save_job_summary(work_dir, safe_id, url, [], work_dir / "analysis.json")
        print("       analyze-only: skipping render")
        return []

    if config.editing.pov_montage:
        print(f"[3/4] Building POV montage from full stream...")
        from twitch_tiktok_bot.plan.action import summarize_clip_action

        action = summarize_clip_action(analysis, config.editing.game_profile)
        if action.warning:
            print(f"       ! {action.warning}")
        plan = create_edit_plan(analysis, config, work_dir=work_dir)
        _print_montage_plan(plan, analysis.duration)

        print(f"[4/4] Rendering POV montage...")
        out_path = output_dir / f"{safe_id}_tiktok.mp4"
        render_short(video_path, analysis, plan, out_path, config, work_dir)
        outputs: list[Path] = [out_path]
        print(f"       -> {out_path}")
        (work_dir / "edit_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2), encoding="utf-8"
        )
    else:
        print(f"[3/4] Finding highlights across stream...")
        plans = create_vod_short_plans(analysis, config, max_shorts=max_shorts)
        print(f"       {len(plans)} TikTok short(s) planned")

        print(f"[4/4] Rendering TikTok shorts...")
        outputs = []
        for idx, plan in enumerate(plans, start=1):
            out_path = output_dir / f"{safe_id}_short_{idx:02d}.mp4"
            short_work = work_dir / f"short_{idx:02d}"
            render_short(video_path, analysis, plan, out_path, config, short_work)
            outputs.append(out_path)
            print(f"       -> {out_path}")

    _save_job_summary(work_dir, safe_id, url, outputs, work_dir / "analysis.json")

    if config.editing.pov_montage:
        plan_data = plan.to_dict()
        job_segments = len(plan_data.get("segments", []))
        hook_text = plan_data.get("hook_text", "")
        hashtags = list(plan_data.get("hashtags", []))
    else:
        first_plan = plans[0].to_dict() if plans else {}
        job_segments = len(plans)
        hook_text = first_plan.get("hook_text", "")
        hashtags = list(first_plan.get("hashtags", []))

    save_job(
        config,
        ClipJob(
            id=safe_id,
            clip_url=url,
            title=vod_title,
            status=ClipStatus.READY,
            output_video=str(outputs[0]) if outputs else "",
            caption_file=str(outputs[0].with_suffix(".txt")) if outputs else "",
            hook_text=hook_text,
            hashtags=hashtags,
            segment_count=job_segments,
            face_crop_center_x=analysis.face_crop_center_x,
            source_type="vod",
            output_videos=[str(p) for p in outputs],
        ),
    )
    return outputs


def process_clip_url(
    url: str,
    config: AppConfig,
    clip_id: str | None = None,
    clip_title: str = "",
    game_name: str = "",
    *,
    skip_download: bool = False,
    redownload: bool = False,
) -> Path:
    data_dir = config.resolve_path(config.paths.data_dir)
    output_dir = config.resolve_path(config.paths.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = clip_id or "clip"
    work_dir = data_dir / safe_id
    work_dir.mkdir(parents=True, exist_ok=True)

    with step("[1/4] Resolving clip video", heartbeat_sec=0):
        video_path, used_cache = resolve_video_path(
            url,
            work_dir,
            video_id=safe_id,
            skip_download=skip_download,
            redownload=redownload,
        )
        print(f"       -> {video_path}" + (" (cached)" if used_cache else ""))

    with step("[2/4] Analyzing video", heartbeat_sec=0):
        analysis = analyze_clip(
            video_path,
            config,
            work_dir,
            clip_title=clip_title,
            game_name=game_name,
        )
        print(
            f"       duration={analysis.duration:.1f}s, "
            f"transcript={len(analysis.transcript_segments)} segments, "
            f"peaks={len(analysis.loud_peaks)}"
        )

    with step("[3/4] Creating edit plan", heartbeat_sec=0):
        from twitch_tiktok_bot.plan.action import summarize_clip_action

        action = summarize_clip_action(analysis, config.editing.game_profile)
        if action.warning:
            print(f"       ! {action.warning}")
        plan = create_edit_plan(analysis, config, work_dir=work_dir)
        _print_montage_plan(plan, analysis.duration)

    with step("[4/4] Rendering TikTok short", heartbeat_sec=0):
        out_path = output_dir / f"{safe_id}_tiktok.mp4"
        render_short(video_path, analysis, plan, out_path, config, work_dir)
        print(f"       -> {out_path}")
        print(f"       caption -> {out_path.with_suffix('.txt')}")

    plan_data = plan.to_dict()
    _save_job_summary(work_dir, safe_id, url, [out_path], work_dir / "analysis.json")
    (work_dir / "edit_plan.json").write_text(
        json.dumps(plan_data, indent=2), encoding="utf-8"
    )

    save_job(
        config,
        ClipJob(
            id=safe_id,
            clip_url=url,
            title=clip_title,
            status=ClipStatus.READY,
            output_video=str(out_path),
            caption_file=str(out_path.with_suffix(".txt")),
            hook_text=plan_data.get("hook_text", ""),
            hashtags=list(plan_data.get("hashtags", [])),
            segment_count=len(plan_data.get("segments", [])),
            face_crop_center_x=analysis.face_crop_center_x,
            source_type="clip",
            output_videos=[str(out_path)],
        ),
    )
    return out_path


def process_media_url(
    url: str,
    config: AppConfig,
    media_id: str | None = None,
    title: str = "",
    game_name: str = "",
    max_shorts: int | None = None,
    *,
    skip_download: bool = False,
    redownload: bool = False,
    analyze_only: bool = False,
) -> list[Path]:
    """Route to clip or VOD pipeline based on URL."""
    if is_longform_url(url):
        return process_vod_url(
            url=url,
            config=config,
            vod_id=media_id or parse_media_id(url),
            vod_title=title,
            game_name=game_name,
            max_shorts=max_shorts,
            skip_download=skip_download,
            redownload=redownload,
            analyze_only=analyze_only,
        )
    return [
        process_clip_url(
            url,
            config,
            clip_id=media_id,
            clip_title=title,
            game_name=game_name,
            skip_download=skip_download,
            redownload=redownload,
        )
    ]


def load_cached_analysis(work_dir: Path) -> ClipAnalysis:
    data = json.loads((work_dir / "analysis.json").read_text(encoding="utf-8"))
    return ClipAnalysis(
        duration=data["duration"],
        transcript_segments=[
            TranscriptSegment(**segment)
            for segment in data.get("transcript_segments", [])
        ],
        loud_peaks=[
            LoudPeak(time=peak["t"], score=peak["score"])
            for peak in data.get("loud_peaks", [])
        ],
        silence_ranges=[
            TimeRange(**gap) for gap in data.get("silence_ranges", [])
        ],
        face_crop_center_x=data.get("face_crop_center_x"),
        face_cam_region=data.get("face_cam_region"),
    )


def render_match_montage(vod_id: str, config: AppConfig) -> Path:
    """Render a montage of all detected fights with action-cut pacing."""
    from twitch_tiktok_bot.labels.detect_fights import (
        load_detected_fights,
        plan_from_detected_fights,
    )
    from twitch_tiktok_bot.plan.action_cuts import apply_action_cuts_to_plan

    data_dir = config.resolve_path(config.paths.data_dir)
    output_dir = config.resolve_path(config.paths.output_dir)
    work_dir = data_dir / vod_id
    render_work = work_dir / "match_montage"
    analysis_path = work_dir / "analysis.json"

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"No analysis at {analysis_path}. Run --from-cache {vod_id} first."
        )

    store = load_detected_fights(work_dir)
    if not store or not store.fights:
        raise ValueError(
            f"No fights in {work_dir / 'detected_fights.json'}. "
            f"Run --scan-matches then --scan-fights {vod_id} first."
        )

    video_path = find_cached_video(work_dir, vod_id)
    if video_path is None:
        videos = sorted(work_dir.glob("*.mp4"))
        video_path = videos[0] if videos else None
    if video_path is None:
        raise FileNotFoundError(f"No cached video in {work_dir}")

    print(f"[1/3] Loading cached analysis for {vod_id}...")
    analysis = load_cached_analysis(work_dir)
    usable = [fight for fight in store.fights if fight.use_for_clips]
    print(
        f"       {len(usable)} fight(s), match "
        f"{store.match_start_sec/60:.1f}–{store.match_end_sec/60:.1f} min"
    )

    print(f"[2/3] Building action-cut montage plan...")
    plan = plan_from_detected_fights(store, analysis, config)
    if plan is None:
        raise ValueError("No usable fights to montage.")

    raw_span = plan.target_duration_sec
    raw_segments = len(plan.segments)
    if config.editing.action_cut_enabled:
        plan = apply_action_cuts_to_plan(
            plan, analysis, config, work_dir=work_dir
        )

    _print_montage_plan(plan, analysis.duration)
    print(
        f"       {raw_segments} fight(s) {raw_span:.0f}s raw -> "
        f"{len(plan.segments)} beats {plan.target_duration_sec:.0f}s cut"
    )

    print(f"[3/3] Rendering match montage...")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{vod_id}_match_montage.mp4"
    render_short(video_path, analysis, plan, out_path, config, render_work)
    print(f"       -> {out_path}")

    (work_dir / "match_montage_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2), encoding="utf-8"
    )
    return out_path


def _format_fight_clock(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def render_fight_clips(vod_id: str, config: AppConfig) -> list[Path]:
    """Render one action-cut video per detected fight (exact fight start → end)."""
    from twitch_tiktok_bot.labels.detect_fights import (
        load_detected_fights,
        plan_from_detected_fight,
    )
    from twitch_tiktok_bot.plan.action_cuts import apply_action_cuts_to_plan

    data_dir = config.resolve_path(config.paths.data_dir)
    output_dir = config.resolve_path(config.paths.output_dir)
    work_dir = data_dir / vod_id
    analysis_path = work_dir / "analysis.json"

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"No analysis at {analysis_path}. Run --from-cache {vod_id} first."
        )

    store = load_detected_fights(work_dir)
    if not store or not store.fights:
        raise ValueError(
            f"No fights in {work_dir / 'detected_fights.json'}. "
            f"Run --scan-matches then --scan-fights {vod_id} first."
        )

    video_path = find_cached_video(work_dir, vod_id)
    if video_path is None:
        videos = sorted(work_dir.glob("*.mp4"))
        video_path = videos[0] if videos else None
    if video_path is None:
        raise FileNotFoundError(f"No cached video in {work_dir}")

    fights = sorted(
        [fight for fight in store.fights if fight.use_for_clips],
        key=lambda fight: fight.start_sec,
    )
    if not fights:
        raise ValueError("No fights marked use_for_clips in detected_fights.json")

    print(f"[1/2] Loading cached analysis for {vod_id}...")
    analysis = load_cached_analysis(work_dir)
    print(f"       Rendering {len(fights)} fight video(s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    print(f"[2/2] Rendering per-fight action-cut clips...")
    for idx, fight in enumerate(fights, start=1):
        start_label = _format_fight_clock(fight.start_sec)
        end_label = _format_fight_clock(fight.end_sec)
        raw_span = fight.end_sec - fight.start_sec

        print(
            f"\n  Fight {idx}  {start_label}–{end_label} ({raw_span:.0f}s)  "
            f"[{fight.confidence}]"
        )
        if fight.start_cue:
            print(f"    start: {fight.start_cue[:75]}")

        plan = plan_from_detected_fight(fight, analysis, config)
        if plan is None:
            print("    skipped (invalid window)")
            continue

        if config.editing.action_cut_enabled:
            plan = apply_action_cuts_to_plan(
                plan, analysis, config, work_dir=work_dir
            )

        if not plan.segments:
            print("    skipped (no segments after action cut)")
            continue

        print(
            f"    plan: {len(plan.segments)} beat(s), "
            f"{plan.target_duration_sec:.0f}s cut from {raw_span:.0f}s"
        )
        for beat_idx, seg in enumerate(plan.segments, start=1):
            print(
                f"      [{beat_idx}] {_format_fight_clock(seg.start)}–"
                f"{_format_fight_clock(seg.end)} ({seg.end - seg.start:.1f}s) "
                f"{seg.reason}"
            )

        render_work = work_dir / f"fight_{idx:02d}"
        out_path = output_dir / f"{vod_id}_fight{idx}_actioncut.mp4"
        render_short(video_path, analysis, plan, out_path, config, render_work)
        outputs.append(out_path)
        print(f"    -> {out_path}")

        (work_dir / f"fight_{idx:02d}_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2), encoding="utf-8"
        )

    manifest = {
        "vod_id": vod_id,
        "fights": [
            {
                "index": idx,
                "id": fight.id,
                "start_sec": fight.start_sec,
                "end_sec": fight.end_sec,
                "output": str(outputs[idx - 1]) if idx - 1 < len(outputs) else "",
            }
            for idx, fight in enumerate(fights, start=1)
        ],
    }
    (work_dir / "fight_clips_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return outputs


def render_smash_game_clips(vod_id: str, config: AppConfig) -> list[Path]:
    """Render one video per detected Smash game (grouped by set in filenames)."""
    from twitch_tiktok_bot.labels.smash_sets import (
        load_smash_sets,
        plan_from_smash_game,
    )

    data_dir = config.resolve_path(config.paths.data_dir)
    output_dir = config.resolve_path(config.paths.output_dir)
    work_dir = data_dir / vod_id
    analysis_path = work_dir / "analysis.json"

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"No analysis at {analysis_path}. Download/analyze the VOD first."
        )

    store = load_smash_sets(work_dir)
    if not store or not store.sets:
        raise ValueError(
            f"No sets in {work_dir / 'smash_sets.json'}. "
            f"Run --scan-sets {vod_id} first."
        )

    video_path = find_cached_video(work_dir, vod_id)
    if video_path is None:
        videos = sorted(work_dir.glob("*.mp4"))
        video_path = videos[0] if videos else None
    if video_path is None:
        raise FileNotFoundError(f"No cached video in {work_dir}")

    print(f"[1/2] Loading cached analysis for {vod_id}...")
    analysis = load_cached_analysis(work_dir)
    game_total = sum(
        1
        for smash_set in store.sets
        for game in smash_set.games
        if game.use_for_clips
    )
    print(f"       {len(store.sets)} set(s), {game_total} game(s) to render")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    print(f"[2/2] Rendering per-game clips...")
    for set_idx, smash_set in enumerate(store.sets, start=1):
        if not smash_set.use_for_clips:
            continue
        fmt = smash_set.format.upper() if smash_set.format != "unknown" else "?"
        print(
            f"\n  Set {set_idx}  {_format_fight_clock(smash_set.start_sec)}–"
            f"{_format_fight_clock(smash_set.end_sec)}  "
            f"[{fmt}, {len(smash_set.games)} game(s), {smash_set.confidence}]"
        )
        for game in smash_set.games:
            if not game.use_for_clips:
                continue
            span = game.duration_sec()
            print(
                f"    Game {game.game_number}  "
                f"{_format_fight_clock(game.start_sec)}–"
                f"{_format_fight_clock(game.end_sec)} ({span:.0f}s)"
            )

            plan = plan_from_smash_game(game, analysis, config)
            if plan is None or not plan.segments:
                print("      skipped (invalid window)")
                continue

            render_work = work_dir / f"smash_set{set_idx:02d}_game{game.game_number:02d}"
            out_path = (
                output_dir
                / f"{vod_id}_set{set_idx}_game{game.game_number}.mp4"
            )
            render_short(video_path, analysis, plan, out_path, config, render_work)
            outputs.append(out_path)
            print(f"      -> {out_path}")

            plan_path = work_dir / f"smash_set{set_idx}_game{game.game_number}_plan.json"
            plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    manifest = {
        "vod_id": vod_id,
        "sets": [
            {
                "set_index": set_idx,
                "format": smash_set.format,
                "games_to_win": smash_set.games_to_win,
                "games": [
                    {
                        "game_number": game.game_number,
                        "start_sec": game.start_sec,
                        "end_sec": game.end_sec,
                        "output": str(
                            output_dir
                            / f"{vod_id}_set{set_idx}_game{game.game_number}.mp4"
                        ),
                    }
                    for game in smash_set.games
                    if game.use_for_clips
                ],
            }
            for set_idx, smash_set in enumerate(store.sets, start=1)
            if smash_set.use_for_clips
        ],
    }
    (work_dir / "smash_clips_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return outputs


def process_cached_clip(
    cache_id: str,
    config: AppConfig,
    clip_title: str = "",
    game_name: str = "",
) -> Path:
    """Re-run analyze + render using data/<cache_id>/ without downloading."""
    data_dir = config.resolve_path(config.paths.data_dir)
    work_dir = data_dir / cache_id
    video_path = find_cached_video(work_dir, cache_id)
    if video_path is None:
        raise FileNotFoundError(
            f"No cached video in {work_dir}. "
            f"Run a normal clip download first (e.g. --clip-url ... --clip-id {cache_id})."
        )

    summary_path = work_dir / "job_summary.json"
    source_url = ""
    if summary_path.exists():
        source_url = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "source_url", ""
        )

    ffmpeg = config.render.ffmpeg_path or "ffmpeg"
    duration = get_video_duration(video_path, ffmpeg=ffmpeg)
    # Long cached streams use chunked VOD analysis + POV montage.
    if duration >= 300 or is_longform_url(source_url):
        outputs = process_vod_url(
            url=source_url or f"cached://{cache_id}",
            config=config,
            vod_id=cache_id,
            vod_title=clip_title,
            skip_download=True,
        )
        return outputs[0]

    return process_clip_url(
        url=source_url or f"cached://{cache_id}",
        config=config,
        clip_id=cache_id,
        clip_title=clip_title,
        game_name=game_name,
        skip_download=True,
    )


def process_twitch_clip(clip: TwitchClip, config: AppConfig) -> Path:
    return process_clip_url(
        url=clip.url,
        config=config,
        clip_id=clip.id,
        clip_title=clip.title,
        game_name=clip.game_name,
    )


def process_twitch_vod(vod: TwitchVod, config: AppConfig, max_shorts: int | None = None) -> list[Path]:
    return process_vod_url(
        url=vod.url,
        config=config,
        vod_id=vod.id,
        vod_title=vod.title,
        max_shorts=max_shorts,
    )
