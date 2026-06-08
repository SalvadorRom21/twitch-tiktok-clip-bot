"""Render vertical short-form video with FFmpeg."""

from __future__ import annotations

import json
import platform
import subprocess
import textwrap
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditEffect, EditPlan, EditSegment
from twitch_tiktok_bot.analyze.face_cam import FaceCamRegion, resolve_face_cam_for_render
from twitch_tiktok_bot.plan.moments import PAYOFF_WORDS, SETUP_WORDS
from twitch_tiktok_bot.render.crop import crop_x_expression
from twitch_tiktok_bot.render.montage import (
    _fps_filter,
    build_trim_concat_filter,
    effect_output_time,
    montage_output_duration,
)
from twitch_tiktok_bot.progress import run_ffmpeg, step
from twitch_tiktok_bot.render.stacked import build_stacked_filter


def _ffmpeg_bin(config: AppConfig) -> str:
    return config.render.ffmpeg_path or "ffmpeg"


def _ffprobe_bin(config: AppConfig) -> str:
    ffmpeg = _ffmpeg_bin(config)
    if ffmpeg.lower().endswith("ffmpeg.exe"):
        return ffmpeg[:-10] + "ffprobe.exe"
    if ffmpeg.lower().endswith("ffmpeg"):
        return ffmpeg[:-6] + "ffprobe"
    return "ffprobe"


def _probe_video_fps(video_path: Path, config: AppConfig) -> float | None:
    """Read the source video frame rate (e.g. 60 for Twitch VODs)."""
    cmd = [
        _ffprobe_bin(config),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = streams[0].get(key, "")
            if not raw or raw == "0/0":
                continue
            if "/" in raw:
                num, den = raw.split("/", 1)
                den_f = float(den)
                if den_f:
                    fps = float(num) / den_f
                    if fps > 1:
                        return fps
            else:
                fps = float(raw)
                if fps > 1:
                    return fps
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, OSError):
        return None
    return None


def _resolve_output_fps(config: AppConfig, source_fps: float | None) -> int:
    """Keep source FPS when possible so 60fps gameplay stays smooth."""
    cap = max(24, int(config.render.fps))
    if config.render.match_source_fps and source_fps is not None:
        return min(cap, max(24, round(source_fps)))
    return cap


def _escape_filter_path(path: Path) -> str:
    """Escape a Windows/Unix path for FFmpeg filter arguments (ass, subtitles, etc.)."""
    posix = path.resolve().as_posix()
    if len(posix) >= 2 and posix[1] == ":":
        posix = posix[0] + "\\:" + posix[2:]
    posix = posix.replace("'", "\\'")
    return f"'{posix}'"


def _escape_drawtext(text: str) -> str:
    normalized = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u2032", "'")
    )
    escaped = normalized.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("%", "\\%")
    return escaped


def _video_encode_args(config: AppConfig, output_fps: int) -> list[str]:
    render = config.render
    return [
        "-c:v",
        "libx264",
        "-preset",
        render.encode_preset,
        "-crf",
        str(render.crf),
        "-fps_mode",
        "cfr",
        "-r",
        str(output_fps),
        "-g",
        str(output_fps * 2),
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
    ]


def _audio_encode_args(config: AppConfig) -> list[str]:
    return [
        "-c:a",
        "aac",
        "-b:a",
        config.render.audio_bitrate,
        "-ar",
        "48000",
    ]


def _segment_timeline(plan: EditPlan) -> list[tuple[EditSegment, float]]:
    """Map each edit segment to its start offset in the output video."""
    timeline: list[tuple[EditSegment, float]] = []
    offset = 0.0
    for seg in plan.segments:
        timeline.append((seg, offset))
        offset += seg.end - seg.start
    return timeline


def _caption_text(text: str, max_words: int) -> str | None:
    text = text.replace("\n", " ").strip()
    if not text or SETUP_WORDS.search(text):
        return None
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.upper() if PAYOFF_WORDS.search(text) else text


def _build_ass_captions(
    analysis: ClipAnalysis,
    plan: EditPlan,
    ass_path: Path,
    max_words: int = 7,
) -> None:
    """Write ASS subtitles aligned to montage or single-segment output."""
    if not analysis.transcript_segments or not plan.segments:
        return

    header = textwrap.dedent(
        """\
        [Script Info]
        ScriptType: v4.00+
        PlayResX: 1080
        PlayResY: 1920

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,0,2,40,40,220,1

        [Events]
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        """
    )

    def fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:01d}:{m:02d}:{s:05.2f}"

    lines = [header]
    for edit_seg, offset in _segment_timeline(plan):
        seg_len = edit_seg.end - edit_seg.start
        for transcript in analysis.transcript_segments:
            if transcript.end < edit_seg.start or transcript.start > edit_seg.end:
                continue
            rel_start = offset + max(0.0, transcript.start - edit_seg.start)
            rel_end = offset + min(
                seg_len, max(transcript.end - edit_seg.start, 0.1)
            )
            if rel_end <= rel_start:
                continue
            text = _caption_text(transcript.text, max_words)
            if not text:
                continue
            lines.append(
                f"Dialogue: 0,{fmt_time(rel_start)},{fmt_time(rel_end)},"
                f"Default,,0,0,0,,{text}"
            )

    ass_path.write_text("\n".join(lines), encoding="utf-8")


def _output_timeline_effects(
    effects: list[EditEffect], segments: list[EditSegment]
) -> list[EditEffect]:
    mapped: list[EditEffect] = []
    for effect in effects:
        mapped.append(
            EditEffect(
                time=effect_output_time(effect, segments),
                effect_type=effect.effect_type,
                scale=effect.scale,
                duration=effect.duration,
                text=effect.text,
                asset=effect.asset,
                volume=effect.volume,
            )
        )
    return mapped


def _overlay_video_filters(
    ass_path: Path | None,
    hook_text: str,
    config: AppConfig,
) -> tuple[str, str]:
    """Return (filter_suffix, output_label) for captions + hook on [vout]."""
    if not config.editing.captions_enabled:
        return "", "[vout]"

    vf_parts: list[str] = []
    if ass_path and ass_path.exists() and ass_path.stat().st_size > 0:
        vf_parts.append(f"ass={_escape_filter_path(ass_path)}")
    hook = _escape_drawtext(hook_text or "Watch this")
    hook_dur = config.editing.hook_duration_sec
    drawtext = (
        f"drawtext=text='{hook}':fontsize=72:fontcolor=white:borderw=5:bordercolor=black:"
        f"x=(w-text_w)/2:y=140:enable='between(t,0,{hook_dur})'"
    )
    if platform.system() == "Windows":
        arial = Path("C:/Windows/Fonts/arial.ttf")
        if arial.exists():
            drawtext = drawtext.replace(
                "drawtext=text=",
                f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text=",
                1,
            )
    vf_parts.append(drawtext)
    if not vf_parts:
        return "", "[vout]"
    return ",".join(vf_parts), "[vfinal]"


def _build_render_filter_complex(
    config: AppConfig,
    segments: list[EditSegment],
    effects: list[EditEffect],
    face_center_x: float | None,
    face_cam_region: FaceCamRegion | None,
    *,
    output_fps: int,
    source_fps: float | None = None,
    ass_path: Path | None = None,
    hook_text: str = "",
) -> tuple[str, str]:
    """Single-pass trim → concat → layout → captions for smooth CFR output."""
    render = config.render
    crossfade = config.editing.montage_crossfade_sec if len(segments) > 1 else 0.0
    trim_graph, video_in, _audio_in = build_trim_concat_filter(
        segments,
        fps=output_fps,
        source_fps=source_fps,
        crossfade_sec=crossfade,
    )
    output_effects = _output_timeline_effects(effects, segments)

    use_stacked = (
        render.layout.lower() == "stacked" and face_cam_region is not None
    )

    if use_stacked:
        layout_graph = build_stacked_filter(
            config,
            face_cam_region,
            output_effects,
            video_in=video_in,
        )
        base_graph = f"{trim_graph};{layout_graph}"
    else:
        crop_x = crop_x_expression(face_center_x)
        fps_step = _fps_filter(output_fps, source_fps)
        layout_graph = (
            f"{video_in}crop=ih*9/16:ih:{crop_x}:0,"
            f"scale={render.width}:{render.height}:flags=lanczos,"
            f"setsar=1,format=yuv420p{fps_step}[vout]"
        )
        base_graph = f"{trim_graph};{layout_graph}"

    overlay_filters, video_label = _overlay_video_filters(
        ass_path, hook_text, config
    )
    if overlay_filters:
        return f"{base_graph};[vout]{overlay_filters}{video_label}", video_label
    return base_graph, "[vout]"


def _render_montage(
    video_path: Path,
    output_path: Path,
    config: AppConfig,
    segments: list[EditSegment],
    effects: list[EditEffect],
    face_center_x: float | None,
    face_cam_region: FaceCamRegion | None,
    *,
    output_fps: int,
    source_fps: float | None = None,
    ass_path: Path | None = None,
    hook_text: str = "",
    duration_sec: float | None = None,
) -> None:
    """
    Render the full montage in one FFmpeg pass (layout + captions + encode).

    Frame-accurate trim/concat avoids choppy keyframe seeks and timestamp
    glitches from multi-file concat copy.
    """
    ffmpeg = _ffmpeg_bin(config)
    filter_complex, video_label = _build_render_filter_complex(
        config,
        segments,
        effects,
        face_center_x,
        face_cam_region,
        output_fps=output_fps,
        source_fps=source_fps,
        ass_path=ass_path,
        hook_text=hook_text,
    )
    cmd = [
        ffmpeg,
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        str(video_path),
        "-filter_complex",
        filter_complex,
        "-map",
        video_label,
        "-map",
        "[asrc]",
        *_video_encode_args(config, output_fps),
        *_audio_encode_args(config),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    preset = config.render.encode_preset
    run_ffmpeg(
        cmd,
        label=f"encoding montage ({preset}, {output_fps}fps)",
        duration_sec=duration_sec,
    )


def render_short(
    video_path: Path,
    analysis: ClipAnalysis,
    plan: EditPlan,
    output_path: Path,
    config: AppConfig,
    work_dir: Path,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not plan.segments:
        raise ValueError("Edit plan has no segments")

    ass_path: Path | None = None
    if config.editing.captions_enabled:
        ass_path = work_dir / "captions.ass"
        _build_ass_captions(
            analysis, plan, ass_path, max_words=config.editing.caption_max_words
        )

    plan_path = work_dir / "edit_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    face_center, face_region = resolve_face_cam_for_render(analysis, config)
    if config.render.layout.lower() == "stacked" and face_region is not None:
        print(
            f"       stacked layout: face panel "
            f"x={face_region.x:.2f} y={face_region.y:.2f} "
            f"({face_region.w:.0%}×{face_region.h:.0%})"
        )

    source_fps = _probe_video_fps(video_path, config)
    output_fps = _resolve_output_fps(config, source_fps)
    crossfade = (
        config.editing.montage_crossfade_sec if len(plan.segments) > 1 else 0.0
    )
    montage_duration = montage_output_duration(
        plan.segments,
        crossfade_sec=crossfade,
        fps=output_fps,
    )
    if source_fps:
        seg_span = 0.0
        if plan.segments:
            seg_span = plan.segments[-1].end - plan.segments[0].start
        print(
            f"       source {source_fps:.0f}fps -> output {output_fps}fps, "
            f"{len(plan.segments)} clips in {seg_span:.0f}s span, "
            f"{crossfade:.2f}s crossfade"
        )

    with step(
        f"  encoding video ({config.render.encode_preset}, crf {config.render.crf}, "
        f"{output_fps}fps)",
        heartbeat_sec=30,
    ):
        _render_montage(
            video_path,
            output_path,
            config,
            plan.segments,
            plan.effects,
            face_center,
            face_region,
            output_fps=output_fps,
            source_fps=source_fps,
            ass_path=ass_path,
            hook_text=plan.hook_text,
            duration_sec=montage_duration,
        )

    caption_txt = output_path.with_suffix(".txt")
    hashtags = " ".join(f"#{tag}" for tag in plan.hashtags)
    caption_txt.write_text(
        f"{plan.hook_text}\n\n{hashtags}\n", encoding="utf-8"
    )
    return output_path
