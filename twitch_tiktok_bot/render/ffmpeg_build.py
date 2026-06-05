"""Render vertical short-form video with FFmpeg."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditEffect, EditPlan, EditSegment
from twitch_tiktok_bot.render.crop import crop_x_expression


def _ffmpeg_bin(config: AppConfig) -> str:
    return config.render.ffmpeg_path or "ffmpeg"


def _escape_drawtext(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("%", "\\%")
    return escaped


def _segment_timeline(plan: EditPlan) -> list[tuple[EditSegment, float]]:
    """Map each edit segment to its start offset in the output video."""
    timeline: list[tuple[EditSegment, float]] = []
    offset = 0.0
    for seg in plan.segments:
        timeline.append((seg, offset))
        offset += seg.end - seg.start
    return timeline


def _build_ass_captions(
    analysis: ClipAnalysis,
    plan: EditPlan,
    ass_path: Path,
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
            text = transcript.text.replace("\n", " ").strip()
            if not text:
                continue
            lines.append(
                f"Dialogue: 0,{fmt_time(rel_start)},{fmt_time(rel_end)},"
                f"Default,,0,0,0,,{text}"
            )

    ass_path.write_text("\n".join(lines), encoding="utf-8")


def _zoom_filter_for_segment(
    effects: list[EditEffect],
    seg: EditSegment,
    render_width: int,
    render_height: int,
) -> str | None:
    for effect in effects:
        if effect.effect_type != "zoom":
            continue
        if not (seg.start <= effect.time <= seg.end):
            continue
        rel_t = effect.time - seg.start
        return (
            f"zoompan=z='if(between(in_time,{rel_t},{rel_t + effect.duration}),"
            f"{effect.scale},1)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"s={render_width}x{render_height}"
        )
    return None


def _render_segment_clip(
    video_path: Path,
    seg: EditSegment,
    output_path: Path,
    config: AppConfig,
    face_center_x: float | None,
    effects: list[EditEffect],
) -> None:
    ffmpeg = _ffmpeg_bin(config)
    render = config.render
    duration = max(0.1, seg.end - seg.start)
    crop_x = crop_x_expression(face_center_x)

    vf_parts = [
        f"crop=ih*9/16:ih:{crop_x}:0",
        f"scale={render.width}:{render.height}",
    ]
    zoom = _zoom_filter_for_segment(effects, seg, render.width, render.height)
    if zoom:
        vf_parts.append(zoom)

    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{seg.start}",
        "-i",
        str(video_path),
        "-t",
        f"{duration}",
        "-vf",
        ",".join(vf_parts),
        "-r",
        str(render.fps),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment render failed:\n{result.stderr}")


def _concat_segments(segment_paths: list[Path], output_path: Path, ffmpeg: str) -> None:
    list_path = output_path.parent / "concat_list.txt"
    lines = [f"file '{path.resolve()}'" for path in segment_paths]
    list_path.write_text("\n".join(lines), encoding="utf-8")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")


def _apply_overlays(
    input_path: Path,
    output_path: Path,
    ass_path: Path | None,
    hook_text: str,
    config: AppConfig,
) -> None:
    ffmpeg = _ffmpeg_bin(config)
    hook = _escape_drawtext(hook_text or "Watch this")
    vf_parts: list[str] = []
    if ass_path and ass_path.exists() and ass_path.stat().st_size > 0:
        vf_parts.append(f"ass={ass_path}")
    vf_parts.append(
        f"drawtext=text='{hook}':fontsize=64:fontcolor=white:borderw=4:bordercolor=black:"
        f"x=(w-text_w)/2:y=120:enable='between(t,0,2)'"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        ",".join(vf_parts),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg overlay failed:\n{result.stderr}")


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
    ffmpeg = _ffmpeg_bin(config)

    if not plan.segments:
        raise ValueError("Edit plan has no segments")

    ass_path = work_dir / "captions.ass"
    _build_ass_captions(analysis, plan, ass_path)

    plan_path = work_dir / "edit_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    face_center = analysis.face_crop_center_x
    segments_dir = work_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    segment_paths: list[Path] = []
    for idx, seg in enumerate(plan.segments):
        seg_path = segments_dir / f"part_{idx:02d}.mp4"
        _render_segment_clip(
            video_path,
            seg,
            seg_path,
            config,
            face_center,
            plan.effects,
        )
        segment_paths.append(seg_path)

    raw_path = work_dir / "raw_montage.mp4"
    if len(segment_paths) == 1:
        raw_path = segment_paths[0]
    else:
        _concat_segments(segment_paths, raw_path, ffmpeg)

    _apply_overlays(
        raw_path,
        output_path,
        ass_path,
        plan.hook_text,
        config,
    )

    caption_txt = output_path.with_suffix(".txt")
    hashtags = " ".join(f"#{tag}" for tag in plan.hashtags)
    caption_txt.write_text(
        f"{plan.hook_text}\n\n{hashtags}\n", encoding="utf-8"
    )
    return output_path
