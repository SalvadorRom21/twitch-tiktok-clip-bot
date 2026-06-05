"""Render vertical short-form video with FFmpeg."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan


def _ffmpeg_bin(config: AppConfig) -> str:
    return config.render.ffmpeg_path or "ffmpeg"


def _escape_drawtext(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("%", "\\%")
    return escaped


def _build_ass_captions(
    analysis: ClipAnalysis,
    plan: EditPlan,
    ass_path: Path,
) -> None:
    """Write simple ASS subtitles for kept segments."""
    if not analysis.transcript_segments:
        return

    kept = plan.segments[0] if plan.segments else None
    if not kept:
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
    offset = kept.start
    for seg in analysis.transcript_segments:
        if seg.end < kept.start or seg.start > kept.end:
            continue
        rel_start = max(0.0, seg.start - offset)
        rel_end = min(kept.end - kept.start, seg.end - offset)
        text = seg.text.replace("\n", " ").strip()
        if not text:
            continue
        lines.append(
            f"Dialogue: 0,{fmt_time(rel_start)},{fmt_time(rel_end)},Default,,0,0,0,,{text}"
        )

    ass_path.write_text("\n".join(lines), encoding="utf-8")


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
    render = config.render

    if not plan.segments:
        raise ValueError("Edit plan has no segments")

    seg = plan.segments[0]
    duration = max(0.1, seg.end - seg.start)
    ass_path = work_dir / "captions.ass"
    _build_ass_captions(analysis, plan, ass_path)

    plan_path = work_dir / "edit_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    # Vertical crop + scale, trim, captions, hook text
    hook = _escape_drawtext(plan.hook_text or "Watch this")
    vf_parts = [
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        f"scale={render.width}:{render.height}",
    ]

    # Zoom effects on loud moments (simple center zoom via zoompan)
    zoom_filters = []
    for effect in plan.effects:
        if effect.effect_type != "zoom":
            continue
        rel_t = max(0.0, effect.time - seg.start)
        if rel_t > duration:
            continue
        zoom_filters.append(
            f"zoompan=z='if(between(in_time,{rel_t},{rel_t + effect.duration}),"
            f"{effect.scale},1)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={render.width}x{render.height}"
        )

    if zoom_filters:
        vf_parts.extend(zoom_filters[:1])  # MVP: one zoom to avoid filter conflicts

    if ass_path.exists() and ass_path.stat().st_size > 0:
        vf_parts.append(f"ass={ass_path}")

    vf_parts.append(
        f"drawtext=text='{hook}':fontsize=64:fontcolor=white:borderw=4:bordercolor=black:"
        f"x=(w-text_w)/2:y=120:enable='between(t,0,2)'"
    )

    vf = ",".join(vf_parts)

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
        vf,
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
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed:\n{result.stderr}")

    caption_txt = output_path.with_suffix(".txt")
    hashtags = " ".join(f"#{tag}" for tag in plan.hashtags)
    caption_txt.write_text(
        f"{plan.hook_text}\n\n{hashtags}\n", encoding="utf-8"
    )
    return output_path
