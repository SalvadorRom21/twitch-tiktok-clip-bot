"""Frame-accurate montage trimming for smooth FFmpeg output."""



from __future__ import annotations



from twitch_tiktok_bot.models import EditEffect, EditSegment





def effect_output_time(effect: EditEffect, segments: list[EditSegment]) -> float:

    """Map an effect timestamp to the montage output timeline."""

    for index, seg in enumerate(segments):

        if seg.start <= effect.time <= seg.end:

            offset = sum(s.end - s.start for s in segments[:index])

            return offset + (effect.time - seg.start)

    return effect.time





def _fps_filter(output_fps: int, source_fps: float | None) -> str:

    """Only resample when source and output FPS differ — avoids redundant judder."""

    if source_fps is not None and abs(source_fps - output_fps) < 0.5:

        return ""

    return f",fps=fps={output_fps}:round=near"





def _snap_segment_frames(

    seg: EditSegment, fps: int

) -> tuple[int, int, float, float]:

    """Snap trim boundaries to whole frames for clean 60fps cuts."""

    start_frame = max(0, int(round(seg.start * fps)))

    end_frame = max(start_frame + 1, int(round(seg.end * fps)))

    start_sec = start_frame / fps

    end_sec = end_frame / fps

    return start_frame, end_frame, start_sec, end_sec





def _trim_segment_filters(

    index: int,

    seg: EditSegment,

    *,

    video_stream: str,

    audio_stream: str,

    fps: int,

    fps_step: str,

    label_suffix: str = "",

) -> tuple[str, str, float]:

    """Build matched video/audio trim filters for one montage segment."""

    start_frame, end_frame, start_sec, end_sec = _snap_segment_frames(seg, fps)

    suffix = label_suffix or str(index)

    video = (

        f"[{video_stream}]trim=start_frame={start_frame}:end_frame={end_frame},"

        f"setpts=PTS-STARTPTS{fps_step}[v{suffix}]"

    )

    audio = (

        f"[{audio_stream}]atrim=start={start_sec:.6f}:end={end_sec:.6f},"

        f"asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[a{suffix}]"

    )

    duration = end_sec - start_sec

    return video, audio, duration





def build_trim_concat_filter(

    segments: list[EditSegment],

    *,

    video_stream: str = "0:v",

    audio_stream: str = "0:a",

    fps: int = 30,

    source_fps: float | None = None,

    crossfade_sec: float = 0.0,

) -> tuple[str, str, str]:

    """

    Trim source segments with frame-aligned boundaries, then join smoothly.



    Uses xfade/acrossfade when crossfade_sec > 0 for multiple segments.



    Returns (filter_graph_prefix, video_label, audio_label).

    """

    if not segments:

        raise ValueError("Montage requires at least one segment")



    fps_step = _fps_filter(fps, source_fps)



    if len(segments) == 1:

        video, audio, _duration = _trim_segment_filters(

            0,

            segments[0],

            video_stream=video_stream,

            audio_stream=audio_stream,

            fps=fps,

            fps_step=fps_step,

            label_suffix="src",

        )

        return f"{video};{audio}", "[vsrc]", "[asrc]"



    durations: list[float] = []

    parts: list[str] = []

    for index, seg in enumerate(segments):

        video, audio, duration = _trim_segment_filters(

            index,

            seg,

            video_stream=video_stream,

            audio_stream=audio_stream,

            fps=fps,

            fps_step=fps_step,

        )

        parts.extend([video, audio])

        durations.append(duration)



    fade = min(

        crossfade_sec,

        min(durations) * 0.35,

        0.5,

    )

    if fade < 0.08:

        concat_inputs = "".join(

            f"[v{index}][a{index}]" for index in range(len(segments))

        )

        parts.append(

            f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[vsrc][asrc]"

        )

        return ";".join(parts), "[vsrc]", "[asrc]"



    v_prev = "[v0]"

    a_prev = "[a0]"

    timeline = durations[0]

    for index in range(1, len(segments)):

        v_next = f"[v{index}]"

        a_next = f"[a{index}]"

        is_last = index == len(segments) - 1

        v_out = "[vsrc]" if is_last else f"[vx{index}]"

        a_out = "[asrc]" if is_last else f"[ax{index}]"

        offset = max(0.0, timeline - fade)

        parts.append(

            f"{v_prev}{v_next}xfade=transition=fade:duration={fade:.3f}:"

            f"offset={offset:.3f}{v_out}"

        )

        parts.append(f"{a_prev}{a_next}acrossfade=d={fade:.3f}:c1=tri:c2=tri{a_out}")

        v_prev = v_out

        a_prev = a_out

        timeline += durations[index] - fade



    return ";".join(parts), "[vsrc]", "[asrc]"





def montage_output_duration(

    segments: list[EditSegment], *, crossfade_sec: float = 0.0, fps: int = 30

) -> float:

    """Estimated montage length after crossfades."""

    if not segments:

        return 0.0

    total = 0.0

    for seg in segments:

        _sf, _ef, start_sec, end_sec = _snap_segment_frames(seg, fps)

        total += end_sec - start_sec

    if len(segments) > 1 and crossfade_sec > 0:

        fade = min(crossfade_sec, 0.5)

        total -= fade * (len(segments) - 1)

    return max(0.0, total)



