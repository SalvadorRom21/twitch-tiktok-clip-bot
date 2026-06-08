"""Speech-to-text using faster-whisper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from twitch_tiktok_bot.analyze.cuda_dlls import ensure_cuda_dll_paths
from twitch_tiktok_bot.models import TranscriptSegment


def _run_transcribe(
    video_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
    on_progress: Callable[[str], None] | None = None,
) -> list[TranscriptSegment]:
    if device == "cuda":
        ensure_cuda_dll_paths()

    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, _info = model.transcribe(
        str(video_path),
        vad_filter=True,
        word_timestamps=False,
    )
    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text)
        )
        if on_progress and len(segments) % 2 == 0:
            last = segments[-1]
            on_progress(
                f"transcribed {len(segments)} lines (up to {last.end:.0f}s)"
            )
    if on_progress and segments:
        on_progress(f"finished — {len(segments)} speech segments")
    return segments


def transcribe(
    video_path: Path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    on_progress: Callable[[str], None] | None = None,
) -> list[TranscriptSegment]:
    try:
        return _run_transcribe(
            video_path, model_size, device, compute_type, on_progress=on_progress
        )
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(
            f"  [analyze] CUDA whisper failed ({exc}); falling back to CPU int8..."
        )
        return _run_transcribe(
            video_path, model_size, "cpu", "int8", on_progress=on_progress
        )
