"""Speech-to-text using faster-whisper."""

from __future__ import annotations

from pathlib import Path

from twitch_tiktok_bot.models import TranscriptSegment


def transcribe(
    video_path: Path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptSegment]:
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
    return segments
