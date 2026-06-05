"""Lightweight vision analysis via sampled frames."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from twitch_tiktok_bot.models import VisionFrame


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_sec: float = 5.0,
    ffmpeg: str = "ffmpeg",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%04d.jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_sec}",
        "-q:v",
        "3",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    return sorted(output_dir.glob("frame_*.jpg"))


def describe_frames_simple(frame_paths: list[Path], interval_sec: float) -> list[VisionFrame]:
    """Heuristic descriptions without an external vision API."""
    frames: list[VisionFrame] = []
    for idx, path in enumerate(frame_paths):
        t = idx * interval_sec
        frames.append(
            VisionFrame(
                time=t,
                description=f"Frame at {t:.0f}s from clip ({path.name})",
            )
        )
    return frames


def describe_frames_llm(
    frame_paths: list[Path],
    interval_sec: float,
    api_key: str,
    model: str,
    base_url: str = "",
) -> list[VisionFrame]:
    """Optional OpenAI-compatible vision descriptions for sampled frames."""
    from openai import OpenAI

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    frames: list[VisionFrame] = []
    for idx, path in enumerate(frame_paths[:8]):  # cap cost
        t = idx * interval_sec
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this Twitch stream frame in one short sentence. "
                                "Mention gameplay, reactions, or UI if visible."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=80,
        )
        description = (response.choices[0].message.content or "").strip()
        frames.append(VisionFrame(time=t, description=description))
    return frames


def build_vision_summary(frames: list[VisionFrame]) -> str:
    if not frames:
        return ""
    unique = []
    seen: set[str] = set()
    for frame in frames:
        key = frame.description.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame.description)
    return " ".join(unique[:3])
