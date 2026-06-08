"""Shared data models for the clip pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class LoudPeak:
    time: float
    score: float


@dataclass
class TimeRange:
    start: float
    end: float


@dataclass
class SceneChange:
    time: float


@dataclass
class VisionFrame:
    time: float
    description: str


@dataclass
class ClipAnalysis:
    duration: float
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)
    loud_peaks: list[LoudPeak] = field(default_factory=list)
    silence_ranges: list[TimeRange] = field(default_factory=list)
    scene_changes: list[SceneChange] = field(default_factory=list)
    vision_frames: list[VisionFrame] = field(default_factory=list)
    vision_summary: str = ""
    clip_title: str = ""
    game_name: str = ""
    # Normalized 0-1 horizontal face center for crop; None = center crop
    face_crop_center_x: float | None = None
    # Face-cam overlay box (x, y, w, h) for stacked layout
    face_cam_region: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "transcript_segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in self.transcript_segments
            ],
            "loud_peaks": [{"t": p.time, "score": p.score} for p in self.loud_peaks],
            "silence_ranges": [
                {"start": r.start, "end": r.end} for r in self.silence_ranges
            ],
            "scene_changes": [{"t": s.time} for s in self.scene_changes],
            "vision_frames": [
                {"t": v.time, "description": v.description} for v in self.vision_frames
            ],
            "vision_summary": self.vision_summary,
            "clip_title": self.clip_title,
            "game_name": self.game_name,
            "face_crop_center_x": self.face_crop_center_x,
            "face_cam_region": self.face_cam_region,
        }


@dataclass
class EditSegment:
    start: float
    end: float
    reason: str = ""


@dataclass
class EditEffect:
    time: float
    effect_type: str
    scale: float = 1.0
    duration: float = 0.8
    text: str = ""
    asset: str = ""
    volume: float = 0.3


@dataclass
class EditPlan:
    target_duration_sec: float
    segments: list[EditSegment]
    effects: list[EditEffect] = field(default_factory=list)
    hook_text: str = ""
    hashtags: list[str] = field(default_factory=list)
    caption_style: str = "bold"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_duration_sec": self.target_duration_sec,
            "segments": [
                {"start": s.start, "end": s.end, "reason": s.reason}
                for s in self.segments
            ],
            "effects": [
                {
                    "t": e.time,
                    "type": e.effect_type,
                    "scale": e.scale,
                    "duration": e.duration,
                    "text": e.text,
                    "asset": e.asset,
                    "volume": e.volume,
                }
                for e in self.effects
            ],
            "hook_text": self.hook_text,
            "hashtags": self.hashtags,
            "caption_style": self.caption_style,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditPlan:
        return cls(
            target_duration_sec=float(data.get("target_duration_sec", 30)),
            segments=[
                EditSegment(
                    start=float(s["start"]),
                    end=float(s["end"]),
                    reason=s.get("reason", ""),
                )
                for s in data.get("segments", [])
            ],
            effects=[
                EditEffect(
                    time=float(e["t"]),
                    effect_type=e["type"],
                    scale=float(e.get("scale", 1.0)),
                    duration=float(e.get("duration", 0.8)),
                    text=e.get("text", ""),
                    asset=e.get("asset", ""),
                    volume=float(e.get("volume", 0.3)),
                )
                for e in data.get("effects", [])
            ],
            hook_text=data.get("hook_text", ""),
            hashtags=list(data.get("hashtags", [])),
            caption_style=data.get("caption_style", "bold"),
        )


@dataclass
class TwitchClip:
    id: str
    url: str
    title: str
    game_id: str = ""
    game_name: str = ""
    view_count: int = 0
    created_at: str = ""


@dataclass
class TwitchVod:
    id: str
    url: str
    title: str
    duration_sec: float = 0.0
    view_count: int = 0
    created_at: str = ""
