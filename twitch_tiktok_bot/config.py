"""Load configuration from YAML and environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class TwitchConfig:
    client_id: str = ""
    client_secret: str = ""
    broadcaster_id: str = ""
    max_clips: int = 5


@dataclass
class AnalysisConfig:
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    vision_frame_interval_sec: float = 5.0


@dataclass
class EditingConfig:
    target_duration_sec: float = 30.0
    min_duration_sec: float = 15.0
    max_duration_sec: float = 60.0
    silence_threshold_sec: float = 0.4
    peak_percentile: float = 85.0
    max_zoom_effects: int = 3
    caption_style: str = "bold"
    montage_enabled: bool = True
    max_montage_segments: int = 4
    min_segment_sec: float = 4.0
    max_segment_sec: float = 12.0


@dataclass
class RenderConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    ffmpeg_path: str = ""
    face_crop_enabled: bool = True
    face_sample_count: int = 8


@dataclass
class LLMConfig:
    enabled: bool = False
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = ""
    model: str = "gpt-4o-mini"


@dataclass
class PathsConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    sfx_dir: str = "assets/sfx"


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class AppConfig:
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    editing: EditingConfig = field(default_factory=EditingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    web: WebConfig = field(default_factory=WebConfig)
    project_root: Path = field(default_factory=lambda: Path.cwd())

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.project_root / path


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    load_dotenv()
    root = project_root or Path.cwd()
    paths_to_try = [
        config_path,
        root / "config.local.yaml",
        root / "config.yaml",
    ]
    raw: dict[str, Any] = {}
    for path in paths_to_try:
        if path and path.exists():
            with path.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            break

    # Env overrides for Twitch credentials
    twitch_raw = dict(raw.get("twitch", {}))
    twitch_raw.setdefault("client_id", os.getenv("TWITCH_CLIENT_ID", ""))
    twitch_raw.setdefault("client_secret", os.getenv("TWITCH_CLIENT_SECRET", ""))
    twitch_raw.setdefault("broadcaster_id", os.getenv("TWITCH_BROADCASTER_ID", ""))

    llm_raw = dict(raw.get("llm", {}))
    if os.getenv("OPENAI_API_KEY"):
        llm_raw["enabled"] = llm_raw.get("enabled", True)

    return AppConfig(
        twitch=_merge_dataclass(TwitchConfig, twitch_raw),
        analysis=_merge_dataclass(AnalysisConfig, raw.get("analysis")),
        editing=_merge_dataclass(EditingConfig, raw.get("editing")),
        render=_merge_dataclass(RenderConfig, raw.get("render")),
        llm=_merge_dataclass(LLMConfig, llm_raw),
        paths=_merge_dataclass(PathsConfig, raw.get("paths")),
        web=_merge_dataclass(WebConfig, raw.get("web")),
        project_root=root,
    )
