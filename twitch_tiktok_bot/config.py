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
    max_vods: int = 3


@dataclass
class VodConfig:
    # How many TikTok shorts to generate from one VOD
    max_shorts_per_vod: int = 5
    # Min seconds between highlight windows
    min_short_gap_sec: float = 120.0
    # Audio peak scan chunk size (seconds)
    audio_scan_chunk_sec: float = 600.0
    # Whisper transcription chunk size (seconds)
    whisper_chunk_sec: float = 300.0
    # Max peaks to consider when scoring highlights
    peak_candidate_limit: int = 80
    # Optional: only download first N seconds of VOD (0 = full VOD)
    max_download_sec: float = 0.0


@dataclass
class AnalysisConfig:
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    vision_frame_interval_sec: float = 5.0


@dataclass
class EditingConfig:
    target_duration_sec: float = 28.0
    min_duration_sec: float = 12.0
    max_duration_sec: float = 45.0
    silence_threshold_sec: float = 0.4
    peak_percentile: float = 88.0
    max_zoom_effects: int = 3
    zoom_scale: float = 1.18
    zoom_duration_sec: float = 0.5
    caption_style: str = "bold"
    caption_max_words: int = 7
    captions_enabled: bool = True
    hook_duration_sec: float = 3.5
    hook_first: bool = True
    montage_enabled: bool = True
    # Scan entire source timeline and mash diverse clips into one short
    scan_full_pov: bool = True
    # VOD: one montage from full stream (false = multiple separate shorts)
    pov_montage: bool = True
    max_montage_segments: int = 5
    min_segment_sec: float = 2.5
    max_segment_sec: float = 7.0
    # Keep montage clips within this many seconds of the best moment (smooth cuts)
    montage_cluster_sec: float = 90.0
    # Crossfade between montage segments (0 = hard cut)
    montage_crossfade_sec: float = 0.3
    # auto | apex | generic — boosts highlight detection for your game
    game_profile: str = "auto"
    # Drop montage segments below this action score (higher = stricter)
    min_moment_score: float = 5.0
    # default | apex_shorts — tune pacing from reference YouTube Shorts
    clip_style: str = "default"
    # Strip looting/walking from full fights — stitch action-only micro-cuts
    action_cut_enabled: bool = False
    action_cut_min_fight_span_sec: float = 25.0
    action_cut_window_sec: float = 3.0
    action_cut_scan_step_sec: float = 1.25
    action_cut_min_sec: float = 2.0
    action_cut_min_score: float = 5.0
    action_cut_percentile: float = 38.0
    action_cut_merge_gap_sec: float = 1.5
    action_cut_pad_before_sec: float = 1.5
    action_cut_pad_after_sec: float = 1.5
    action_cut_cold_gap_sec: float = 2.5
    action_cut_split_span_sec: float = 20.0
    action_cut_max_segments: int = 0
    # After team wipe / death box, stop before looting (seconds past wipe cue)
    action_cut_post_wipe_pad_sec: float = 1.5


@dataclass
class RenderConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 60
    # Match source FPS (e.g. 60fps Twitch) up to the fps cap above
    match_source_fps: bool = True
    ffmpeg_path: str = ""
    encode_preset: str = "medium"
    crf: int = 20
    audio_bitrate: str = "192k"
    # stacked = face cam on top + gameplay below; crop = single 9:16 crop
    layout: str = "stacked"
    face_panel_ratio: float = 0.34
    face_crop_enabled: bool = True
    face_sample_count: int = 10
    # auto | top-left | top-right | bottom-left | bottom-right
    face_cam_corner: str = "auto"
    # Optional manual face-cam box (normalized 0-1): x, y, w, h
    face_cam_override: dict[str, float] | None = None


@dataclass
class LLMConfig:
    enabled: bool = False
    # "cursor" uses CURSOR_API_KEY + Cursor SDK (no OpenAI key needed)
    # "openai" uses an OpenAI-compatible API
    provider: str = "cursor"
    api_key_env: str = "OPENAI_API_KEY"
    cursor_api_key_env: str = "CURSOR_API_KEY"
    cursor_model: str = "composer-2.5"
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
    port: int = 8081


@dataclass
class AppConfig:
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
    vod: VodConfig = field(default_factory=VodConfig)
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_if_empty(data: dict[str, Any], key: str, env_name: str) -> None:
    if not str(data.get(key, "")).strip():
        env_val = os.getenv(env_name, "")
        if env_val:
            data[key] = env_val


def apply_game_profile(config: AppConfig, profile: str | None) -> None:
    """Set highlight detection profile for this run (CLI flag or web UI)."""
    if profile is None:
        return
    from twitch_tiktok_bot.plan.game_profiles import normalize_game_profile

    config.editing.game_profile = normalize_game_profile(profile)


def load_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    load_dotenv()
    root = project_root or Path.cwd()

    if config_path and config_path.exists():
        raw = _load_yaml_dict(config_path)
    else:
        raw = {}
        if (root / "config.yaml").exists():
            raw = _load_yaml_dict(root / "config.yaml")
        if (root / "config.local.yaml").exists():
            raw = _deep_merge(raw, _load_yaml_dict(root / "config.local.yaml"))

    # Env overrides for Twitch credentials (fill empty/missing values)
    twitch_raw = dict(raw.get("twitch", {}))
    _apply_env_if_empty(twitch_raw, "client_id", "TWITCH_CLIENT_ID")
    _apply_env_if_empty(twitch_raw, "client_secret", "TWITCH_CLIENT_SECRET")
    _apply_env_if_empty(twitch_raw, "broadcaster_id", "TWITCH_BROADCASTER_ID")

    llm_raw = dict(raw.get("llm", {}))
    provider = str(llm_raw.get("provider", "cursor")).lower()
    if provider == "cursor" and os.getenv(
        llm_raw.get("cursor_api_key_env", "CURSOR_API_KEY"), ""
    ):
        llm_raw.setdefault("enabled", True)
    elif os.getenv(llm_raw.get("api_key_env", "OPENAI_API_KEY"), ""):
        llm_raw.setdefault("enabled", True)

    return AppConfig(
        twitch=_merge_dataclass(TwitchConfig, twitch_raw),
        vod=_merge_dataclass(VodConfig, raw.get("vod")),
        analysis=_merge_dataclass(AnalysisConfig, raw.get("analysis")),
        editing=_merge_dataclass(EditingConfig, raw.get("editing")),
        render=_merge_dataclass(RenderConfig, raw.get("render")),
        llm=_merge_dataclass(LLMConfig, llm_raw),
        paths=_merge_dataclass(PathsConfig, raw.get("paths")),
        web=_merge_dataclass(WebConfig, raw.get("web")),
        project_root=root,
    )
