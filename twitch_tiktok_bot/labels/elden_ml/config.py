"""ML detector config + class taxonomy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Frame classes the MobileNet classifier predicts.
CLASS_NAMES: tuple[str, ...] = ("other", "boss_hud", "you_died", "enemy_felled")
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Human-facing labels for the frame labeler UI.
CLASS_LABELS: list[dict[str, str]] = [
    {"id": "other", "key": "1", "label": "Other (explore / cutscene / trash)"},
    {"id": "boss_hud", "key": "2", "label": "Boss HUD (name + red bar)"},
    {"id": "you_died", "key": "3", "label": "YOU DIED"},
    {"id": "enemy_felled", "key": "4", "label": "ENEMY FELLED"},
]


@dataclass
class MLConfig:
    """Inference / segmentation thresholds for the ML attempt detector."""

    model_path: str = "model.pt"
    image_size: int = 224
    # Classifier confidence floors.
    tau_hud: float = 0.45
    tau_terminal: float = 0.50
    tau_other: float = 0.35
    # Sustained HUD to open an attempt (frames @ sample interval).
    min_hud_frames: int = 2
    # Ignore terminals that fire without a prior HUD window.
    require_hud_before_terminal: bool = True
    terminal_hold_sec: float = 3.0
    min_attempt_sec: float = 8.0
    sample_interval_sec: float = 2.0
    # Last train metrics (informational).
    last_accuracy: float = 0.0
    last_trained_at: str = ""
    labeled_frames: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> MLConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def defaults(cls) -> MLConfig:
        return cls()


def ml_root(data_dir: Path) -> Path:
    return data_dir / "reference" / "elden_ml"


def config_path(data_dir: Path) -> Path:
    return ml_root(data_dir) / "config.json"


def labels_path(data_dir: Path) -> Path:
    return ml_root(data_dir) / "labels.jsonl"


def frames_dir(data_dir: Path) -> Path:
    return ml_root(data_dir) / "frames"


def model_file(data_dir: Path, cfg: MLConfig | None = None) -> Path:
    cfg = cfg or load_ml_config(data_dir)
    root = ml_root(data_dir)
    path = Path(cfg.model_path)
    return path if path.is_absolute() else root / path


def load_ml_config(data_dir: Path) -> MLConfig:
    path = config_path(data_dir)
    if not path.exists():
        return MLConfig.defaults()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MLConfig.from_dict(data.get("config", data))
    except (json.JSONDecodeError, TypeError, ValueError, OSError):
        return MLConfig.defaults()


def save_ml_config(data_dir: Path, cfg: MLConfig, meta: dict | None = None) -> Path:
    root = ml_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = config_path(data_dir)
    payload = {"config": cfg.to_dict(), "classes": list(CLASS_NAMES), **(meta or {})}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def model_ready(data_dir: Path) -> bool:
    return model_file(data_dir).exists()
