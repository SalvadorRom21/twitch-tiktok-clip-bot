"""Run Elden ML frame classifier on images / scan caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable

import cv2
import numpy as np
import torch
from torchvision import transforms

from twitch_tiktok_bot.labels.elden_ml.config import (
    CLASS_NAMES,
    MLConfig,
    load_ml_config,
    model_file,
    model_ready,
)
from twitch_tiktok_bot.labels.elden_ml.crop import crop_gameplay
from twitch_tiktok_bot.labels.elden_ml.model import load_checkpoint

_lock = Lock()
_cache: dict[str, tuple[object, list[str], torch.device]] = {}


@dataclass
class FramePrediction:
    time_sec: float
    label: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_model(data_dir: Path, cfg: MLConfig | None = None):
    cfg = cfg or load_ml_config(data_dir)
    path = model_file(data_dir, cfg)
    key = str(path.resolve())
    with _lock:
        if key in _cache and path.exists():
            return _cache[key]
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Label frames and click Train model first."
            )
        device = _device()
        model, class_names = load_checkpoint(path, device=device)
        model.eval()
        _cache[key] = (model, class_names, device)
        return _cache[key]


def clear_model_cache() -> None:
    with _lock:
        _cache.clear()


def predict_bgr(
    frame_bgr: np.ndarray,
    data_dir: Path,
    *,
    time_sec: float = 0.0,
    cfg: MLConfig | None = None,
) -> FramePrediction:
    cfg = cfg or load_ml_config(data_dir)
    model, class_names, device = get_model(data_dir, cfg)
    crop = crop_gameplay(frame_bgr)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = _transform(cfg.image_size)(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs_t = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs = {class_names[i]: float(probs_t[i]) for i in range(len(class_names))}
    # Prefer mapped CLASS_NAMES order when present.
    best_i = int(np.argmax(probs_t))
    label = class_names[best_i]
    conf = float(probs_t[best_i])
    return FramePrediction(time_sec=time_sec, label=label, confidence=conf, probs=probs)


def resolve_label(pred: FramePrediction, cfg: MLConfig) -> str:
    """Apply confidence floors — low-confidence non-other → other."""
    probs = pred.probs
    hud = probs.get("boss_hud", 0.0)
    died = probs.get("you_died", 0.0)
    felled = probs.get("enemy_felled", 0.0)
    other = probs.get("other", 0.0)

    # Terminals first (mutually exclusive with HUD for segmentation).
    if died >= cfg.tau_terminal and died >= felled and died >= hud:
        return "you_died"
    if felled >= cfg.tau_terminal and felled >= died and felled >= hud:
        return "enemy_felled"
    if hud >= cfg.tau_hud and hud >= other:
        return "boss_hud"
    return "other"


def _empty_pred(time_sec: float) -> FramePrediction:
    return FramePrediction(
        time_sec=time_sec,
        label="other",
        confidence=0.0,
        probs={n: 0.0 for n in CLASS_NAMES},
    )


def _predict_path(
    path: Path,
    data_dir: Path,
    *,
    time_sec: float,
    cfg: MLConfig,
) -> FramePrediction:
    img = cv2.imread(str(path))
    if img is None:
        return _empty_pred(time_sec)
    raw = predict_bgr(img, data_dir, time_sec=time_sec, cfg=cfg)
    label = resolve_label(raw, cfg)
    return FramePrediction(
        time_sec=time_sec,
        label=label,
        confidence=raw.probs.get(label, raw.confidence),
        probs=raw.probs,
    )


def _tensor_from_path(path: Path, tfm) -> torch.Tensor | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    crop = crop_gameplay(img)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return tfm(rgb)


def predict_scan_frames(
    work_dir: Path,
    data_dir: Path,
    *,
    interval_sec: float = 2.0,
    cfg: MLConfig | None = None,
    batch_size: int | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> list[FramePrediction]:
    """Classify boss_scan_frames + edge_probes (batched inference)."""
    cfg = cfg or load_ml_config(data_dir)
    frames_path = work_dir / "boss_scan_frames"
    paths: list[tuple[float, Path]] = []
    if frames_path.is_dir():
        idx = 0
        while True:
            p = frames_path / f"frame_{idx:05d}.jpg"
            if not p.exists():
                break
            paths.append((idx * interval_sec, p))
            idx += 1
    # Dense start/end/terminal probes encode time as tenths in the filename.
    probe_dir = work_dir / "edge_probes"
    if probe_dir.is_dir():
        for p in sorted(probe_dir.glob("*.jpg")):
            stem = p.stem
            for prefix in ("start_", "probe_", "term_", "end_"):
                if stem.startswith(prefix):
                    try:
                        t = int(stem[len(prefix) :]) / 10.0
                    except ValueError:
                        break
                    paths.append((t, p))
                    break
    # Prefer scan-grid entry when times collide.
    by_t: dict[float, Path] = {}
    for t, p in paths:
        key = round(t, 1)
        if key not in by_t or "boss_scan_frames" in str(p).replace("\\", "/"):
            by_t[key] = p
    ordered = sorted(by_t.items(), key=lambda kv: kv[0])
    total = max(1, len(ordered))
    if not ordered:
        return []

    model, class_names, device = get_model(data_dir, cfg)
    tfm = _transform(cfg.image_size)
    if batch_size is None:
        batch_size = 64 if device.type == "cuda" else 32

    out: list[FramePrediction] = []
    done = 0
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        keep: list[tuple[int, float]] = []
        empties: dict[int, FramePrediction] = {}
        for local_i, (t, path) in enumerate(chunk):
            tensor = _tensor_from_path(path, tfm)
            if tensor is None:
                empties[local_i] = _empty_pred(t)
            else:
                tensors.append(tensor)
                keep.append((local_i, t))

        probs_by_local: dict[int, np.ndarray] = {}
        if tensors:
            batch = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad():
                logits = model(batch)
                probs_np = torch.softmax(logits, dim=1).cpu().numpy()
            for row, (local_i, _t) in enumerate(keep):
                probs_by_local[local_i] = probs_np[row]

        for local_i, (t, _path) in enumerate(chunk):
            if local_i in empties:
                out.append(empties[local_i])
                continue
            probs_t = probs_by_local[local_i]
            probs = {class_names[i]: float(probs_t[i]) for i in range(len(class_names))}
            raw = FramePrediction(
                time_sec=t,
                label=class_names[int(np.argmax(probs_t))],
                confidence=float(np.max(probs_t)),
                probs=probs,
            )
            label = resolve_label(raw, cfg)
            out.append(
                FramePrediction(
                    time_sec=t,
                    label=label,
                    confidence=raw.probs.get(label, raw.confidence),
                    probs=raw.probs,
                )
            )

        done = min(len(ordered), start + len(chunk))
        if on_progress and (done % max(10, batch_size) < batch_size or done >= total):
            on_progress(
                {
                    "phase": "infer",
                    "message": f"Classifying frames ({done}/{total})",
                    "current": done,
                    "total": total,
                    "pct": round((done / total) * 100.0, 1),
                }
            )
    return out


def model_is_ready(data_dir: Path) -> bool:
    return model_ready(data_dir)
