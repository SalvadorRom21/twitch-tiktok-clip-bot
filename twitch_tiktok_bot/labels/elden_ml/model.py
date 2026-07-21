"""MobileNetV3-Small frame classifier for Elden Ring UI states."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from twitch_tiktok_bot.labels.elden_ml.config import CLASS_NAMES


def build_model(num_classes: int | None = None, pretrained: bool = True) -> nn.Module:
    num_classes = num_classes or len(CLASS_NAMES)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    try:
        net = models.mobilenet_v3_small(weights=weights)
    except Exception:
        # Offline / first install without weight download.
        net = models.mobilenet_v3_small(weights=None)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    return net


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    class_names: tuple[str, ...] | list[str] = CLASS_NAMES,
    meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(class_names),
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[nn.Module, list[str]]:
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    class_names = list(payload.get("class_names") or CLASS_NAMES)
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
