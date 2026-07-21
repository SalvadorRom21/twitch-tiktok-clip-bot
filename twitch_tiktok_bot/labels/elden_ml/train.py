"""Train MobileNetV3 on labeled Elden Ring frames."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from twitch_tiktok_bot.labels.elden_ml.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    MLConfig,
    frames_dir,
    load_ml_config,
    model_file,
    save_ml_config,
)
from twitch_tiktok_bot.labels.elden_ml.dataset import label_counts, read_labels
from twitch_tiktok_bot.labels.elden_ml.model import build_model, save_checkpoint


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, image_size: int, train: bool):
        self.rows = rows
        self.root = root
        self.image_size = image_size
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.tf = transforms.Compose(
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self.root / row["path"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            # Fallback black image so training does not crash on a missing file.
            rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb)
        y = CLASS_TO_IDX[row["label"]]
        return x, y


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    data_dir: Path,
    *,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    min_labels: int = 20,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Train classifier from labels.jsonl. Raises if too few labels."""
    rows = [r for r in read_labels(data_dir) if r.get("label") in CLASS_TO_IDX]
    counts = label_counts(data_dir)
    if len(rows) < min_labels:
        raise ValueError(
            f"Need at least {min_labels} labeled frames to train (have {len(rows)}). "
            "Label boss_hud / you_died / enemy_felled / other in the trainer."
        )
    # Prefer at least 2 classes present.
    present = sum(1 for c in counts.values() if c > 0)
    if present < 2:
        raise ValueError("Need labels for at least 2 classes before training.")

    cfg = load_ml_config(data_dir)

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac)) if len(rows) >= 10 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:] or rows

    # Class weights for imbalance.
    class_counts = [max(1, counts[name]) for name in CLASS_NAMES]
    total = float(sum(class_counts))
    weights = [total / (len(CLASS_NAMES) * c) for c in class_counts]

    device = _pick_device()
    train_ds = FrameDataset(train_rows, frames_dir(data_dir), cfg.image_size, train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = None
    if val_rows:
        val_ds = FrameDataset(val_rows, frames_dir(data_dir), cfg.image_size, train=False)
        val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    def _emit(fields: dict) -> None:
        if on_progress:
            on_progress(fields)

    best_acc = -1.0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            running_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total_n += xb.size(0)
        train_loss = running_loss / max(1, total_n)
        train_acc = correct / max(1, total_n)

        val_acc = train_acc
        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    pred = logits.argmax(dim=1)
                    v_correct += int((pred == yb).sum().item())
                    v_total += xb.size(0)
            val_acc = v_correct / max(1, v_total)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
            }
        )
        _emit(
            {
                "phase": "train",
                "message": f"Epoch {epoch}/{epochs} — val acc {val_acc:.0%}",
                "current": epoch,
                "total": epochs,
                "pct": round((epoch / epochs) * 100.0, 1),
                "train_loss": train_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            out = model_file(data_dir, cfg)
            save_checkpoint(
                out,
                model,
                class_names=CLASS_NAMES,
                meta={
                    "val_acc": best_acc,
                    "labeled_frames": len(rows),
                    "counts": counts,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    cfg.last_accuracy = float(best_acc)
    cfg.last_trained_at = datetime.now(timezone.utc).isoformat()
    cfg.labeled_frames = len(rows)
    save_ml_config(
        data_dir,
        cfg,
        meta={"mode": "ml", "history": history[-5:], "counts": counts},
    )
    return {
        "status": "done",
        "accuracy": best_acc,
        "epochs": epochs,
        "labeled_frames": len(rows),
        "counts": counts,
        "device": str(device),
        "model_path": str(model_file(data_dir, cfg)),
        "history": history,
    }
