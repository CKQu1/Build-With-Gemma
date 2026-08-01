from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value < self.best_loss:
            self.best_loss = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unsupported loss: {name}")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

