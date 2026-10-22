"""MobileNetV2 landmark regressor: 224x224x3 image -> 98 x (x, y) coordinates in [0, 1]."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .landmarks import NUM_LANDMARKS

CHECKPOINT_FORMAT = 1


class PretrainedWeightsError(RuntimeError):
    """Raised instead of silently training from random weights when ImageNet weights fail to load."""


class LandmarkRegressor(nn.Module):
    """MobileNetV2 features -> global average pooling -> Linear-ReLU-Dropout-Linear -> 196 values."""

    def __init__(self, num_landmarks: int = NUM_LANDMARKS, hidden_dim: int = 512,
                 dropout: float = 0.2, weights: str | None = "IMAGENET1K_V2"):
        super().__init__()
        self.num_landmarks = num_landmarks
        backbone = mobilenet_v2(weights=_resolve_weights(weights))
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.last_channel, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_landmarks * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).view(-1, self.num_landmarks, 2)

    def init_mean_shape(self, mean_shape: np.ndarray) -> None:
        """Start the head at the mean face so early epochs refine a shape instead of finding one."""
        last: nn.Linear = self.head[-1]
        nn.init.normal_(last.weight, std=1e-3)
        with torch.no_grad():
            last.bias.copy_(torch.as_tensor(mean_shape, dtype=torch.float32).reshape(-1))

    def parameter_groups(self, lr: float, head_lr_mult: float) -> list[dict[str, Any]]:
        return [
            {"params": self.features.parameters(), "lr": lr},
            {"params": self.head.parameters(), "lr": lr * head_lr_mult},
        ]


def _resolve_weights(name: str | None) -> MobileNet_V2_Weights | None:
    if not name:
        return None
    try:
        weights = MobileNet_V2_Weights[name]
        weights.get_state_dict(progress=False)  # fail here, loudly, if the download is impossible
    except Exception as e:  # noqa: BLE001 - re-raised with context
        raise PretrainedWeightsError(
            f"Could not load pretrained MobileNetV2 weights '{name}': {e}\n"
            "Connect to the internet once so torchvision can download them "
            "(cached under ~/.cache/torch/hub/checkpoints), or set `model.weights: null` in "
            "config.yaml to train from random initialisation on purpose."
        ) from e
    return weights


def build_model(model_cfg: dict[str, Any], pretrained: bool = True) -> LandmarkRegressor:
    return LandmarkRegressor(
        num_landmarks=NUM_LANDMARKS,
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        weights=model_cfg.get("weights") if pretrained else None,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_inference_checkpoint(path: Path, model: LandmarkRegressor, cfg: dict[str, Any],
                              experiment: str, extra: dict[str, Any] | None = None) -> None:
    """Weights + everything inference needs to reproduce the training preprocessing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "experiment": experiment,
        "model_state": model.state_dict(),
        "model_cfg": cfg["model"],
        "preprocess": {
            "input_size": int(cfg["data"]["input_size"]),
            "box_scale": float(cfg["data"]["box_scale"]),
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        **(extra or {}),
    }, path)


def load_inference_checkpoint(path: str | Path, device: torch.device
                              ) -> tuple[LandmarkRegressor, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT or "model_state" not in ckpt:
        raise ValueError(f"{path} is not an APAC-Face inference checkpoint")
    model = build_model(ckpt["model_cfg"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state"}
    return model, meta
