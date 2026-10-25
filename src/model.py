"""Landmark models: 224x224x3 image -> 98 x (x, y) coordinates in [0, 1].

LandmarkRegressor (MobileNetV2 + global pooling + MLP) and HeatmapLandmarkRegressor
(MobileNetV2 + multi-scale decoder + heatmaps + soft-argmax) share one interface.
"""
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


def _conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class HeatmapLandmarkRegressor(nn.Module):
    """MobileNetV2 + multi-scale decoder + per-landmark heatmaps + soft-argmax (integral regression).

    Global average pooling throws away *where* features are; this head keeps a spatial map. Features
    from strides 4, 8, 16 and 32 are fused top-down (FPN-style) into a 56x56 map (stride 4), one
    heatmap per landmark is predicted, and a spatial softmax turns each heatmap into a probability
    map whose expectation is the landmark coordinate. Coordinates stay differentiable, so it trains
    with the same coordinate loss as the regressor. The softmax peak doubles as a per-landmark
    confidence (high = sharp, unambiguous evidence).
    """

    # MobileNetV2 feature stages and their output channels for a 224x224 input
    STAGES = ((0, 4, 24), (4, 7, 32), (7, 14, 96), (14, 18, 320))   # strides 4, 8, 16, 32

    def __init__(self, num_landmarks: int = NUM_LANDMARKS, decoder_dim: int = 96,
                 weights: str | None = "IMAGENET1K_V2"):
        super().__init__()
        self.num_landmarks = num_landmarks
        feats = mobilenet_v2(weights=_resolve_weights(weights)).features
        self.stages = nn.ModuleList(nn.Sequential(*feats[a:b]) for a, b, _ in self.STAGES)
        self.lateral = nn.ModuleList(nn.Conv2d(c, decoder_dim, 1) for _, _, c in self.STAGES)
        self.smooth = nn.ModuleList(_conv_bn(decoder_dim, decoder_dim) for _ in self.STAGES[:-1])
        self.head = nn.Sequential(_conv_bn(decoder_dim, decoder_dim), nn.Conv2d(decoder_dim, num_landmarks, 1))

    def features_map(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        y = self.lateral[-1](outs[-1])
        for i in range(len(outs) - 2, -1, -1):  # top-down: 7 -> 14 -> 28 -> 56
            y = nn.functional.interpolate(y, size=outs[i].shape[-2:], mode="bilinear", align_corners=False)
            y = self.smooth[i](y + self.lateral[i](outs[i]))
        return y

    def forward_with_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(self.features_map(x)).float()      # softmax in fp32 even under AMP
        n, k, h, w = logits.shape
        prob = torch.softmax(logits.view(n, k, h * w), dim=-1).view(n, k, h, w)
        # cell centres in normalised input coordinates (pixel-centre convention, as the targets)
        size = x.shape[-1]
        gx = ((torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) * (size / w) - 0.5) / size
        gy = ((torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) * (size / h) - 0.5) / size
        coords = torch.stack([(prob.sum(2) * gx).sum(-1), (prob.sum(3) * gy).sum(-1)], dim=-1)
        conf = prob.view(n, k, h * w).amax(-1)
        return coords, conf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_confidence(x)[0]

    def init_mean_shape(self, mean_shape: np.ndarray) -> None:
        """No-op: heatmaps start uniform; kept for interface parity with LandmarkRegressor."""

    def parameter_groups(self, lr: float, head_lr_mult: float) -> list[dict[str, Any]]:
        new = list(self.lateral.parameters()) + list(self.smooth.parameters()) + list(self.head.parameters())
        return [{"params": self.stages.parameters(), "lr": lr}, {"params": new, "lr": lr * head_lr_mult}]


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


def build_model(model_cfg: dict[str, Any], pretrained: bool = True) -> nn.Module:
    """`model_cfg["head"]`: "regression" (global pooling + MLP, default) or "heatmap"."""
    weights = model_cfg.get("weights") if pretrained else None
    if model_cfg.get("head", "regression") == "heatmap":
        return HeatmapLandmarkRegressor(NUM_LANDMARKS, int(model_cfg.get("decoder_dim", 96)), weights)
    return LandmarkRegressor(
        num_landmarks=NUM_LANDMARKS,
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        weights=model_cfg.get("weights") if pretrained else None,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_config(cfg: dict[str, Any], experiment: str) -> dict[str, Any]:
    """Global model settings overridden by the experiment's own `model:` block."""
    return {**cfg["model"], **cfg["experiments"].get(experiment, {}).get("model", {})}


def save_inference_checkpoint(path: Path, model: nn.Module, cfg: dict[str, Any],
                              experiment: str, extra: dict[str, Any] | None = None) -> None:
    """Weights + everything inference needs to reproduce the training preprocessing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "experiment": experiment,
        "model_state": model.state_dict(),
        "model_cfg": model_config(cfg, experiment) if experiment in cfg.get("experiments", {}) else cfg["model"],
        "preprocess": {
            "input_size": int(cfg["data"]["input_size"]),
            "box_scale": float(cfg["data"]["box_scale"]),
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        **(extra or {}),
    }, path)


def load_inference_checkpoint(path: str | Path, device: torch.device
                              ) -> tuple[nn.Module, dict[str, Any]]:
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
