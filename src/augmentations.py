"""Landmark-consistent cropping and data augmentation.

Every geometric change (crop, scale, rotation, translation, flip) is folded into ONE 2x3 affine
matrix that is applied to the image with cv2.warpAffine *and* to the landmark coordinates, so the
image and its labels cannot drift apart. A horizontal flip additionally remaps left/right landmark
indices (landmarks.FLIP_PERMUTATION). Photometric changes never move pixels, so they leave the
landmarks untouched.

Coordinates use the OpenCV pixel-centre convention: pixel (0, 0) covers [-0.5, 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import cv2
import numpy as np
import torch

from .landmarks import FLIP_PERMUTATION

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

Box = tuple[float, float, float]  # (centre x, centre y, side length) of a square region


def crop_matrix(box: Box, output_size: int, angle_deg: float = 0.0, flip: bool = False) -> np.ndarray:
    """Affine matrix mapping the square `box` (optionally rotated) onto an output_size^2 image."""
    cx, cy, side = box
    s = output_size / side
    a = np.deg2rad(angle_deg)
    c, n = np.cos(a) * s, np.sin(a) * s
    out_c = (output_size - 1) / 2.0
    m = np.array(
        [[c, n, out_c - (c * cx + n * cy)],
         [-n, c, out_c - (-n * cx + c * cy)]],
        dtype=np.float64,
    )
    if flip:
        m[0] = -m[0]
        m[0, 2] += output_size - 1
    return m


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return (pts @ matrix[:, :2].T + matrix[:, 2]).astype(np.float32)


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(matrix.astype(np.float64))


def warp(image: np.ndarray, matrix: np.ndarray, output_size: int) -> np.ndarray:
    return cv2.warpAffine(image, matrix, (output_size, output_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB -> float CxHxW tensor normalised with ImageNet statistics."""
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


@dataclass
class AugmentationParams:
    flip_prob: float = 0.5
    rotation_deg: float = 20.0
    scale_range: float = 0.10
    translate: float = 0.05
    brightness: float = 0.25
    contrast: float = 0.25
    blur_prob: float = 0.15
    grayscale_prob: float = 0.10
    occlusion_prob: float = 0.20

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "AugmentationParams":
        cfg = cfg or {}
        known = {f.name for f in fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown augmentation options: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in cfg.items()})


class LandmarkAugmenter:
    """Crops a face region to the network input size, optionally with random augmentation."""

    def __init__(self, params: AugmentationParams, output_size: int, enabled: bool = True):
        self.p = params
        self.size = output_size
        self.enabled = enabled

    def sample_geometry(self, box: Box, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        if not self.enabled:
            return crop_matrix(box, self.size), False
        cx, cy, side = box
        side *= rng.uniform(1.0 - self.p.scale_range, 1.0 + self.p.scale_range)
        tx, ty = rng.uniform(-self.p.translate, self.p.translate, size=2) * side
        angle = rng.uniform(-self.p.rotation_deg, self.p.rotation_deg)
        flip = bool(rng.random() < self.p.flip_prob)
        return crop_matrix((cx + tx, cy + ty, side), self.size, angle, flip), flip

    def __call__(self, image: np.ndarray, landmarks: np.ndarray, box: Box,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        matrix, flipped = self.sample_geometry(box, rng)
        out_img = warp(image, matrix, self.size)
        out_lm = apply_affine(landmarks, matrix)
        if flipped:
            out_lm = out_lm[FLIP_PERMUTATION]
        if self.enabled:
            out_img = self.photometric(out_img, rng)
        return out_img, out_lm

    def photometric(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        img = image.astype(np.float32)
        alpha = 1.0 + rng.uniform(-self.p.contrast, self.p.contrast)
        beta = rng.uniform(-self.p.brightness, self.p.brightness) * 255.0
        mean = img.mean()
        img = np.clip((img - mean) * alpha + mean + beta, 0, 255).astype(np.uint8)

        if rng.random() < self.p.grayscale_prob:
            img = np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
        if rng.random() < self.p.blur_prob:
            k = int(rng.choice([3, 5, 7]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if rng.random() < self.p.occlusion_prob:
            img = self._occlude(img, rng)
        return img

    def _occlude(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        ow = int(w * rng.uniform(0.15, 0.40))
        oh = int(h * rng.uniform(0.15, 0.40))
        x0 = int(rng.integers(0, w - ow + 1))
        y0 = int(rng.integers(0, h - oh + 1))
        out = img.copy()
        if rng.random() < 0.5:
            out[y0:y0 + oh, x0:x0 + ow] = rng.integers(0, 256, size=3, dtype=np.uint8)
        else:
            out[y0:y0 + oh, x0:x0 + ow] = rng.integers(0, 256, size=(oh, ow, 3), dtype=np.uint8)
        return out
