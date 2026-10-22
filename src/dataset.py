"""WFLW annotation parsing, face-crop caching and the PyTorch dataset.

WFLW annotation line (one face per line, 207 whitespace-separated fields):
    x0 y0 ... x97 y97  x_min y_min x_max y_max  pose expression illumination make-up occlusion blur  image_name

Run `python -m src.dataset` once to parse the annotations, split train/val, write pre-cropped faces
to the cache directory and save a visual sanity check to results/dataset_samples.png.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import LandmarkAugmenter, apply_affine, crop_matrix, to_tensor, warp
from .landmarks import NUM_LANDMARKS, draw_landmarks, landmark_box
from .utils import get_logger, load_config, load_json, resolve_path, save_json

log = get_logger(__name__)

ATTRIBUTES = ("pose", "expression", "illumination", "makeup", "occlusion", "blur")
ATTRIBUTE_LABELS = {
    "pose": "Large pose",
    "expression": "Exaggerated expression",
    "illumination": "Extreme illumination",
    "makeup": "Make-up",
    "occlusion": "Occlusion",
    "blur": "Blur",
}
SUBSET_FILES = {
    "pose": "list_98pt_test_largepose.txt",
    "expression": "list_98pt_test_expression.txt",
    "illumination": "list_98pt_test_illumination.txt",
    "makeup": "list_98pt_test_makeup.txt",
    "occlusion": "list_98pt_test_occlusion.txt",
    "blur": "list_98pt_test_blur.txt",
}
FIELDS_PER_LINE = NUM_LANDMARKS * 2 + 4 + len(ATTRIBUTES) + 1
CACHE_VERSION = 1


class DatasetNotFoundError(FileNotFoundError):
    """Raised with setup instructions when the WFLW files are not where the config says."""


def setup_instructions(root: Path) -> str:
    return (
        f"WFLW dataset not found under: {root}\n\n"
        "Expected layout:\n"
        "  data/WFLW/WFLW_images/<event folders>/*.jpg\n"
        "  data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt\n"
        "  data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt\n"
        "  data/WFLW/WFLW_annotations/list_98pt_test/list_98pt_test_*.txt\n\n"
        "Get the files from https://github.com/wywu/LAB (README -> WFLW Dataset Download):\n"
        "  1. WFLW_images.tar.gz (Google Drive / Baidu link in the README)\n"
        "  2. WFLW_annotations.tar.gz\n"
        "Then extract both into data/WFLW/, e.g.\n"
        "  tar -xzf WFLW_images.tar.gz -C data/WFLW\n"
        "  tar -xzf WFLW_annotations.tar.gz -C data/WFLW\n"
    )


@dataclass
class WFLWPaths:
    root: Path
    images: Path
    train_list: Path
    test_list: Path
    subset_dir: Path


def wflw_paths(root: str | Path) -> WFLWPaths:
    root = resolve_path(root)
    ann = root / "WFLW_annotations"
    paths = WFLWPaths(
        root=root,
        images=root / "WFLW_images",
        train_list=ann / "list_98pt_rect_attr_train_test" / "list_98pt_rect_attr_train.txt",
        test_list=ann / "list_98pt_rect_attr_train_test" / "list_98pt_rect_attr_test.txt",
        subset_dir=ann / "list_98pt_test",
    )
    missing = [p for p in (paths.images, paths.train_list, paths.test_list) if not p.exists()]
    if missing:
        raise DatasetNotFoundError(
            setup_instructions(root) + "\nMissing: " + ", ".join(str(m) for m in missing))
    return paths


@dataclass
class WFLWSample:
    image_name: str          # path relative to WFLW_images/, forward slashes
    landmarks: np.ndarray    # (98, 2) float32, original image pixels
    rect: np.ndarray         # (4,) detection rectangle x_min, y_min, x_max, y_max
    attributes: np.ndarray   # (6,) int8 in ATTRIBUTES order


def parse_annotation_line(line: str) -> WFLWSample:
    parts = line.split()
    if len(parts) != FIELDS_PER_LINE:
        raise ValueError(f"expected {FIELDS_PER_LINE} fields, found {len(parts)}")
    nums = np.asarray(parts[:-1], dtype=np.float64)
    landmarks = nums[:NUM_LANDMARKS * 2].reshape(NUM_LANDMARKS, 2).astype(np.float32)
    rect = nums[NUM_LANDMARKS * 2:NUM_LANDMARKS * 2 + 4].astype(np.float32)
    attrs = nums[NUM_LANDMARKS * 2 + 4:]
    if not np.isfinite(landmarks).all():
        raise ValueError("non-finite landmark coordinate")
    if not np.isin(attrs, (0, 1)).all():
        raise ValueError(f"attribute flags must be 0/1, got {attrs.tolist()}")
    return WFLWSample(parts[-1].replace("\\", "/"), landmarks, rect, attrs.astype(np.int8))


def parse_annotation_file(path: Path) -> list[WFLWSample]:
    samples, bad = [], 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                samples.append(parse_annotation_line(line))
            except ValueError as e:
                bad += 1
                if bad <= 5:
                    log.warning("%s:%d skipped (%s)", path.name, lineno, e)
    if bad:
        log.warning("%s: %d malformed line(s) skipped", path.name, bad)
    if not samples:
        raise ValueError(f"No valid annotations parsed from {path}")
    return samples


def split_train_val(samples: list[WFLWSample], val_fraction: float, seed: int
                    ) -> tuple[list[WFLWSample], list[WFLWSample]]:
    """Split by source image so that faces from one photo never land in both splits."""
    names = sorted({s.image_name for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_val = int(round(len(names) * val_fraction))
    val_names = set(names[:n_val])
    train = [s for s in samples if s.image_name not in val_names]
    val = [s for s in samples if s.image_name in val_names]
    return train, val


# --------------------------------------------------------------------------- cache


def _cache_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    d = cfg["data"]
    return {
        "version": CACHE_VERSION,
        "input_size": int(d["input_size"]),
        "box_scale": float(d["box_scale"]),
        "cache_expand": float(d["cache_expand"]),
        "cache_size": int(round(d["input_size"] * d["cache_expand"])),
        "val_fraction": float(d["val_fraction"]),
        "seed": int(cfg["seed"]),
    }


def cache_is_valid(cfg: dict[str, Any]) -> bool:
    cache_dir = resolve_path(cfg["paths"]["cache_dir"])
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        return False
    meta = load_json(meta_path)
    if meta.get("settings") != _cache_settings(cfg):
        return False
    return all((cache_dir / f"{s}.npz").is_file() for s in ("train", "val", "test"))


def _crop_face(image: np.ndarray, sample: WFLWSample, settings: dict[str, Any]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, side = landmark_box(sample.landmarks, settings["box_scale"])
    matrix = crop_matrix((cx, cy, side * settings["cache_expand"]), settings["cache_size"])
    crop = warp(image, matrix, settings["cache_size"])
    return crop, apply_affine(sample.landmarks, matrix), matrix.astype(np.float32)


def _write_split(name: str, samples: list[WFLWSample], paths: WFLWPaths, cache_dir: Path,
                 settings: dict[str, Any], workers: int = 8) -> dict[str, int]:
    out_dir = cache_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_image[s.image_name].append(i)

    lm_cache = np.zeros((len(samples), NUM_LANDMARKS, 2), np.float32)
    matrices = np.zeros((len(samples), 2, 3), np.float32)
    ok = np.zeros(len(samples), bool)

    def process(item: tuple[str, list[int]]) -> None:
        image_name, idxs = item
        image = cv2.imread(str(paths.images / image_name), cv2.IMREAD_COLOR)
        if image is None:
            return
        for i in idxs:
            crop, lm, m = _crop_face(image, samples[i], settings)
            if cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                lm_cache[i], matrices[i], ok[i] = lm, m, True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(process, by_image.items()))

    missing = int((~ok).sum())
    if missing:
        log.warning("%s: %d face(s) skipped because their image could not be read", name, missing)
    keep = np.flatnonzero(ok)
    np.savez_compressed(
        cache_dir / f"{name}.npz",
        files=np.array([f"{name}/{i:05d}.jpg" for i in keep]),
        landmarks=lm_cache[keep],
        landmarks_orig=np.stack([samples[i].landmarks for i in keep]),
        matrices=matrices[keep],
        attributes=np.stack([samples[i].attributes for i in keep]),
        rects=np.stack([samples[i].rect for i in keep]),
        image_names=np.array([samples[i].image_name for i in keep]),
    )
    return {"faces": int(len(keep)), "skipped": missing,
            "images": len({samples[i].image_name for i in keep})}


def _verify_test_subsets(paths: WFLWPaths, test: list[WFLWSample]) -> dict[str, Any]:
    """Check the attribute flags against the official per-subset test lists."""
    report: dict[str, Any] = {}
    for j, attr in enumerate(ATTRIBUTES):
        from_flags = int(sum(int(s.attributes[j]) for s in test))
        subset_file = paths.subset_dir / SUBSET_FILES[attr]
        official = None
        if subset_file.is_file():
            with subset_file.open(encoding="utf-8") as f:
                official = sum(1 for line in f if line.strip())
        report[attr] = {"from_attribute_flags": from_flags, "official_list": official,
                        "match": official is None or official == from_flags}
    return report


def build_cache(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    cache_dir = resolve_path(cfg["paths"]["cache_dir"])
    if not force and cache_is_valid(cfg):
        return load_json(cache_dir / "meta.json")

    settings = _cache_settings(cfg)
    paths = wflw_paths(cfg["paths"]["wflw_root"])
    log.info("Parsing WFLW annotations from %s", paths.root)
    train_all = parse_annotation_file(paths.train_list)
    test = parse_annotation_file(paths.test_list)
    train, val = split_train_val(train_all, settings["val_fraction"], settings["seed"])
    log.info("Faces: train=%d val=%d test=%d", len(train), len(val), len(test))

    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        log.info("Caching %s crops (%d faces) ...", name, len(split))
        counts[name] = _write_split(name, split, paths, cache_dir, settings)

    prevalence = {
        name: {a: float(np.mean([s.attributes[j] for s in split])) for j, a in enumerate(ATTRIBUTES)}
        for name, split in (("train", train), ("val", val), ("test", test))
    }
    meta = {
        "settings": settings,
        "counts": counts,
        "attribute_prevalence": prevalence,
        "test_subset_check": _verify_test_subsets(paths, test),
        "annotation_sha1": {
            p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in (paths.train_list, paths.test_list)
        },
    }
    save_json(meta, cache_dir / "meta.json")
    return meta


# --------------------------------------------------------------------------- torch dataset


def load_split(cfg: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    cache_dir = resolve_path(cfg["paths"]["cache_dir"])
    path = cache_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Cached split {path} missing - run `python -m src.dataset` first")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def canonical_cache_box(cfg: dict[str, Any]) -> tuple[float, float, float]:
    """The training crop expressed in cached-crop pixel coordinates."""
    s = _cache_settings(cfg)
    c = (s["cache_size"] - 1) / 2.0
    return c, c, s["cache_size"] / s["cache_expand"]


DETECTOR_BOXES_FILE = "detector_boxes.npz"


def load_detector_boxes(cfg: dict[str, Any], split: str) -> np.ndarray:
    """MediaPipe-style crop boxes (N, 3) in cached-crop pixels; NaN where MediaPipe found no face."""
    path = resolve_path(cfg["paths"]["cache_dir"]) / DETECTOR_BOXES_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing - run python -m src.mediapipe_baseline --export-boxes")
    with np.load(path) as data:
        return data[split]


class WFLWCropDataset(Dataset):
    """Serves (image tensor, normalised landmarks in [0, 1], index) from the face-crop cache.

    `detector_boxes` (N, 3) replaces the canonical crop box with a detector-style box for a share
    `detector_prob` of the samples (1.0 = always, e.g. for detector-seeded validation). Augmentation
    jitter is applied on top of whichever box is used.
    """

    def __init__(self, cfg: dict[str, Any], split: str, augmenter: LandmarkAugmenter,
                 limit: int | None = None, detector_boxes: np.ndarray | None = None,
                 detector_prob: float = 0.0):
        data = load_split(cfg, split)
        n = len(data["files"]) if limit is None else min(limit, len(data["files"]))
        self.cache_dir = resolve_path(cfg["paths"]["cache_dir"])
        self.files = data["files"][:n]
        self.landmarks = data["landmarks"][:n]
        self.attributes = data["attributes"][:n]
        self.augmenter = augmenter
        self.box = canonical_cache_box(cfg)
        self.detector_boxes = None if detector_boxes is None else np.asarray(detector_boxes[:n], np.float32)
        self.detector_prob = float(detector_prob)
        self._rng: np.random.Generator | None = None

    def crop_box(self, idx: int, rng: np.random.Generator) -> tuple[float, float, float]:
        if self.detector_boxes is None or self.detector_prob <= 0:
            return self.box
        b = self.detector_boxes[idx]
        if not np.isfinite(b).all():
            return self.box
        if self.detector_prob >= 1.0 or rng.random() < self.detector_prob:
            return float(b[0]), float(b[1]), float(b[2])
        return self.box

    def __len__(self) -> int:
        return len(self.files)

    def _generator(self) -> np.random.Generator:
        # Created lazily inside each DataLoader worker so every worker gets its own stream,
        # and the stream keeps advancing across epochs (fresh augmentations every epoch).
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**63)
        return self._rng

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        bgr = cv2.imread(str(self.cache_dir / self.files[idx]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cached crop missing: {self.files[idx]} (rebuild the cache)")
        rng = self._generator()
        image, lm = self.augmenter(bgr[:, :, ::-1], self.landmarks[idx], self.crop_box(idx, rng), rng)
        target = torch.from_numpy(lm / float(self.augmenter.size)).float()
        return to_tensor(image), target, idx


def balanced_sample_weights(attributes: np.ndarray, columns: list[int], power: float) -> np.ndarray:
    """Per-sample weights that up-weight rare challenge conditions.

    For each attribute the weight of a sample is 0.5 / P(its value), i.e. both values of a binary
    attribute would receive equal total mass. Weights are averaged over attributes, tempered by
    `power` (0 = uniform, 1 = full inverse frequency) and normalised to mean 1.
    """
    attributes = np.asarray(attributes)
    w = np.zeros(len(attributes), np.float64)
    used = 0
    for j in columns:
        values, counts = np.unique(attributes[:, j], return_counts=True)
        if len(values) < 2:
            continue
        freq = dict(zip(values.tolist(), (counts / counts.sum()).tolist()))
        w += np.array([0.5 / freq[v] for v in attributes[:, j].tolist()])
        used += 1
    if used == 0:
        return np.ones(len(attributes))
    w = (w / used) ** power
    return w / w.mean()


# --------------------------------------------------------------------------- CLI


def save_sample_grid(cfg: dict[str, Any], out_path: Path, n: int = 24, cols: int = 6) -> Path:
    data = load_split(cfg, "train")
    cache_dir = resolve_path(cfg["paths"]["cache_dir"])
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(data["files"]), size=min(n, len(data["files"])), replace=False)
    box = canonical_cache_box(cfg)
    size = int(cfg["data"]["input_size"])
    tiles = []
    for i in idx:
        img = cv2.imread(str(cache_dir / data["files"][i]))[:, :, ::-1]
        m = crop_matrix(box, size)
        tiles.append(draw_landmarks(warp(img, m, size), apply_affine(data["landmarks"][i], m)))
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows)[:, :, ::-1])
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse WFLW and build the face-crop cache")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if the cache is valid")
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        meta = build_cache(cfg, force=args.rebuild)
    except DatasetNotFoundError as e:
        print(e)
        raise SystemExit(1)
    for split, c in meta["counts"].items():
        log.info("%-5s faces=%5d images=%5d skipped=%d", split, c["faces"], c["images"], c["skipped"])
    for attr, r in meta["test_subset_check"].items():
        log.info("test subset %-12s flags=%4d official=%s %s", attr, r["from_attribute_flags"],
                 r["official_list"], "OK" if r["match"] else "MISMATCH")
    results = resolve_path(cfg["paths"]["results"])
    save_json(meta, results / "dataset_summary.json")
    log.info("Sample grid written to %s", save_sample_grid(cfg, results / "dataset_samples.png"))


if __name__ == "__main__":
    main()
