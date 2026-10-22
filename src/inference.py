"""Inference pipeline: face detection -> square crop -> landmark regression -> refinement.

The network was trained on crops centred on the landmark extent (box_scale x longest side).
A face detector box has a different scale and offset, so:
  1. the detector box is mapped to an approximate training crop with a calibration learned on
     WFLW (`python -m src.inference --calibrate-detector`), then
  2. after each prediction the crop is recomputed from the predicted landmarks exactly as in
     training, and the face is predicted again (`refine_iterations` times).

    python -m src.inference --image face.jpg --model proposed --out results/predictions/face.jpg
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .augmentations import Box, apply_affine, crop_matrix, invert_affine, to_tensor, warp
from .landmarks import draw_landmarks, landmark_box
from .model import load_inference_checkpoint
from .utils import get_device, get_logger, load_config, load_json, resolve_path, save_json

log = get_logger("inference")

CALIBRATION_FILE = "detector_calibration.json"
DEFAULT_CALIBRATION = {"side_ratio": 1.25, "dx": 0.0, "dy": 0.08, "source": "default (uncalibrated)"}


# --------------------------------------------------------------------------- image I/O


def decode_image(data: bytes) -> np.ndarray:
    """Encoded JPEG/PNG bytes -> RGB uint8 array (EXIF orientation applied)."""
    if not data:
        raise ValueError("Empty file")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File could not be decoded as an image")
    if min(image.shape[:2]) < 32:
        raise ValueError(f"Image too small ({image.shape[1]}x{image.shape[0]}); need at least 32 px")
    return np.ascontiguousarray(image[:, :, ::-1])


def load_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def encode_png(image_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_rgb[:, :, ::-1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- face detector


class FaceDetector:
    """OpenCV Haar cascade (shipped with opencv-python 4.x) plus a learned box calibration."""

    def __init__(self, calibration: dict[str, Any] | None = None, max_side: int = 800):
        self.calibration = calibration or dict(DEFAULT_CALIBRATION)
        self.max_side = max_side
        self.cascade = None
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if data_dir:
            path = Path(data_dir) / "haarcascade_frontalface_default.xml"
            if path.is_file():
                cascade = cv2.CascadeClassifier(str(path))
                if not cascade.empty():
                    self.cascade = cascade
        if self.cascade is None:
            log.warning("Haar cascade not found (opencv-python>=5 no longer ships it; install "
                        "opencv-python<5). Uploaded images will be treated as face crops.")

    @property
    def available(self) -> bool:
        return self.cascade is not None

    def detect(self, image_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Face rectangles (x, y, w, h) in image pixels, largest first."""
        if self.cascade is None:
            return []
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        scale = min(1.0, self.max_side / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        min_size = max(20, int(min(gray.shape) * 0.08))
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(min_size, min_size))
        rects = [tuple(int(round(v / scale)) for v in f) for f in faces]
        return sorted(rects, key=lambda r: r[2] * r[3], reverse=True)

    def to_crop_box(self, rect: tuple[int, int, int, int]) -> Box:
        x, y, w, h = rect
        c = self.calibration
        return (x + w / 2 + c["dx"] * w, y + h / 2 + c["dy"] * h, w * c["side_ratio"])


def load_calibration(cfg: dict[str, Any]) -> dict[str, Any] | None:
    path = resolve_path(cfg["paths"]["checkpoints"]) / CALIBRATION_FILE
    return load_json(path) if path.is_file() else None


def calibrate_detector(cfg: dict[str, Any], samples: int = 800) -> dict[str, Any]:
    """Fit detector-box -> training-crop mapping on WFLW training faces (median ratios)."""
    from .dataset import parse_annotation_file, wflw_paths

    paths = wflw_paths(cfg["paths"]["wflw_root"])
    faces = parse_annotation_file(paths.train_list)
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(faces)
    detector = FaceDetector()
    if not detector.available:
        raise RuntimeError("Haar cascade unavailable; install opencv-python<5")
    box_scale = float(cfg["data"]["box_scale"])
    ratios, dxs, dys, tried = [], [], [], 0
    for s in faces[:samples]:
        image = cv2.imread(str(paths.images / s.image_name))
        if image is None:
            continue
        tried += 1
        cx, cy, side = landmark_box(s.landmarks, box_scale)
        best, best_iou = None, 0.0
        for r in detector.detect(image[:, :, ::-1]):
            iou = _iou((cx - side / 2, cy - side / 2, side, side), r)
            if iou > best_iou:
                best, best_iou = r, iou
        if best is None or best_iou < 0.35:
            continue
        x, y, w, h = best
        ratios.append(side / w)
        dxs.append((cx - (x + w / 2)) / w)
        dys.append((cy - (y + h / 2)) / h)
    if len(ratios) < 30:
        raise RuntimeError(f"Only {len(ratios)} detector matches - calibration unreliable")
    calib = {
        "side_ratio": float(np.median(ratios)), "dx": float(np.median(dxs)), "dy": float(np.median(dys)),
        "matched": len(ratios), "tried": tried, "detection_rate": len(ratios) / max(tried, 1),
        "source": "WFLW train, Haar frontalface_default, median of matched faces",
    }
    save_json(calib, resolve_path(cfg["paths"]["checkpoints"]) / CALIBRATION_FILE)
    return calib


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


# --------------------------------------------------------------------------- predictor


@dataclass
class Prediction:
    landmarks: np.ndarray            # (98, 2) original-image pixels
    box: Box                         # final crop box (cx, cy, side)
    face_detected: bool
    faces_found: int
    model_ms: float                  # network forward time (all passes)
    total_ms: float                  # detection + crops + network
    model: str
    image_size: tuple[int, int]      # (width, height)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["landmarks"] = np.round(self.landmarks, 2).tolist()
        d["box"] = [round(float(v), 2) for v in self.box]
        d["num_landmarks"] = int(len(self.landmarks))
        return d


class LandmarkPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto",
                 detector: FaceDetector | None = None, refine_iterations: int = 1):
        self.device = get_device(device)
        self.model, self.meta = load_inference_checkpoint(checkpoint, self.device)
        self.name = self.meta.get("experiment", Path(checkpoint).stem)
        self.size = int(self.meta["preprocess"]["input_size"])
        self.box_scale = float(self.meta["preprocess"]["box_scale"])
        self.detector = detector or FaceDetector()
        self.refine_iterations = refine_iterations
        self.warmup()

    @torch.no_grad()
    def warmup(self) -> None:
        """One dummy forward pass so CUDA kernel set-up is not billed to the first real request."""
        self.model(torch.zeros(1, 3, self.size, self.size, device=self.device))
        self._sync()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def predict_boxes(self, image_rgb: np.ndarray, boxes: list[Box]) -> np.ndarray:
        """Landmarks (K, 98, 2) in image pixels for K crop boxes of the same image (one batch)."""
        mats = [crop_matrix(b, self.size) for b in boxes]
        batch = torch.stack([to_tensor(warp(image_rgb, m, self.size)) for m in mats]).to(self.device)
        out = self.model(batch).float().cpu().numpy() * self.size
        return np.stack([apply_affine(o, invert_affine(m)) for o, m in zip(out, mats)])

    def initial_box(self, image_rgb: np.ndarray) -> tuple[Box, bool, int, list[str]]:
        h, w = image_rgb.shape[:2]
        faces = self.detector.detect(image_rgb)
        if faces:
            notes = [f"{len(faces)} faces found; showing the largest."] if len(faces) > 1 else []
            return self.detector.to_crop_box(faces[0]), True, len(faces), notes
        note = ("No face detected - treating the image as a centred face crop. "
                "Results may be poor if the face is small or off-centre.")
        return ((w - 1) / 2, (h - 1) / 2, min(w, h) * 0.9), False, 0, [note]

    def predict(self, image_rgb: np.ndarray, box: Box | None = None,
                refine_iterations: int | None = None) -> Prediction:
        t0 = time.perf_counter()
        warnings: list[str] = []
        detected, n_faces = box is not None, int(box is not None)
        if box is None:
            box, detected, n_faces, warnings = self.initial_box(image_rgb)
        iters = self.refine_iterations if refine_iterations is None else refine_iterations
        model_ms = 0.0
        for i in range(iters + 1):
            if i > 0:
                box = landmark_box(lm, self.box_scale)
            self._sync()
            t = time.perf_counter()
            lm = self.predict_boxes(image_rgb, [box])[0]
            self._sync()
            model_ms += (time.perf_counter() - t) * 1000
        if not np.isfinite(lm).all():
            warnings.append("Model produced non-finite coordinates.")
        h, w = image_rgb.shape[:2]
        return Prediction(lm, box, detected, n_faces, model_ms, (time.perf_counter() - t0) * 1000,
                          self.name, (w, h), warnings)


def annotate(image_rgb: np.ndarray, prediction: Prediction, draw_mesh: bool = True,
             draw_box: bool = False) -> np.ndarray:
    out = draw_landmarks(image_rgb, prediction.landmarks, draw_mesh=draw_mesh)
    if draw_box:
        cx, cy, side = prediction.box
        p0 = (int(cx - side / 2), int(cy - side / 2))
        p1 = (int(cx + side / 2), int(cy + side / 2))
        cv2.rectangle(out, p0, p1, (99, 102, 241), max(1, int(side / 150)), cv2.LINE_AA)
    return out


def available_checkpoints(cfg: dict[str, Any]) -> dict[str, Path]:
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"])
    return {name: ckpt_dir / f"{name}.pth" for name in cfg["experiments"]
            if (ckpt_dir / f"{name}.pth").is_file()}


def load_predictors(cfg: dict[str, Any], device: str = "auto") -> dict[str, LandmarkPredictor]:
    detector = FaceDetector(load_calibration(cfg))
    refine = int(cfg["inference"]["refine_iterations"])
    return {name: LandmarkPredictor(path, device, detector, refine)
            for name, path in available_checkpoints(cfg).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APAC-Face on an image")
    parser.add_argument("--config", default=None)
    parser.add_argument("--image", help="input JPG/PNG")
    parser.add_argument("--model", default=None, help="experiment name (default from config)")
    parser.add_argument("--out", default=None, help="where to save the annotated image")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--calibrate-detector", action="store_true",
                        help="fit the face-detector box calibration on WFLW train images")
    parser.add_argument("--samples", type=int, default=800)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.calibrate_detector:
        calib = calibrate_detector(cfg, args.samples)
        log.info("Calibration: %s", calib)
        return
    if not args.image:
        parser.error("--image is required (or use --calibrate-detector)")
    name = args.model or cfg["inference"]["default_model"]
    ckpt = resolve_path(cfg["paths"]["checkpoints"]) / f"{name}.pth"
    if not ckpt.is_file():
        raise SystemExit(f"No checkpoint for '{name}' at {ckpt}. Train it first: "
                         f"python -m src.train --experiment {name}")
    predictor = LandmarkPredictor(ckpt, args.device, FaceDetector(load_calibration(cfg)),
                                  int(cfg["inference"]["refine_iterations"]))
    image = load_image(args.image)
    pred = predictor.predict(image)
    for w in pred.warnings:
        log.warning(w)
    log.info("%s: %d landmarks, model %.1f ms, total %.1f ms", pred.model, len(pred.landmarks),
             pred.model_ms, pred.total_ms)
    out = Path(args.out) if args.out else resolve_path(cfg["paths"]["results"]) / "predictions" / (
        Path(args.image).stem + f"_{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_png(annotate(image, pred, draw_box=True)))
    log.info("Annotated image saved to %s", out)


if __name__ == "__main__":
    main()
