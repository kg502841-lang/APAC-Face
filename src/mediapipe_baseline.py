"""Google MediaPipe Face Landmarker as the external baseline, scored on real WFLW ground truth.

MediaPipe predicts a 478-vertex face mesh and our models predict the 98 WFLW points, so both are
reduced to the common 68-point iBUG layout (MediaPipe through the Faceify web app's vertex map,
WFLW through the standard 98->68 subset) and compared with the WFLW manual annotations reduced the
same way. NME is normalised by the outer-eye-corner distance (68-point 36/45 = WFLW 60/72).

Fairness rules:
  * every method sees the same face crop: the cached WFLW test crop (1.8x the landmark extent);
  * the headline comparison uses the faces MediaPipe detected, for every method; misses are
    reported separately (detection rate) and counted as failures in the all-faces numbers;
  * MediaPipe's mesh follows its own landmark convention (most visibly on the jaw line), so a
    second baseline, "MediaPipe (calibrated)", adds the mean per-point offset to ground truth that
    was learned in an eye-aligned frame on WFLW *training* faces. It removes the systematic
    convention gap and leaves MediaPipe's real localisation error.

    python -m src.mediapipe_baseline
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from . import plotting
from .dataset import ATTRIBUTE_LABELS, ATTRIBUTES, DETECTOR_BOXES_FILE, build_cache, load_split
from .evaluate import predict_test_set
from .landmarks import EYES_68, INNER_68, landmark_box, to_68
from .metrics import (auc, failure_rate, paired_bootstrap, per_sample_nme, relative_improvement)
from .model import load_inference_checkpoint
from .utils import get_device, get_logger, load_config, resolve_path, save_json

log = get_logger("mediapipe")

POINT_FAIL = 0.08   # the Faceify app's per-point "clinical tolerance" (normalised error > 8%)


class MediaPipeLandmarker:
    """Thin wrapper around the MediaPipe Tasks FaceLandmarker (IMAGE mode, one face)."""

    def __init__(self, model_path: str | Path, min_confidence: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise ImportError("MediaPipe is not installed: pip install mediapipe --no-deps absl-py "
                              "flatbuffers (see README)") from e
        path = resolve_path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {path}")
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=min_confidence, min_face_presence_confidence=min_confidence)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self.version = mp.__version__

    def detect(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """(478, 2) landmarks in OpenCV pixel coordinates (pixel centres at integers), or None."""
        h, w = image_rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], np.float32)
        return pts - 0.5  # MediaPipe uses continuous coordinates (0 = image edge)

    def close(self) -> None:
        self._landmarker.close()


def _cache_to_original(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    inv = cv2.invertAffineTransform(matrix.astype(np.float64))
    return (points.astype(np.float64) @ inv[:, :2].T + inv[:, 2]).astype(np.float32)


def run_on_split(cfg: dict[str, Any], split: str, landmarker: MediaPipeLandmarker,
                 limit: int | None = None, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MediaPipe on cached face crops -> (indices, 478 points in original pixels, the same points in
    crop pixels, detected mask)."""
    data = load_split(cfg, split)
    cache_dir = resolve_path(cfg["paths"]["cache_dir"])
    idx = np.arange(len(data["files"]))
    if limit is not None and limit < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=limit, replace=False))
    out = np.full((len(idx), 478, 2), np.nan, np.float32)
    in_crop = np.full((len(idx), 478, 2), np.nan, np.float32)
    found = np.zeros(len(idx), bool)
    t0 = time.time()
    for k, i in enumerate(idx):
        crop = cv2.imread(str(cache_dir / data["files"][i]))[:, :, ::-1]
        pts = landmarker.detect(crop)
        if pts is not None:
            in_crop[k] = pts
            out[k] = _cache_to_original(pts, data["matrices"][i])
            found[k] = True
        if (k + 1) % 500 == 0:
            log.info("%s: %d/%d faces (%.0fs)", split, k + 1, len(idx), time.time() - t0)
    return idx, out, in_crop, found


def mediapipe_seed_box(mp478: np.ndarray, calib: dict[str, float]) -> tuple[float, float, float]:
    """Our training-style crop box from MediaPipe's mesh extent (what the web app does)."""
    x0, y0 = mp478.min(axis=0)
    x1, y1 = mp478.max(axis=0)
    size = max(x1 - x0, y1 - y0)
    return ((x0 + x1) / 2 + calib["dx"] * size, (y0 + y1) / 2 + calib["dy"] * size, calib["side_ratio"] * size)


def predict_mediapipe_seeded(cfg: dict[str, Any], predictor: Any, mp_crop: np.ndarray,
                             found: np.ndarray, calib: dict[str, float]) -> np.ndarray:
    """Our model with no ground-truth box: the crop comes from MediaPipe's own face localisation,
    one pass (refinement passes did not help with MediaPipe seeds on WFLW validation)."""
    test = load_split(cfg, "test")
    cache_dir = resolve_path(cfg["paths"]["cache_dir"])
    out = np.full((len(found), 98, 2), np.nan, np.float32)
    for i in np.flatnonzero(found):
        crop = np.ascontiguousarray(cv2.imread(str(cache_dir / test["files"][i]))[:, :, ::-1])
        lm = predictor.predict_boxes(crop, [mediapipe_seed_box(mp_crop[i], calib)])[0]
        out[i] = _cache_to_original(lm, test["matrices"][i])
    return out


def _eye_frame(p68: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origin, rotation and scale of an eye-aligned frame from predicted 68-point eye corners."""
    a, b = p68[..., EYES_68[0], :], p68[..., EYES_68[1], :]
    v = b - a
    scale = np.linalg.norm(v, axis=-1)
    c, s = v[..., 0] / scale, v[..., 1] / scale
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # columns: x-axis, y-axis
    return (a + b) / 2, rot, scale


def fit_offset(mp68: np.ndarray, gt68: np.ndarray) -> np.ndarray:
    """Median per-point offset (gt - mediapipe) in MediaPipe's own eye-aligned, IOD-scaled frame."""
    _, rot, scale = _eye_frame(mp68)
    delta = (gt68 - mp68) / scale[:, None, None]
    local = np.einsum("nji,nkj->nki", rot, delta)  # rotate into the frame (R^T d)
    return np.median(local, axis=0)


def apply_offset(mp68: np.ndarray, offset: np.ndarray) -> np.ndarray:
    _, rot, scale = _eye_frame(mp68)
    return mp68 + np.einsum("nij,kj->nki", rot, offset) * scale[:, None, None]


def fit_box_calibration(mp478: np.ndarray, gt98: np.ndarray, box_scale: float) -> dict[str, float]:
    """Map MediaPipe's mesh extent to our training crop (used by the web app to seed our model)."""
    ratios, dxs, dys = [], [], []
    for mp, gt in zip(mp478, gt98):
        x0, y0 = mp.min(axis=0)
        x1, y1 = mp.max(axis=0)
        size = max(x1 - x0, y1 - y0)
        cx, cy, side = landmark_box(gt, box_scale)
        ratios.append(side / size)
        dxs.append((cx - (x0 + x1) / 2) / size)
        dys.append((cy - (y0 + y1) / 2) / size)
    return {"side_ratio": float(np.median(ratios)), "dx": float(np.median(dxs)),
            "dy": float(np.median(dys)), "faces": len(ratios)}


def export_detector_boxes(cfg: dict[str, Any], landmarker: MediaPipeLandmarker,
                          calib: dict[str, float]) -> Path:
    """MediaPipe-seeded crop boxes for every cached face (in cached-crop pixels).

    Used to train and validate a detector-robust model: the boxes are exactly what the web app
    builds from MediaPipe's detection. Training faces MediaPipe misses get the canonical box
    perturbed by the box error of a randomly drawn detected training face, so every training face
    has a realistic detector-style box. Missed val/test faces are NaN.
    """
    from .dataset import canonical_cache_box

    cx0, cy0, side0 = canonical_cache_box(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        _, _, in_crop, found = run_on_split(cfg, split, landmarker)
        boxes = np.full((len(found), 3), np.nan, np.float32)
        for i in np.flatnonzero(found):
            boxes[i] = mediapipe_seed_box(in_crop[i], calib)
        if split == "train":
            ok = np.isfinite(boxes[:, 0])
            rel = np.column_stack([(boxes[ok, 0] - cx0) / side0, (boxes[ok, 1] - cy0) / side0, boxes[ok, 2] / side0])
            for i in np.flatnonzero(~ok):
                dx, dy, s = rel[rng.integers(len(rel))]
                boxes[i] = (cx0 + dx * side0, cy0 + dy * side0, s * side0)
            arrays["train_detected"] = ok
            arrays["train_box_error"] = rel
        arrays[split] = boxes
        log.info("detector boxes %s: MediaPipe found %d/%d faces", split, int(found.sum()), len(found))
    out = resolve_path(cfg["paths"]["cache_dir"]) / DETECTOR_BOXES_FILE
    np.savez_compressed(out, **arrays)
    return out


def score(pred68: np.ndarray, gt68: np.ndarray, thr: float) -> dict[str, Any]:
    nme = per_sample_nme(pred68, gt68, outer_eyes=EYES_68)
    inner = per_sample_nme(pred68, gt68, outer_eyes=EYES_68, points=INNER_68)
    iod = np.linalg.norm(gt68[:, EYES_68[0]] - gt68[:, EYES_68[1]], axis=-1)
    dist = np.linalg.norm(pred68 - gt68, axis=-1)
    ok = np.isfinite(nme)
    return {
        "faces": int(len(nme)),
        "nme": float(np.nanmean(nme)) if ok.any() else float("nan"),
        "nme_median": float(np.nanmedian(nme)) if ok.any() else float("nan"),
        "nme_inner": float(np.nanmean(inner)) if ok.any() else float("nan"),
        "mean_error_px": float(np.nanmean(dist[ok])) if ok.any() else float("nan"),
        "failure_rate": failure_rate(nme, thr),
        "auc": auc(nme, thr),
        "point_failure_rate": float(np.mean(dist[ok] / iod[ok, None] > POINT_FAIL)) if ok.any() else float("nan"),
        "regions": {name: float(np.nanmean(dist[ok][:, a:b] / iod[ok, None])) for name, a, b in
                    (("Jawline", 0, 17), ("Eyebrows", 17, 27), ("Nose", 27, 36), ("Eyes", 36, 48), ("Mouth", 48, 68))},
        "_nme": nme,
        "_inner": inner,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark APAC-Face against Google MediaPipe on WFLW")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-faces", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--export-boxes", action="store_true",
                        help="only write MediaPipe-style crop boxes for all faces (detector-robust training)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_cache(cfg)
    mcfg = cfg.get("mediapipe", {})
    thr = float(cfg["eval"]["failure_threshold"])
    n_boot = int(cfg["eval"]["bootstrap_samples"])
    landmarker = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"], float(mcfg.get("min_confidence", 0.5)))
    if args.export_boxes:
        from .utils import load_json
        calib_path = resolve_path(cfg["paths"]["checkpoints"]) / "mediapipe_calibration.json"
        if not calib_path.is_file():
            raise SystemExit("Run python -m src.mediapipe_baseline once first (it learns the box calibration)")
        out = export_detector_boxes(cfg, landmarker, load_json(calib_path))
        landmarker.close()
        log.info("Wrote %s", out)
        return

    # 1) Learn MediaPipe's convention offset (and its box->crop mapping) on training faces only.
    tr_idx, tr_mp, _, tr_found = run_on_split(cfg, "train", landmarker,
                                              args.train_faces or int(mcfg.get("train_faces", 2000)), cfg["seed"])
    train = load_split(cfg, "train")
    tr_gt98 = train["landmarks_orig"][tr_idx][tr_found]
    offset = fit_offset(to_68(tr_mp[tr_found], "mediapipe"), to_68(tr_gt98, "wflw"))
    box_cal = fit_box_calibration(tr_mp[tr_found], tr_gt98, float(cfg["data"]["box_scale"]))
    log.info("Train: MediaPipe found %d/%d faces; offset learned", tr_found.sum(), len(tr_found))

    # 2) MediaPipe on every test face.
    _, mp478, mp478_crop, found = run_on_split(cfg, "test", landmarker)
    landmarker.close()
    test = load_split(cfg, "test")
    gt68 = to_68(test["landmarks_orig"], "wflw")
    mp68 = to_68(mp478, "mediapipe")
    mp68_cal = mp68.copy()
    mp68_cal[found] = apply_offset(mp68[found], offset)

    # 3) Our models on the same faces.
    device = get_device(args.device)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"])
    methods: dict[str, tuple[str, np.ndarray]] = {
        "mediapipe": ("Google MediaPipe", mp68),
        "mediapipe_calibrated": ("MediaPipe (calibrated)", mp68_cal),
    }
    from .inference import LandmarkPredictor
    ours: list[str] = []
    for name, exp in cfg["experiments"].items():
        path = ckpt_dir / f"{name}.pth"
        if not path.is_file():
            continue
        label = exp.get("label", name)
        model, _ = load_inference_checkpoint(path, device)
        with torch.no_grad():
            methods[name] = (label, to_68(predict_test_set(model, cfg, device), "wflw"))
        # Same face-location information MediaPipe had: the crop comes from MediaPipe's detection.
        seeded = predict_mediapipe_seeded(cfg, LandmarkPredictor(path, args.device), mp478_crop, found, box_cal)
        methods[f"{name}_mp_located"] = (f"{label} (MediaPipe-located)", to_68(seeded, "wflw"))
        ours += [name, f"{name}_mp_located"]

    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mediapipe_version": landmarker.version,
        "mediapipe_model": Path(cfg["paths"]["mediapipe_model"]).name,
        "protocol": {
            "dataset": "WFLW test (2,500 faces, manual annotations)",
            "layout": "68-point iBUG (WFLW 98->68 subset; MediaPipe 478->68 vertex map)",
            "normalisation": "outer eye corners (68-pt 36/45 = WFLW 60/72)",
            "crop": "identical cached face crop (1.8x the landmark extent) for every method",
            "calibration": f"MediaPipe offset learned on {int(tr_found.sum())} WFLW training faces",
            "face_failure_threshold": thr, "point_failure_threshold": POINT_FAIL,
        },
        "detection_rate": float(found.mean()), "faces_detected": int(found.sum()), "faces_total": int(len(found)),
        "box_calibration": box_cal,
        "methods": {}, "subsets": {}, "comparison": {},
    }
    nmes, inners = {}, {}
    for key, (label, pred) in methods.items():
        on_detected = score(pred[found], gt68[found], thr)
        all_faces = score(pred, gt68, thr)
        nmes[key], inners[key] = on_detected.pop("_nme"), on_detected.pop("_inner")
        all_faces.pop("_nme"), all_faces.pop("_inner")
        report["methods"][key] = {"label": label, "detected_faces": on_detected,
                                  "all_faces": {k: all_faces[k] for k in ("faces", "nme", "failure_rate", "auc")}}
        report["subsets"][key] = {}
        for j, a in enumerate(ATTRIBUTES):
            mask = found & (test["attributes"][:, j] == 1)
            s = score(pred[mask], gt68[mask], thr)
            report["subsets"][key][a] = {k: s[k] for k in ("faces", "nme", "nme_inner", "failure_rate")}
    for key in ours:
        for ref in ("mediapipe", "mediapipe_calibrated"):
            report["comparison"][f"{key}_vs_{ref}"] = {
                "nme_reduction_pct": relative_improvement(report["methods"][ref]["detected_faces"]["nme"],
                                                          report["methods"][key]["detected_faces"]["nme"]),
                "inner_nme_reduction_pct": relative_improvement(
                    report["methods"][ref]["detected_faces"]["nme_inner"],
                    report["methods"][key]["detected_faces"]["nme_inner"]),
                "bootstrap_nme_diff": paired_bootstrap(nmes[ref], nmes[key], n_boot, cfg["seed"]),
                "bootstrap_inner_nme_diff": paired_bootstrap(inners[ref], inners[key], n_boot, cfg["seed"]),
            }
    # Detector-robust vs proposed under the deployment condition (same face location as MediaPipe).
    if "robust_mp_located" in nmes and "proposed_mp_located" in nmes:
        report["comparison"]["robust_mp_located_vs_proposed_mp_located"] = {
            "nme_reduction_pct": relative_improvement(report["methods"]["proposed_mp_located"]["detected_faces"]["nme"],
                                                      report["methods"]["robust_mp_located"]["detected_faces"]["nme"]),
            "bootstrap_nme_diff": paired_bootstrap(nmes["proposed_mp_located"], nmes["robust_mp_located"],
                                                   n_boot, cfg["seed"]),
        }

    res = resolve_path(cfg["paths"]["results"])
    save_json(report, res / "mediapipe_comparison.json")
    save_json({**box_cal, "offset_68": offset.tolist(), "source": "WFLW train faces"},
              ckpt_dir / "mediapipe_calibration.json")
    det_idx = np.flatnonzero(found)
    with (res / "mediapipe_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "image_name", *[f"nme68_{k}" for k in nmes], *ATTRIBUTES])
        for r, i in enumerate(det_idx):
            w.writerow([i, test["image_names"][i], *[f"{nmes[k][r]:.6f}" for k in nmes],
                        *test["attributes"][i].tolist()])
    _figures(report, res, thr)
    for key, m in report["methods"].items():
        d = m["detected_faces"]
        log.info("%-24s NME68 %.2f%%  inner %.2f%%  mean err %.2f px  FR %.2f%%", m["label"], d["nme"] * 100,
                 d["nme_inner"] * 100, d["mean_error_px"], d["failure_rate"] * 100)
    for k, c in report["comparison"].items():
        b = c["bootstrap_nme_diff"]
        line = f"{k}: NME change {-c['nme_reduction_pct']:+.1f}% (95% CI of drop {b['ci_low'] * 100:.2f}-{b['ci_high'] * 100:.2f} pts)"
        if "bootstrap_inner_nme_diff" in c:
            bi = c["bootstrap_inner_nme_diff"]
            line += (f"; inner {-c['inner_nme_reduction_pct']:+.1f}% (CI of drop {bi['ci_low'] * 100:.2f}-"
                     f"{bi['ci_high'] * 100:.2f} pts, p={bi['p_value']:.4f})")
        log.info(line)
    log.info("MediaPipe detected %d/%d test faces. Wrote mediapipe_comparison.json/.png", found.sum(), len(found))


def _figures(report: dict[str, Any], res: Path, thr: float) -> None:
    order = [k for k in ("mediapipe", "mediapipe_calibrated", "baseline", "proposed", "proposed_mp_located",
                         "robust", "robust_mp_located") if k in report["methods"]]
    short = {"mediapipe": "MediaPipe", "mediapipe_calibrated": "MediaPipe\ncorrected", "baseline": "Baseline\nGT crop",
             "proposed": "Proposed\nGT crop", "proposed_mp_located": "Proposed\nMP-located",
             "robust": "Robust\nGT crop", "robust_mp_located": "Robust\nMP-located"}
    labels = [short.get(k, report["methods"][k]["label"]) for k in order]
    d = [report["methods"][k]["detected_faces"] for k in order]
    colors = {"mediapipe": "#898781", "mediapipe_calibrated": "#c3c2b7", "proposed_mp_located": "#199e70",
              "robust_mp_located": "#c98500"}
    plotting.MODEL_COLORS.update(colors)
    plotting.bar_panels(
        res / "mediapipe_comparison.png", order, labels,
        [("NME, 68 points (%) - lower is better", [x["nme"] * 100 for x in d], "{:.2f}"),
         ("Mean error (px) - lower is better", [x["mean_error_px"] for x in d], "{:.2f}"),
         (f"Failure rate @{thr * 100:.0f}% (%)", [x["failure_rate"] * 100 for x in d], "{:.2f}")],
        f"Google MediaPipe vs APAC-Face on WFLW test ({report['faces_detected']:,} faces MediaPipe detected)")
    sub_order = [k for k in ("mediapipe", "mediapipe_calibrated", "proposed_mp_located", "robust_mp_located")
                 if k in report["subsets"]]
    groups = [ATTRIBUTE_LABELS[a].replace(" ", "\n", 1) for a in ATTRIBUTES]
    vals = np.array([[report["subsets"][k][a]["nme"] * 100 for a in ATTRIBUTES] for k in sub_order])
    plotting.grouped_bars(res / "mediapipe_subsets.png", groups, sub_order,
                          [report["methods"][k]["label"] for k in sub_order], vals,
                          "NME, 68 points (%) - lower is better",
                          "MediaPipe vs APAC-Face by WFLW challenge condition")


if __name__ == "__main__":
    main()
