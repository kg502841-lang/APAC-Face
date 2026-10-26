"""Machine-made landmark labels (pseudo-labels) for unannotated APAC photos, for self-training.

These labels are NOT ground truth. They are predictions of the trained heatmap model ("teacher"), made more
carefully than at run time: every face is predicted from 10 views (the photo and its mirror image, each with
the MediaPipe-located crop and 4 slightly shifted / scaled crops) and the views are averaged. The spread of
the 10 views is an uncertainty estimate; faces whose views disagree are dropped. They are stored apart from
the human labels, marked as machine labels, and never used for testing.

    python -m src.pseudo_labels copy  --source D:/Projects/Dataset1     # copy photos (source untouched)
    python -m src.pseudo_labels check                                     # teacher vs single pass, WFLW val
    python -m src.pseudo_labels label                                     # label + cache for training

`check` must show that averaged predictions are more accurate than the single pass the model was trained
to make, otherwise training on them could not teach the model anything new.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .augmentations import apply_affine, crop_matrix, warp
from .dataset import load_detector_boxes, load_split
from .inference import LandmarkPredictor
from .landmarks import flip_landmarks, interocular_distance, landmark_box
from .metrics import per_sample_nme
from .utils import get_logger, load_config, load_json, resolve_path, save_json

log = get_logger("pseudo_labels")

EXTRA_DIR = "data/APAC_extra"             # copied photos + raw pseudo-labels
CACHE_NAME = "pseudo"                     # data/cache/pseudo/*.jpg + data/cache/pseudo.npz
LABEL_SOURCE = "pseudo-label: heatmap teacher, 10-view average (machine-made, not ground truth)"
JITTER = ((0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (-0.03, 0.0, 0.0), (0.0, 0.03, 0.05), (0.0, -0.03, -0.05))


def teacher_predict(pred: LandmarkPredictor, image: np.ndarray, box: tuple[float, float, float]
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """Average of 10 views -> (98, 2) landmarks, spread (mean distance of the views to their average /
    inter-ocular distance) and the single-pass prediction for comparison."""
    w = image.shape[1]
    cx, cy, side = box
    boxes = [(cx + dx * side, cy + dy * side, side * (1 + ds)) for dx, dy, ds in JITTER]
    views = list(pred.predict_boxes(image, boxes))
    mirrored = np.ascontiguousarray(image[:, ::-1])
    boxes_m = [((w - 1) - b[0], b[1], b[2]) for b in boxes]
    views += [flip_landmarks(v, w) for v in pred.predict_boxes(mirrored, boxes_m)]
    views = np.stack(views)
    mean = views.mean(axis=0)
    iod = float(interocular_distance(mean))
    spread = float(np.linalg.norm(views - mean, axis=-1).mean() / max(iod, 1e-6))
    return mean.astype(np.float32), spread, views[0]


# ----------------------------------------------------------------------------------------------- copy

def copy_photos(source: Path, dest: Path) -> int:
    """Copy *.jpg/*.png from `source` into `dest`; the source folder is only read."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(source.iterdir()):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            target = dest / p.name
            if not target.exists() or target.stat().st_size != p.stat().st_size:
                shutil.copy2(p, target)
            n += 1
    return n


# ---------------------------------------------------------------------------------------------- check

def check(cfg: dict[str, Any], model: str, device: str) -> dict[str, Any]:
    """On WFLW validation (real labels, never trained on), seeded like deployment: is the 10-view teacher
    more accurate than one pass, and does its spread flag the faces it gets wrong?"""
    val = load_split(cfg, "val")
    boxes = load_detector_boxes(cfg, "val")
    cache = resolve_path(cfg["paths"]["cache_dir"])
    pred = LandmarkPredictor(resolve_path(cfg["paths"]["checkpoints"]) / f"{model}.pth", device)
    single, teacher, spreads, gts = [], [], [], []
    for i in np.flatnonzero(np.isfinite(boxes).all(axis=1)):
        img = np.ascontiguousarray(cv2.imread(str(cache / val["files"][i]))[:, :, ::-1])
        mean, spread, one = teacher_predict(pred, img, tuple(boxes[i]))
        teacher.append(mean); single.append(one); spreads.append(spread); gts.append(val["landmarks"][i])
    gts, spreads = np.stack(gts), np.array(spreads)
    nme_1, nme_t = per_sample_nme(np.stack(single), gts), per_sample_nme(np.stack(teacher), gts)
    out = {"faces": int(len(gts)), "single_pass_nme": float(nme_1.mean()), "teacher_nme": float(nme_t.mean()),
           "spread_vs_error_spearman": float(_spearman(spreads, nme_t)), "by_spread_quantile": {}}
    for q in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
        thr = float(np.quantile(spreads, q))
        keep = spreads <= thr
        out["by_spread_quantile"][f"{q:.2f}"] = {"threshold": thr, "kept": float(keep.mean()),
                                                 "teacher_nme_kept": float(nme_t[keep].mean()),
                                                 "failure_rate_kept": float((nme_t[keep] > 0.10).mean())}
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------------------------- label

def label(cfg: dict[str, Any], model: str, max_spread: float, device: str) -> dict[str, Any]:
    from .mediapipe_baseline import MediaPipeLandmarker, mediapipe_seed_box

    photos = resolve_path(EXTRA_DIR) / "images"
    names = sorted(p.name for p in photos.glob("*.jpg"))
    if not names:
        raise SystemExit(f"No photos in {photos} - run: python -m src.pseudo_labels copy --source <folder>")
    # Faces set aside for the human-labelled test set never enter training.
    manifest = resolve_path(cfg["paths"]["web"]) / "labeling" / "manifest.json"
    held_out = {it["id"] for it in load_json(manifest)["items"]} if manifest.is_file() else set()
    calib = load_json(resolve_path(cfg["paths"]["checkpoints"]) / "mediapipe_calibration.json")
    pred = LandmarkPredictor(resolve_path(cfg["paths"]["checkpoints"]) / f"{model}.pth", device)
    landmarker = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"])

    d = cfg["data"]
    size = int(round(d["input_size"] * d["cache_expand"]))
    cache = resolve_path(cfg["paths"]["cache_dir"])
    out_dir = cache / CACHE_NAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rec: dict[str, list] = {k: [] for k in ("files", "landmarks", "landmarks_orig", "matrices", "image_names",
                                            "groups", "spread", "det_boxes")}
    counts = {"photos": len(names), "held_out": 0, "no_face": 0, "too_uncertain": 0, "kept": 0}
    all_spreads: list[float] = []
    t0 = time.time()
    for k, name in enumerate(names):
        stem = Path(name).stem
        if stem in held_out:
            counts["held_out"] += 1
            continue
        image = np.ascontiguousarray(cv2.imread(str(photos / name))[:, :, ::-1])
        mesh = landmarker.detect(image)
        if mesh is None:
            counts["no_face"] += 1
            continue
        seed = mediapipe_seed_box(mesh, calib)
        lm, spread, _ = teacher_predict(pred, image, seed)
        all_spreads.append(spread)
        if spread > max_spread:
            counts["too_uncertain"] += 1
            continue
        cx, cy, side = landmark_box(lm, float(d["box_scale"]))
        m = crop_matrix((cx, cy, side * float(d["cache_expand"])), size)
        i = len(rec["files"])
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), warp(image, m, size)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        s = size / (side * float(d["cache_expand"]))
        c = apply_affine(np.array([[seed[0], seed[1]]]), m)[0]
        rec["files"].append(f"{CACHE_NAME}/{i:05d}.jpg")
        rec["landmarks"].append(apply_affine(lm, m))
        rec["landmarks_orig"].append(lm)
        rec["matrices"].append(m.astype(np.float32))
        rec["image_names"].append(name)
        rec["groups"].append(stem.split("_", 1)[1] if "_" in stem else "unknown")
        rec["spread"].append(spread)
        rec["det_boxes"].append([c[0], c[1], seed[2] * s])   # MediaPipe-style box in crop pixels
        counts["kept"] += 1
        if (k + 1) % 1000 == 0:
            log.info("%d/%d photos (%.0fs), kept %d", k + 1, len(names), time.time() - t0, counts["kept"])
    landmarker.close()

    n = len(rec["files"])
    np.savez_compressed(
        cache / f"{CACHE_NAME}.npz",
        files=np.array(rec["files"]), landmarks=np.stack(rec["landmarks"]).astype(np.float32),
        landmarks_orig=np.stack(rec["landmarks_orig"]).astype(np.float32), matrices=np.stack(rec["matrices"]),
        attributes=np.zeros((n, 6), np.int64), image_names=np.array(rec["image_names"]),
        groups=np.array(rec["groups"]), spread=np.array(rec["spread"], np.float32),
        det_boxes=np.array(rec["det_boxes"], np.float32), label_source=np.array(LABEL_SOURCE))
    groups, gcount = np.unique(rec["groups"], return_counts=True)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "teacher": model, "label_source": LABEL_SOURCE,
              "max_spread": max_spread, "counts": counts, "kept_by_group": dict(zip(groups.tolist(), gcount.tolist())),
              "spread_median": float(np.median(all_spreads)) if all_spreads else None}
    save_json(report, resolve_path(EXTRA_DIR) / "pseudo_labels_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-labels for unannotated APAC photos")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copy", help="copy photos into data/APAC_extra/images (source is only read)")
    c.add_argument("--source", required=True)
    k = sub.add_parser("check", help="teacher vs single pass on WFLW validation")
    k.add_argument("--model", default="heatmap")
    k.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    lab = sub.add_parser("label", help="pseudo-label the copied photos and cache them for training")
    lab.add_argument("--model", default="heatmap")
    lab.add_argument("--max-spread", type=float, default=None,
                     help="drop faces whose views disagree more (default: from the check report)")
    lab.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "copy":
        n = copy_photos(Path(args.source), resolve_path(EXTRA_DIR) / "images")
        log.info("Copied %d photos to %s (source untouched)", n, resolve_path(EXTRA_DIR) / "images")
    elif args.cmd == "check":
        r = check(cfg, args.model, args.device)
        save_json(r, resolve_path(EXTRA_DIR) / "teacher_check_val.json")
        log.info("WFLW val, %d faces: single pass NME %.2f%%, 10-view teacher %.2f%%; spread-error rank corr %.2f",
                 r["faces"], r["single_pass_nme"] * 100, r["teacher_nme"] * 100, r["spread_vs_error_spearman"])
        for q, v in r["by_spread_quantile"].items():
            log.info("  keep spread <= %.4f (%.0f%% of faces): teacher NME %.2f%%, failures %.1f%%",
                     v["threshold"], v["kept"] * 100, v["teacher_nme_kept"] * 100, v["failure_rate_kept"] * 100)
    else:
        max_spread = args.max_spread
        if max_spread is None:
            chk = resolve_path(EXTRA_DIR) / "teacher_check_val.json"
            if not chk.is_file():
                raise SystemExit("Run `python -m src.pseudo_labels check` first (or pass --max-spread)")
            max_spread = load_json(chk)["by_spread_quantile"]["0.90"]["threshold"]
        r = label(cfg, args.model, max_spread, args.device)
        log.info("Pseudo-labelled %d faces (max spread %.4f): %s; by group %s", r["counts"]["kept"], max_spread,
                 r["counts"], r["kept_by_group"])


if __name__ == "__main__":
    main()
