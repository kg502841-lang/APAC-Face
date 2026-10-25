"""Video benchmark with real ground truth: tracking accuracy, jitter and failures vs MediaPipe.

No video with per-frame manual landmarks exists in this project, so sequences are synthesised from
WFLW faces whose manual 98-point annotation is known:
  * each sequence places one cached WFLW face crop in a 480x480 frame: 30 still frames (1 s), then
    60 frames (2 s) of combined translation, rotation and scaling at 30 fps;
  * every frame gets fresh camera noise - Gaussian sensor noise, brightness/gain flicker and JPEG
    compression - so a still face still produces slightly different frames, as on a webcam;
  * ground truth per frame = the known transform applied to the manual annotation.

Every method is scored on the 68 common points, normalised by that frame's inter-ocular distance:
  still jitter   mean temporal spread (std) of each landmark over the still frames - lower = steadier
  still NME      accuracy while the face is still
  motion NME     accuracy while it moves (includes any lag a smoothing filter adds)
  failures       share of frames without a face
  lost tracks    share of sequences whose mean NME exceeds 10% (tracking on the wrong place)
  ms / frame     processing time per frame on this machine

MediaPipe runs in VIDEO mode (its own tracking). APAC-Face models run the live tracker
(src/live.py) seeded by MediaPipe's face location like the web app, then each smoothing filter is
applied to the same raw predictions.

    python -m src.video_eval --split val --faces 40 --tune   # choose the filter (never on test)
    python -m src.video_eval --split test --faces 60         # report
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import plotting
from .dataset import load_split
from .inference import LandmarkPredictor
from .landmarks import EYES_68, to_68
from .live import EmaFilter, LiveTracker, NoFilter, OneEuroFilter, make_filter
from .mediapipe_baseline import MediaPipeLandmarker, mediapipe_seed_box
from .utils import get_logger, load_config, load_json, resolve_path, save_json

log = get_logger("video_eval")

FPS = 30
STILL, MOTION, SIZE = 30, 60, 480


def make_sequence(crop: np.ndarray, gt: np.ndarray, rng: np.random.Generator
                  ) -> tuple[list[np.ndarray], np.ndarray]:
    """Frames (RGB uint8) and per-frame ground truth (T, 98, 2) for one face."""
    c_src = (crop.shape[1] - 1) / 2
    c_dst = (SIZE - 1) / 2
    phase = rng.uniform(0, 2 * np.pi)
    frames, gts = [], []
    for t in range(STILL + MOTION):
        if t < STILL:
            tx = ty = ang = 0.0
            s = 1.0
        else:
            u = (t - STILL) / MOTION
            tx = 60 * np.sin(2 * np.pi * u + phase)
            ty = 25 * np.sin(4 * np.pi * u)
            ang = 12 * np.sin(2 * np.pi * u)
            s = 1 + 0.15 * np.sin(2 * np.pi * u + phase / 2)
        m = cv2.getRotationMatrix2D((c_src, c_src), ang, s)
        m[:, 2] += (c_dst - c_src + tx, c_dst - c_src + ty)
        img = cv2.warpAffine(crop, m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = img.astype(np.float32) * rng.uniform(0.95, 1.05) + rng.uniform(-6, 6)
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
        frames.append(np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]))
        gts.append(gt @ m[:, :2].T + m[:, 2])
    return frames, np.stack(gts).astype(np.float32)


class MediaPipeSeeder:
    """Face 'detector' for the live tracker: MediaPipe's mesh -> our training-style crop box."""

    def __init__(self, landmarker: MediaPipeLandmarker, calib: dict[str, float]):
        self.landmarker, self.calib = landmarker, calib
        self.available = True

    def detect(self, image_rgb: np.ndarray) -> list[np.ndarray]:
        mesh = self.landmarker.detect(image_rgb)
        return [] if mesh is None else [mesh]

    def to_crop_box(self, mesh: np.ndarray):
        return mediapipe_seed_box(mesh, self.calib)


def track_raw(predictor: LandmarkPredictor, frames: list[np.ndarray], live_cfg: dict[str, Any]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Raw tracker output: landmarks (T, 98, 2) NaN on failure, crop sides (T,), new-track flags, ms/frame."""
    tracker = LiveTracker(predictor, live_cfg.get("redetect_every", 15), live_cfg.get("max_missed_detections", 2),
                          landmark_filter=NoFilter())
    out = np.full((len(frames), 98, 2), np.nan, np.float32)
    sides = np.full(len(frames), np.nan)
    seeded = np.zeros(len(frames), bool)
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        r = tracker.process(f, timestamp=i / FPS)
        if r.raw is not None:
            out[i], sides[i], seeded[i] = r.raw, r.box[2], r.state == "detected"
    return out, sides, seeded, (time.perf_counter() - t0) * 1000 / len(frames)


def apply_filter(flt, raw: np.ndarray, sides: np.ndarray, seeded: np.ndarray) -> np.ndarray:
    """Run a filter over a tracked sequence exactly as the live tracker would (reset on new tracks)."""
    out = np.full_like(raw, np.nan)
    flt.reset()
    for i in range(len(raw)):
        if not np.isfinite(raw[i]).all():
            flt.reset()
            continue
        if seeded[i]:
            flt.reset()
        out[i] = flt(raw[i], i / FPS, sides[i])
    return out


def sequence_metrics(pred68: np.ndarray, gt68: np.ndarray) -> dict[str, float]:
    iod = np.linalg.norm(gt68[:, EYES_68[0]] - gt68[:, EYES_68[1]], axis=-1)
    ok = np.isfinite(pred68).all(axis=(1, 2))
    err = np.linalg.norm(pred68 - gt68, axis=-1).mean(-1) / iod
    still = slice(5, STILL)  # skip the first frames (detection / filter warm-up)
    s_ok = ok[still]
    if s_ok.sum() >= 5:
        p = pred68[still][s_ok]
        jitter = float(np.sqrt(p.var(axis=0).sum(-1)).mean() / iod[still][s_ok].mean())
    else:
        jitter = float("nan")
    return {
        "still_jitter": jitter,
        "still_nme": float(np.nanmean(np.where(ok, err, np.nan)[still])) if s_ok.any() else float("nan"),
        "motion_nme": float(np.nanmean(np.where(ok, err, np.nan)[STILL:])) if ok[STILL:].any() else float("nan"),
        "all_nme": float(np.nanmean(np.where(ok, err, np.nan))) if ok.any() else float("nan"),
        "failures": float(1 - ok.mean()),
    }


def summarise(rows: list[dict[str, float]]) -> dict[str, float]:
    """Means over sequences, plus medians: a few lost tracks dominate the means of every method."""
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys} | {
        f"{k}_median": float(np.nanmedian([r[k] for r in rows])) for k in keys if k != "failures"} | {
        "lost_tracks": float(np.mean([not (r["all_nme"] <= 0.10) for r in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Video tracking benchmark with real ground truth")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--faces", type=int, default=60)
    parser.add_argument("--models", nargs="*", default=None, help="experiments (default: robust, heatmap)")
    parser.add_argument("--tune", action="store_true", help="grid-search One Euro settings (use --split val)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    if args.tune and args.split == "test":
        parser.error("tune the filter on --split val, never on the test set")
    cfg = load_config(args.config)
    live_cfg = cfg.get("live", {})
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"])
    calib = load_json(ckpt_dir / "mediapipe_calibration.json")
    models = [m for m in (args.models or ["robust", "heatmap"]) if (ckpt_dir / f"{m}.pth").is_file()]

    data = load_split(cfg, args.split)
    cache = resolve_path(cfg["paths"]["cache_dir"])
    rng = np.random.default_rng(int(cfg["seed"]) + (1 if args.split == "val" else 2))
    picks = np.sort(rng.choice(len(data["files"]), size=min(args.faces, len(data["files"])), replace=False))

    seed_mp = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"])
    predictors = {m: LandmarkPredictor(ckpt_dir / f"{m}.pth", args.device, MediaPipeSeeder(seed_mp, calib))
                  for m in models}
    filters: dict[str, Any] = {"none": NoFilter(), "ema": EmaFilter(live_cfg.get("smoothing", 0.4)),
                               "one_euro": make_filter({**live_cfg, "filter": "one_euro"})}
    grid = list(itertools.product((0.3, 0.6, 1.0, 2.0), (2.0, 5.0, 10.0, 20.0, 40.0))) if args.tune else []

    rows: dict[str, list[dict[str, float]]] = {}
    ms: dict[str, list[float]] = {}
    tune_rows: dict[tuple[str, float, float], list[dict[str, float]]] = {}
    t_start = time.time()
    for n, i in enumerate(picks):
        crop = cv2.imread(str(cache / data["files"][i]))[:, :, ::-1]
        frames, gts = make_sequence(crop, data["landmarks"][i], np.random.default_rng(int(i)))
        gt68 = to_68(gts, "wflw")
        mpv = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"], video=True)
        t0 = time.perf_counter()
        mp = [mpv.detect(f, timestamp_ms=int(k * 1000 / FPS)) for k, f in enumerate(frames)]
        ms.setdefault("Google MediaPipe", []).append((time.perf_counter() - t0) * 1000 / len(frames))
        mpv.close()
        mp68 = np.stack([to_68(p, "mediapipe") if p is not None else np.full((68, 2), np.nan) for p in mp])
        rows.setdefault("Google MediaPipe", []).append(sequence_metrics(mp68, gt68))
        for m, pred in predictors.items():
            raw, sides, seeded, per_frame = track_raw(pred, frames, live_cfg)
            label = cfg["experiments"][m].get("label", m)
            for fname, flt in filters.items():
                key = f"{label} · {fname}"
                rows.setdefault(key, []).append(sequence_metrics(to_68(apply_filter(flt, raw, sides, seeded), "wflw"), gt68))
                ms.setdefault(key, []).append(per_frame)
            for mc, beta in grid:
                f = OneEuroFilter(mc, beta, live_cfg.get("d_cutoff", 1.0))
                tune_rows.setdefault((m, mc, beta), []).append(
                    sequence_metrics(to_68(apply_filter(f, raw, sides, seeded), "wflw"), gt68))
        if (n + 1) % 10 == 0:
            log.info("%d/%d sequences (%.0fs)", n + 1, len(picks), time.time() - t_start)
    seed_mp.close()

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "split": args.split, "sequences": int(len(picks)),
        "frames_per_sequence": STILL + MOTION, "fps": FPS,
        "protocol": "WFLW faces with manual annotations animated (1 s still, 2 s moving) with camera noise "
                    "(Gaussian sigma 6, gain/brightness flicker, JPEG q75); 68 common points; NME / jitter "
                    "normalised by the per-frame inter-ocular distance",
        "filters": {"ema": {"smoothing": live_cfg.get("smoothing", 0.4)},
                    "one_euro": {k: live_cfg.get(k) for k in ("min_cutoff", "beta", "d_cutoff")}},
        "methods": {k: {**summarise(v), "ms_per_frame": float(np.mean(ms[k]))} for k, v in rows.items()},
    }
    if grid:
        tuned = {f"{m} min_cutoff={mc} beta={b}": summarise(v) for (m, mc, b), v in tune_rows.items()}
        report["one_euro_grid"] = tuned
        for m in models:
            cand = {k: v for k, v in tuned.items() if k.startswith(f"{m} ")}
            best = min(cand, key=lambda k: cand[k]["all_nme"])
            report.setdefault("one_euro_best", {})[m] = {"setting": best, **cand[best]}
            log.info("best One Euro for %s on %s: %s -> NME %.2f%%, jitter %.3f%%", m, args.split, best,
                     cand[best]["all_nme"] * 100, cand[best]["still_jitter"] * 100)
    res = resolve_path(cfg["paths"]["results"])
    save_json(report, res / f"video_benchmark_{args.split}.json")
    with (res / f"video_benchmark_{args.split}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["still_jitter", "still_jitter_median", "still_nme", "still_nme_median", "motion_nme",
                "motion_nme_median", "all_nme", "all_nme_median", "failures", "lost_tracks"]
        w.writerow(["method", *(f"{c}_pct" for c in cols), "ms_per_frame"])
        for k, v in report["methods"].items():
            w.writerow([k, *(f"{v[c] * 100:.3f}" for c in cols), f"{v['ms_per_frame']:.1f}"])
    for k, v in report["methods"].items():
        log.info("%-36s jitter %.3f%% (median %.3f%%)  still NME %.2f%% (%.2f%%)  motion NME %.2f%% (%.2f%%)  "
                 "no face %.1f%%  lost tracks %.1f%%  %.1f ms", k, v["still_jitter"] * 100,
                 v["still_jitter_median"] * 100, v["still_nme"] * 100, v["still_nme_median"] * 100,
                 v["motion_nme"] * 100, v["motion_nme_median"] * 100, v["failures"] * 100, v["lost_tracks"] * 100,
                 v["ms_per_frame"])
    if args.split == "test":
        _figure(report, res / "video_benchmark.png")


def _figure(report: dict[str, Any], path: Path) -> None:
    keys = list(report["methods"])
    short = {"none": "raw", "ema": "EMA", "one_euro": "One Euro"}
    split = [k.split(" · ") for k in keys]
    labels = ["MediaPipe" if len(s) == 1 else f"{s[0].split(' ')[0].replace('Detector-robust', 'Robust')}\n{short.get(s[1], s[1])}" for s in split]
    names = [f"m{i}" for i in range(len(keys))]
    # colour follows the model (MediaPipe grey, one hue per APAC-Face model), shade follows the filter
    shades = [["#f5c766", "#eda100", "#c98500"], ["#f2b3cb", "#e87ba4", "#c9628a"], ["#9fc3ef", "#2a78d6", "#1f5ca8"]]
    models = list(dict.fromkeys(s[0] for s in split if len(s) == 2))
    filters = {"none": 0, "ema": 1, "one_euro": 2}
    for n, s in zip(names, split):
        plotting.MODEL_COLORS[n] = ("#898781" if len(s) == 1
                                    else shades[models.index(s[0]) % len(shades)][filters.get(s[1], 1)])
    m = report["methods"]
    plotting.bar_panels(path, names, labels,
                        [("Median still jitter (% of inter-ocular) - lower is steadier",
                          [m[k]["still_jitter_median"] * 100 for k in keys], "{:.2f}"),
                         ("Median NME while moving (%) - lower is better",
                          [m[k]["motion_nme_median"] * 100 for k in keys], "{:.2f}"),
                         ("Lost tracks (% of sequences, NME > 10%)", [m[k]["lost_tracks"] * 100 for k in keys], "{:.1f}")],
                        f"Video benchmark: {report['sequences']} WFLW test faces, {report['frames_per_sequence']} frames each")


if __name__ == "__main__":
    main()
