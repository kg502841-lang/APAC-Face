"""East Asian vs Caucasian landmark accuracy on SCUT-FBP5500 (manual 86-point annotations).

SCUT-FBP5500 (Liang et al., ICPR 2018; non-commercial research use only) has 5,500 frontal faces:
2,000 Asian female, 2,000 Asian male, 750 Caucasian female, 750 Caucasian male (file prefixes AF, AM,
CF, CM). Its README states the 86 points were labelled by hand. None of our models saw these images.

Its 86-point layout differs from the 68-point iBUG layout every comparison here uses. Only points with the
same anatomical definition in both are scored - no point is interpolated:

    brows   upper edge, 5 per brow        iBUG 17-26  <- SCUT 26-22, 36-32
    eyes    corners + the lid points next to each corner (the same lid points WFLW -> 68 keeps)
                                          iBUG 36-47  <- SCUT 46,45,43,42,49,47 / 54,53,51,50,57,55
    nose    bottom centre under the tip   iBUG 33     <- SCUT 66
    mouth   upper outer lip, 7 points     iBUG 48-54  <- SCUT 79-73

Left out: the jaw (SCUT's outline has 15 points from eyebrow height and also covers the forehead), the nose
bridge and nostrils (SCUT traces the sides of the nose instead), the lower lip (6 points vs 5) and the inner
lip (not in SCUT). The ordering behind this map was checked on the mean shape of all faces (>99.6% of faces
agree on every ordering it relies on). NME = mean error over these 30 points / outer-eye-corner distance.

APAC-Face is located by MediaPipe's face detection (one pass, as in the web app); every method is scored on
the faces MediaPipe finds.

    python -m src.scut_eval --root data/SCUT-FBP5500/SCUT-FBP5500_v2
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import plotting
from .inference import load_predictors
from .landmarks import to_68
from .metrics import failure_rate, paired_bootstrap
from .utils import get_logger, load_config, load_json, resolve_path, save_json

log = get_logger("scut_eval")

# iBUG-68 index -> SCUT-FBP5500 index, for points with the same anatomical definition (see module docstring)
SCUT_TO_68: dict[int, int] = {
    17: 26, 18: 25, 19: 24, 20: 23, 21: 22, 22: 36, 23: 35, 24: 34, 25: 33, 26: 32,
    33: 66,
    36: 46, 37: 45, 38: 43, 39: 42, 40: 49, 41: 47, 42: 54, 43: 53, 44: 51, 45: 50, 46: 57, 47: 55,
    48: 79, 49: 78, 50: 77, 51: 76, 52: 75, 53: 74, 54: 73,
}
COMMON_68 = tuple(sorted(SCUT_TO_68))
REGIONS = {"Eyebrows": range(17, 27), "Eyes": range(36, 48), "Nose + upper lip": (33, *range(48, 55))}
GROUPS = {"AF": ("Asian", "female"), "AM": ("Asian", "male"), "CF": ("Caucasian", "female"), "CM": ("Caucasian", "male")}
NUM_POINTS = 86


def read_pts(path: Path) -> np.ndarray | None:
    """SCUT .pts files are binary: int32 point count, then float32 (x, y) pairs."""
    raw = path.read_bytes()
    if len(raw) < 4:
        return None
    n = int(np.frombuffer(raw[:4], "<i4")[0])
    if n != NUM_POINTS or len(raw) != 4 + 8 * n:
        return None
    return np.frombuffer(raw[4:], "<f4").reshape(n, 2).astype(np.float32)


def scut_68(pts86: np.ndarray) -> np.ndarray:
    """(86, 2) -> (68, 2) with NaN where SCUT has no corresponding point."""
    out = np.full((68, 2), np.nan, np.float32)
    for i, j in SCUT_TO_68.items():
        out[i] = pts86[j]
    return out


def nme_common(pred68: np.ndarray, gt68: np.ndarray, points=COMMON_68) -> np.ndarray:
    """Per-face NME over `points`, normalised by the ground-truth outer eye corners (36, 45)."""
    iod = np.linalg.norm(gt68[:, 36] - gt68[:, 45], axis=-1)
    idx = list(points)
    return np.linalg.norm(pred68[:, idx] - gt68[:, idx], axis=-1).mean(-1) / iod


def gap_bootstrap(a_vals: np.ndarray, c_vals: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """CI of mean(Asian) - mean(Caucasian): faces resampled within each group."""
    rng = np.random.default_rng(seed)
    gaps = [rng.choice(a_vals, len(a_vals)).mean() - rng.choice(c_vals, len(c_vals)).mean() for _ in range(n_boot)]
    return {"gap": float(a_vals.mean() - c_vals.mean()), "ci_low": float(np.percentile(gaps, 2.5)),
            "ci_high": float(np.percentile(gaps, 97.5))}


def gap_difference_bootstrap(a: dict[str, np.ndarray], c: dict[str, np.ndarray], m1: str, m2: str,
                             n_boot: int, seed: int) -> dict[str, float]:
    """CI of gap(m1) - gap(m2), resampling the same faces for both methods."""
    rng = np.random.default_rng(seed)
    na, nc = len(a[m1]), len(c[m1])
    diffs = []
    for _ in range(n_boot):
        ia, ic = rng.integers(0, na, na), rng.integers(0, nc, nc)
        diffs.append((a[m1][ia].mean() - c[m1][ic].mean()) - (a[m2][ia].mean() - c[m2][ic].mean()))
    d = (a[m1].mean() - c[m1].mean()) - (a[m2].mean() - c[m2].mean())
    return {"difference": float(d), "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_value": float(min(1.0, 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))))}


def run(cfg: dict[str, Any], root: Path, device: str = "auto", limit: int | None = None) -> dict[str, Any]:
    from .mediapipe_baseline import MediaPipeLandmarker, apply_offset, mediapipe_seed_box

    img_dir, pts_dir = root / "Images", root / "facial landmark"
    if not img_dir.is_dir() or not pts_dir.is_dir():
        raise SystemExit(f"{root} must contain Images/ and 'facial landmark/' (SCUT-FBP5500_v2 layout)")
    names = sorted(p.stem for p in img_dir.glob("*.jpg") if p.stem[:2] in GROUPS)
    if limit:
        names = names[::max(1, len(names) // limit)][:limit]
    calib = load_json(resolve_path(cfg["paths"]["checkpoints"]) / "mediapipe_calibration.json")
    offset = np.asarray(calib["offset_68"], np.float32)
    predictors = load_predictors(cfg, device)
    landmarker = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"])
    methods = ["mediapipe", "mediapipe_calibrated", *predictors]

    gts, keys, found, missing = [], [], [], []
    preds: dict[str, list[np.ndarray]] = defaultdict(list)
    t0 = time.time()
    for k, name in enumerate(names):
        pts = read_pts(pts_dir / f"{name}.pts")
        if pts is None:
            missing.append(name)
            continue
        image = np.ascontiguousarray(cv2.imread(str(img_dir / f"{name}.jpg"))[:, :, ::-1])
        mesh = landmarker.detect(image)
        if mesh is None:
            found.append(False)
            for m in methods:
                preds[m].append(np.full((68, 2), np.nan, np.float32))
        else:
            found.append(True)
            mp68 = to_68(mesh, "mediapipe").astype(np.float32)
            preds["mediapipe"].append(mp68)
            preds["mediapipe_calibrated"].append(apply_offset(mp68[None], offset)[0])
            box = mediapipe_seed_box(mesh, calib)
            for m, p in predictors.items():
                preds[m].append(to_68(p.predict_boxes(image, [box])[0], "wflw"))
        gts.append(scut_68(pts))
        keys.append(name)
        if (k + 1) % 1000 == 0:
            log.info("%d/%d faces (%.0fs)", k + 1, len(names), time.time() - t0)
    landmarker.close()

    gt = np.stack(gts)
    found_a = np.array(found)
    prefix = np.array([n[:2] for n in keys])
    race = np.array([GROUPS[p][0] for p in prefix])
    thr = float(cfg["eval"]["failure_threshold"])
    n_boot = int(cfg["eval"]["bootstrap_samples"])
    nme = {m: nme_common(np.stack(preds[m]), gt) for m in methods}
    region_nme = {m: {r: nme_common(np.stack(preds[m]), gt, pts) for r, pts in REGIONS.items()} for m in methods}
    labels = {"mediapipe": "Google MediaPipe", "mediapipe_calibrated": "MediaPipe, offset-corrected"}
    labels |= {m: cfg["experiments"].get(m, {}).get("label", m) for m in predictors}
    deployed = cfg["inference"].get("web_model", cfg["inference"]["default_model"])

    def summary(mask: np.ndarray) -> dict[str, Any]:
        sel = mask & found_a
        return {"faces": int(sel.sum()), "faces_total": int(mask.sum()),
                "mediapipe_detection_rate": float(found_a[mask].mean()) if mask.any() else float("nan"),
                "methods": {m: {"nme": float(nme[m][sel].mean()), "nme_median": float(np.median(nme[m][sel])),
                                "failure_rate": failure_rate(nme[m][sel], thr),
                                "regions": {r: float(v[sel].mean()) for r, v in region_nme[m].items()}}
                            for m in methods}}

    groups = {"Asian": race == "Asian", "Caucasian": race == "Caucasian"}
    groups |= {f"{GROUPS[p][0]} {GROUPS[p][1]}": prefix == p for p in GROUPS}
    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": "SCUT-FBP5500 v2 (non-commercial research use)",
        "faces": int(len(gt)), "missing_annotations": missing, "mediapipe_found": int(found_a.sum()),
        "deployed_model": deployed, "points": list(COMMON_68), "scut_to_68": SCUT_TO_68,
        "protocol": ("30 points with the same definition in SCUT's 86-point and the iBUG 68-point layout "
                     "(brows, eyes, nose bottom centre, upper outer lip); NME normalised by the outer eye corners; "
                     "APAC-Face located by MediaPipe's face detection; faces MediaPipe misses are excluded for all."),
        "methods": labels, "groups": {g: summary(mask) for g, mask in groups.items()},
    }
    a_mask, c_mask = groups["Asian"] & found_a, groups["Caucasian"] & found_a
    a_vals = {m: nme[m][a_mask] for m in methods}
    c_vals = {m: nme[m][c_mask] for m in methods}
    report["gap_asian_minus_caucasian"] = {m: gap_bootstrap(a_vals[m], c_vals[m], n_boot, cfg["seed"]) for m in methods}
    report["comparison"] = {}
    for ref in ("mediapipe", "mediapipe_calibrated", "baseline"):
        if ref not in nme or ref == deployed:
            continue
        for g in ("Asian", "Caucasian"):
            mask = groups[g] & found_a
            report["comparison"][f"{deployed}_vs_{ref}_{g}"] = paired_bootstrap(nme[ref][mask], nme[deployed][mask],
                                                                                n_boot, cfg["seed"])
        report["comparison"][f"gap_{ref}_minus_gap_{deployed}"] = gap_difference_bootstrap(
            a_vals, c_vals, ref, deployed, n_boot, cfg["seed"])

    res = resolve_path(cfg["paths"]["results"])
    save_json(report, res / "scut_benchmark.json")
    with (res / "scut_per_face.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "group", "mediapipe_found", *(f"nme_{m}" for m in methods)])
        for i, name in enumerate(keys):
            w.writerow([name, prefix[i], bool(found_a[i]),
                        *(f"{nme[m][i]:.5f}" if np.isfinite(nme[m][i]) else "" for m in methods)])
    _figure(report, res / "scut_benchmark.png", methods)
    return report


def _figure(report: dict[str, Any], path: Path, methods: list[str]) -> None:
    order = [m for m in ("mediapipe", "mediapipe_calibrated", "baseline", "proposed", "robust", "heatmap") if m in methods]
    plotting.MODEL_COLORS.update({"mediapipe": "#898781", "mediapipe_calibrated": "#c3c2b7"})
    short = {"mediapipe": "MediaPipe", "mediapipe_calibrated": "MediaPipe\ncorrected", "baseline": "Baseline",
             "proposed": "Proposed", "robust": "Robust", "heatmap": "Heatmap"}
    panels = [(f"{g} faces (n = {report['groups'][g]['faces']:,}): NME (%)",
               [report["groups"][g]["methods"][m]["nme"] * 100 for m in order], "{:.2f}")
              for g in ("Asian", "Caucasian")]
    plotting.bar_panels(path, order, [short.get(m, m) for m in order], panels,
                        "SCUT-FBP5500, 30 common points, manual labels - lower is better")


def main() -> None:
    parser = argparse.ArgumentParser(description="Asian vs Caucasian accuracy on SCUT-FBP5500")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default="data/SCUT-FBP5500/SCUT-FBP5500_v2")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only this many faces (smoke test)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = run(cfg, resolve_path(args.root), args.device, args.limit)
    log.info("%d faces with annotations (%d found by MediaPipe); missing/empty annotations: %s", report["faces"],
             report["mediapipe_found"], ", ".join(report["missing_annotations"]) or "none")
    for g, e in report["groups"].items():
        cells = "  ".join(f"{m} {v['nme'] * 100:.2f}%" for m, v in e["methods"].items())
        log.info("%-18s n=%-5d %s", g, e["faces"], cells)
    for m, v in report["gap_asian_minus_caucasian"].items():
        log.info("gap Asian - Caucasian  %-22s %+.2f pts [%+.2f, %+.2f]", m, v["gap"] * 100, v["ci_low"] * 100, v["ci_high"] * 100)
    log.info("Wrote results/scut_benchmark.json, scut_per_face.csv, scut_benchmark.png")


if __name__ == "__main__":
    main()
