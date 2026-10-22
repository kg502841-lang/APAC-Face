"""Label-free landmark consistency per demographic group.

Input: a folder or .zip of face images named {id}_{Group}.jpg, e.g. 000016_Southeast_Asian.jpg
(the layout of the FairFace-style "APAC asian images" archive). Group labels are taken ONLY from
the dataset's own metadata (the file name) - never inferred from appearance.

These images have NO ground-truth landmarks, so accuracy (NME against ground truth) cannot be
measured on them. Two label-free proxies are reported per group and per model instead:

  flip consistency  predict on the image and on its mirror image, un-mirror the second prediction
                    (with left/right index remapping) and measure the disagreement / inter-ocular
                    distance. An ideal landmark detector is exactly flip-equivariant.
  crop stability    predict from K slightly shifted / scaled face boxes and measure how far the
                    predictions scatter (mean distance to their centroid / inter-ocular distance).

Lower is better for both. They expose unstable predictions, not systematic bias: a model that is
consistently wrong in the same way still scores well. A true per-population accuracy claim needs
an annotated APAC test set (see README, "Evaluating on an annotated APAC set").

    python -m src.apac_eval --source "D:/data/APAC asian images.zip" --per-group 300

When an annotated APAC set exists, true per-group accuracy (NME against ground truth) is computed:

    python -m src.apac_eval --annotations apac_test.txt --images data/APAC_annotated [--groups groups.csv]

The annotation file uses the WFLW line format: 196 coordinates, optionally the 4 rectangle values and
6 attribute flags, then the image path relative to --images. Groups come from {id}_{Group} file names
or from a CSV with columns image,group.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import plotting
from .inference import LandmarkPredictor, decode_image, load_predictors
from .landmarks import flip_landmarks, interocular_distance
from .metrics import bootstrap_mean_ci, paired_bootstrap, relative_improvement
from .utils import get_logger, load_config, resolve_path, save_json

log = get_logger("apac_eval")

NAME_RE = re.compile(r"^(?P<id>\d+)_(?P<group>[A-Za-z_]+)\.(jpe?g|png)$", re.IGNORECASE)
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

DISCLAIMER = (
    "Per-group values are label-free consistency proxies (flip consistency and crop stability), "
    "NOT accuracy against ground-truth landmarks, because the APAC images carry no landmark "
    "annotations. Group labels come from the dataset's file names (annotator-provided demographic "
    "metadata), not from inference on facial appearance. Lower is better.")


class ImageSource:
    """Read images from a directory tree or a .zip archive without extracting it."""

    def __init__(self, path: str | Path):
        self.path = resolve_path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"APAC image source not found: {self.path}\n"
                "Pass --source <folder or .zip> with images named {id}_{Group}.jpg, "
                "or set paths.apac_source in config.yaml.")
        self.zip = zipfile.ZipFile(self.path) if self.path.suffix.lower() == ".zip" else None

    def names(self) -> list[str]:
        if self.zip:
            return [n for n in self.zip.namelist() if Path(n).suffix.lower() in IMAGE_EXT]
        return [p.relative_to(self.path).as_posix() for p in self.path.rglob("*")
                if p.suffix.lower() in IMAGE_EXT]

    def read(self, name: str) -> bytes:
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()


def group_of(name: str) -> str | None:
    m = NAME_RE.match(Path(name).name)
    return m.group("group") if m else None


def sample_by_group(names: Iterable[str], per_group: int, seed: int) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        g = group_of(n)
        if g:
            groups[g].append(n)
    rng = np.random.default_rng(seed)
    out = {}
    for g in sorted(groups):
        items = sorted(groups[g])
        k = min(per_group, len(items))
        out[g] = [items[i] for i in sorted(rng.choice(len(items), size=k, replace=False))]
    return out


def consistency_for_image(predictors: dict[str, LandmarkPredictor], image: np.ndarray,
                          jitter: np.ndarray) -> tuple[bool, dict[str, tuple[float, float]]]:
    """Returns (face detected, {model: (flip consistency, crop stability)})."""
    h, w = image.shape[:2]
    first = next(iter(predictors.values()))
    box0, detected, _, _ = first.initial_box(image)          # shared by every model
    mirrored = np.ascontiguousarray(image[:, ::-1])
    box0_m = ((w - 1) - box0[0], box0[1], box0[2])
    out = {}
    for name, p in predictors.items():
        pred = p.predict(image, box=box0)
        pred_m = flip_landmarks(p.predict(mirrored, box=box0_m).landmarks, w)
        iod = float(interocular_distance((pred.landmarks + pred_m) / 2))
        if not np.isfinite(iod) or iod <= 1:
            out[name] = (float("nan"), float("nan"))
            continue
        flip = float(np.linalg.norm(pred.landmarks - pred_m, axis=-1).mean() / iod)
        cx, cy, side = pred.box
        boxes = [(cx + dx * side, cy + dy * side, side * (1 + ds)) for dx, dy, ds in jitter]
        preds = p.predict_boxes(image, boxes)
        centroid = preds.mean(axis=0)
        stab = float(np.linalg.norm(preds - centroid, axis=-1).mean() / iod)
        out[name] = (flip, stab)
    return detected, out


def run(cfg: dict[str, Any], source: str, per_group: int, device: str = "auto") -> dict[str, Any]:
    acfg = cfg["apac"]
    predictors = load_predictors(cfg, device)
    if not predictors:
        raise SystemExit("No trained checkpoints found - train the models first.")
    src = ImageSource(source)
    sampled = sample_by_group(src.names(), per_group, int(cfg["seed"]))
    if not sampled:
        raise SystemExit(f"No images named {{id}}_{{Group}}.jpg found in {src.path}")
    rng = np.random.default_rng(int(cfg["seed"]))
    k = int(acfg["jitter_samples"])
    shift, scale = float(acfg["jitter_shift"]), float(acfg["jitter_scale"])

    rows: list[dict[str, Any]] = []
    sizes: dict[str, int] = defaultdict(int)
    t0 = time.time()
    for g, names in sampled.items():
        for name in names:
            try:
                image = decode_image(src.read(name))
            except ValueError as e:
                log.warning("skip %s: %s", name, e)
                continue
            sizes[f"{image.shape[1]}x{image.shape[0]}"] += 1
            jitter = np.column_stack([rng.uniform(-shift, shift, (k, 2)), rng.uniform(-scale, scale, k)])
            detected, res = consistency_for_image(predictors, image, jitter)
            for model, (flip, stab) in res.items():
                rows.append({"image": name, "group": g, "model": model, "face_detected": detected,
                             "flip_consistency": flip, "crop_stability": stab})
        log.info("%-16s %d images done (%.0fs)", g, len(names), time.time() - t0)
    return summarise(cfg, rows, list(predictors), src.path, dict(sizes))


def summarise(cfg: dict[str, Any], rows: list[dict[str, Any]], models: list[str], source: Path,
              sizes: dict[str, int]) -> dict[str, Any]:
    acfg = cfg["apac"]
    labels = acfg.get("group_labels", {})
    apac_groups = [g for g in acfg["apac_groups"]]
    groups = sorted({r["group"] for r in rows}, key=lambda g: (g not in apac_groups, labels.get(g, g)))
    n_boot = int(cfg["eval"]["bootstrap_samples"])

    def vals(model: str, group: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["model"] == model and r["group"] == group], float)

    per_group: dict[str, Any] = {}
    for g in groups:
        det = [r["face_detected"] for r in rows if r["group"] == g and r["model"] == models[0]]
        entry = {"label": labels.get(g, g.replace("_", " ")), "apac": g in apac_groups,
                 "images": len(det), "face_detection_rate": float(np.mean(det)) if det else float("nan"),
                 "models": {}}
        for m in models:
            f, s = vals(m, g, "flip_consistency"), vals(m, g, "crop_stability")
            entry["models"][m] = {
                "flip_consistency": float(np.nanmean(f)), "flip_ci": bootstrap_mean_ci(f, n_boot, cfg["seed"]),
                "crop_stability": float(np.nanmean(s)), "stability_ci": bootstrap_mean_ci(s, n_boot, cfg["seed"]),
            }
        per_group[g] = entry

    summary_models: dict[str, Any] = {}
    for m in models:
        apac_vals = {g: per_group[g]["models"][m]["flip_consistency"] for g in groups if g in apac_groups}
        all_vals = {g: per_group[g]["models"][m]["flip_consistency"] for g in groups}
        summary_models[m] = {
            "apac_mean_flip_consistency": float(np.mean(list(apac_vals.values()))) if apac_vals else None,
            "apac_worst_group": max(apac_vals, key=apac_vals.get) if apac_vals else None,
            "apac_gap_flip_consistency": (max(apac_vals.values()) - min(apac_vals.values())) if apac_vals else None,
            "all_groups_gap_flip_consistency": max(all_vals.values()) - min(all_vals.values()),
        }

    base, prop = cfg["eval"]["baseline"], cfg["eval"]["proposed"]
    comparison: dict[str, Any] = {}
    if base in models and prop in models:
        for g in groups:
            fb, fp = vals(base, g, "flip_consistency"), vals(prop, g, "flip_consistency")
            sb, sp = vals(base, g, "crop_stability"), vals(prop, g, "crop_stability")
            comparison[g] = {
                "flip_improvement_pct": relative_improvement(np.nanmean(fb), np.nanmean(fp)),
                "flip_bootstrap": paired_bootstrap(fb, fp, n_boot, cfg["seed"]),
                "stability_improvement_pct": relative_improvement(np.nanmean(sb), np.nanmean(sp)),
                "stability_bootstrap": paired_bootstrap(sb, sp, n_boot, cfg["seed"]),
            }

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source.name,
        "image_sizes": sizes,
        "jitter": {"samples": acfg["jitter_samples"], "shift": acfg["jitter_shift"], "scale": acfg["jitter_scale"]},
        "disclaimer": DISCLAIMER,
        "models": models,
        "apac_groups": apac_groups,
        "groups": per_group,
        "summary": summary_models,
        "comparison": comparison,
    }
    res = resolve_path(cfg["paths"]["results"])
    save_json(report, res / "apac_consistency.json")
    with (res / "apac_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (res / "apac_consistency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "label", "apac", "images", "face_detection_rate", "model",
                    "flip_consistency_pct", "flip_ci_low_pct", "flip_ci_high_pct",
                    "crop_stability_pct", "stability_ci_low_pct", "stability_ci_high_pct"])
        for g, e in per_group.items():
            for m, v in e["models"].items():
                w.writerow([g, e["label"], e["apac"], e["images"], f"{e['face_detection_rate']:.3f}", m,
                            f"{v['flip_consistency'] * 100:.3f}", f"{v['flip_ci'][0] * 100:.3f}",
                            f"{v['flip_ci'][1] * 100:.3f}", f"{v['crop_stability'] * 100:.3f}",
                            f"{v['stability_ci'][0] * 100:.3f}", f"{v['stability_ci'][1] * 100:.3f}"])
    _figure(cfg, report, res / "apac_group_analysis.png")
    return report


def _figure(cfg: dict[str, Any], report: dict[str, Any], path: Path) -> None:
    groups = list(report["groups"])
    models = report["models"]
    labels = [cfg["experiments"].get(m, {}).get("label", m) for m in models]
    names = [report["groups"][g]["label"] + ("" if report["groups"][g]["apac"] else "\n(reference)")
             for g in groups]
    vals = np.array([[report["groups"][g]["models"][m]["flip_consistency"] * 100 for g in groups] for m in models])
    lo = np.array([[report["groups"][g]["models"][m]["flip_ci"][0] * 100 for g in groups] for m in models])
    hi = np.array([[report["groups"][g]["models"][m]["flip_ci"][1] * 100 for g in groups] for m in models])
    plotting.grouped_bars(path, names, models, labels, vals,
                          "Flip inconsistency (% of inter-ocular) - lower is better",
                          "Label-free flip inconsistency by demographic group (95% CI) - not ground-truth accuracy",
                          errors=np.stack([vals - lo, hi - vals]))


def parse_annotated_set(path: Path) -> list[tuple[str, np.ndarray]]:
    """WFLW-format lines with or without the rectangle/attribute columns -> (image, landmarks)."""
    out = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) not in (197, 207):
                raise ValueError(f"{path.name}:{lineno}: expected 197 or 207 fields, found {len(parts)}")
            out.append((parts[-1].replace("\\", "/"),
                        np.asarray(parts[:196], dtype=np.float32).reshape(98, 2)))
    return out


def annotated_accuracy(cfg: dict[str, Any], annotations: str, images: str, groups_csv: str | None,
                       device: str = "auto") -> dict[str, Any]:
    from .landmarks import landmark_box
    from .metrics import per_sample_nme, summarize

    predictors = load_predictors(cfg, device)
    if not predictors:
        raise SystemExit("No trained checkpoints found - train the models first.")
    entries = parse_annotated_set(resolve_path(annotations))
    root = resolve_path(images)
    group_map: dict[str, str] = {}
    if groups_csv:
        with resolve_path(groups_csv).open(encoding="utf-8") as f:
            group_map = {r["image"].replace("\\", "/"): r["group"] for r in csv.DictReader(f)}
    box_scale = float(cfg["data"]["box_scale"])
    thr = float(cfg["eval"]["failure_threshold"])
    gts, preds, grp = [], defaultdict(list), []
    for name, gt in entries:
        g = group_map.get(name) or group_of(name)
        if g is None:
            log.warning("skip %s: no group label (use {id}_{Group}.jpg names or --groups)", name)
            continue
        image = decode_image((root / name).read_bytes())
        box = landmark_box(gt, box_scale)  # same ground-truth crop protocol as the WFLW evaluation
        for m, p in predictors.items():
            preds[m].append(p.predict(image, box=box, refine_iterations=0).landmarks)
        gts.append(gt)
        grp.append(g)
    gts_a, grp_a = np.stack(gts), np.array(grp)
    report: dict[str, Any] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "faces": len(gts),
                              "annotations": Path(annotations).name, "groups": {}, "models": list(predictors)}
    nmes = {m: per_sample_nme(np.stack(v), gts_a) for m, v in preds.items()}
    for g in sorted(set(grp)):
        mask = grp_a == g
        report["groups"][g] = {m: summarize(nmes[m][mask], None, thr) for m in predictors}
    base, prop = cfg["eval"]["baseline"], cfg["eval"]["proposed"]
    if base in nmes and prop in nmes:
        report["comparison"] = {
            g: {"relative_improvement_pct": relative_improvement(report["groups"][g][base]["nme"],
                                                                 report["groups"][g][prop]["nme"]),
                "bootstrap_nme_diff": paired_bootstrap(nmes[base][grp_a == g], nmes[prop][grp_a == g],
                                                       int(cfg["eval"]["bootstrap_samples"]), cfg["seed"])}
            for g in report["groups"]}
    save_json(report, resolve_path(cfg["paths"]["results"]) / "apac_annotated_accuracy.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-group landmark consistency on APAC images")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None, help="folder or .zip of {id}_{Group}.jpg images")
    parser.add_argument("--per-group", type=int, default=None)
    parser.add_argument("--annotations", default=None, help="WFLW-format ground truth for APAC faces")
    parser.add_argument("--images", default=None, help="image root for --annotations")
    parser.add_argument("--groups", default=None, help="optional CSV with columns image,group")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.annotations:
        if not args.images:
            parser.error("--images is required with --annotations")
        report = annotated_accuracy(cfg, args.annotations, args.images, args.groups, args.device)
        for g, per_model in report["groups"].items():
            cells = "  ".join(f"{m} NME {v['nme'] * 100:.2f}%" for m, v in per_model.items())
            log.info("%-18s n=%d  %s", g, next(iter(per_model.values()))["n"], cells)
        log.info("Wrote results/apac_annotated_accuracy.json")
        return
    report = run(cfg, args.source or cfg["paths"]["apac_source"],
                 args.per_group or int(cfg["apac"]["per_group"]), args.device)
    for g, e in report["groups"].items():
        cells = "  ".join(f"{m} {v['flip_consistency'] * 100:.2f}%" for m, v in e["models"].items())
        log.info("%-24s n=%d det=%.0f%%  flip: %s", e["label"], e["images"], e["face_detection_rate"] * 100, cells)
    log.info("Wrote apac_consistency.json/.csv, apac_per_image.csv, apac_group_analysis.png")


if __name__ == "__main__":
    main()
