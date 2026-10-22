"""Evaluate trained models on the WFLW test set and write the comparison artefacts.

    python -m src.evaluate                                 # every experiment with a checkpoint
    python -m src.evaluate --experiments baseline proposed

Outputs in results/:
    metrics.json            all numbers below + bootstrap significance of the improvement
    comparison.csv          one row per model (NME, FR, AUC, MAE, latency, improvement)
    subgroup_metrics.csv    one row per (model, WFLW challenge subset)
    per_sample_<exp>.csv    per-face NME/MAE with WFLW attribute flags
    nme_comparison.png, subgroup_analysis.png, ced_curve.png, training_loss.png

The WFLW subsets (pose, expression, illumination, make-up, occlusion, blur) are IMAGING
CONDITIONS, not demographic groups. WFLW carries no ethnicity labels, so these results say
nothing about East/South/Southeast Asian accuracy - see src/apac_eval.py for that.
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import plotting
from .augmentations import AugmentationParams, LandmarkAugmenter, crop_matrix
from .dataset import (ATTRIBUTE_LABELS, ATTRIBUTES, DatasetNotFoundError, WFLWCropDataset,
                      build_cache, canonical_cache_box, load_split)
from .metrics import (ced_curve, paired_bootstrap, per_sample_mae, per_sample_nme,
                      relative_improvement, summarize)
from .model import count_parameters, load_inference_checkpoint
from .utils import (describe_device, get_device, get_logger, load_config, load_json,
                    resolve_path, save_json)

log = get_logger("evaluate")

DISCLAIMER = ("WFLW challenge subsets are imaging conditions (pose, expression, illumination, "
              "make-up, occlusion, blur), not demographic groups. WFLW has no ethnicity labels, so "
              "these numbers do not measure accuracy on East, South or Southeast Asian faces.")


def _to3(m: np.ndarray) -> np.ndarray:
    out = np.eye(3)
    out[:2] = m
    return out


def input_to_original(cfg: dict[str, Any], matrices: np.ndarray) -> np.ndarray:
    """Per-face 2x3 matrices mapping network-input pixels back to original-image pixels."""
    size = int(cfg["data"]["input_size"])
    canon = _to3(crop_matrix(canonical_cache_box(cfg), size))  # cache -> input
    inv_canon = np.linalg.inv(canon)
    return np.stack([(np.linalg.inv(_to3(m)) @ inv_canon)[:2] for m in matrices])


@torch.no_grad()
def predict_test_set(model: torch.nn.Module, cfg: dict[str, Any], device: torch.device,
                     batch_size: int = 64) -> np.ndarray:
    size = int(cfg["data"]["input_size"])
    params = AugmentationParams.from_config(cfg["augmentation"])
    ds = WFLWCropDataset(cfg, "test", LandmarkAugmenter(params, size, enabled=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    out = np.zeros((len(ds), 98, 2), np.float32)
    for images, _, idx in loader:
        pred = model(images.to(device, non_blocking=True)).float().cpu().numpy() * size
        out[idx.numpy()] = pred
    to_orig = input_to_original(cfg, load_split(cfg, "test")["matrices"])
    return (np.einsum("nij,nkj->nki", to_orig[:, :, :2], out) + to_orig[:, None, :, 2]).astype(np.float32)


@torch.no_grad()
def measure_latency(model: torch.nn.Module, device: torch.device, size: int, runs: int) -> dict[str, Any]:
    def bench(m: torch.nn.Module, dev: torch.device, n: int) -> list[float]:
        x = torch.randn(1, 3, size, size, device=dev)
        for _ in range(10):
            m(x)
        times = []
        for _ in range(n):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t = time.perf_counter()
            m(x)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times.append((time.perf_counter() - t) * 1000)
        return times

    result: dict[str, Any] = {"batch_size": 1, "input": f"{size}x{size}"}
    if device.type == "cuda":
        t = bench(model, device, runs)
        result.update(gpu=describe_device(device), gpu_ms_median=float(np.median(t)),
                      gpu_ms_p95=float(np.percentile(t, 95)))
    cpu_model = copy.deepcopy(model).cpu().eval()
    t = bench(cpu_model, torch.device("cpu"), max(10, runs // 5))
    result.update(cpu_ms_median=float(np.median(t)), cpu_ms_p95=float(np.percentile(t, 95)),
                  cpu_threads=torch.get_num_threads())
    return result


def evaluate_experiment(cfg: dict[str, Any], name: str, device: torch.device) -> tuple[dict, np.ndarray]:
    ckpt = resolve_path(cfg["paths"]["checkpoints"]) / f"{name}.pth"
    model, meta = load_inference_checkpoint(ckpt, device)
    test = load_split(cfg, "test")
    pred = predict_test_set(model, cfg, device)
    gt = test["landmarks_orig"]
    nme, mae = per_sample_nme(pred, gt), per_sample_mae(pred, gt)
    thr = float(cfg["eval"]["failure_threshold"])
    exp_cfg = cfg["experiments"].get(name, {})
    result = {
        "experiment": name,
        "label": exp_cfg.get("label", name),
        "description": exp_cfg.get("description", ""),
        "checkpoint": _display_path(ckpt),
        "best_epoch": meta.get("epoch"),
        "val_nme": meta.get("val_nme"),
        "parameters": count_parameters(model),
        "overall": summarize(nme, mae, thr),
        "subsets": {a: summarize(nme[test["attributes"][:, j] == 1], mae[test["attributes"][:, j] == 1], thr)
                    for j, a in enumerate(ATTRIBUTES)},
        "latency": measure_latency(model, device, int(cfg["data"]["input_size"]),
                                   int(cfg["eval"]["latency_runs"])),
    }
    worst = max(ATTRIBUTES, key=lambda a: result["subsets"][a]["nme"])
    result["robustness"] = {
        "worst_subset": worst,
        "worst_subset_nme": result["subsets"][worst]["nme"],
        "gap_worst_minus_full": result["subsets"][worst]["nme"] - result["overall"]["nme"],
        "subset_nme_spread": float(np.std([result["subsets"][a]["nme"] for a in ATTRIBUTES])),
    }
    _write_per_sample(cfg, name, test, nme, mae)
    o = result["overall"]
    log.info("%-10s NME %.2f%%  FR@%.0f%% %.2f%%  AUC %.4f  MAE %.2f px  GPU %.2f ms", name,
             o["nme"] * 100, thr * 100, o["failure_rate"] * 100, o["auc"], o["mae_px"],
             result["latency"].get("gpu_ms_median", float("nan")))
    return result, nme


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(resolve_path("").resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_per_sample(cfg, name, test, nme, mae) -> None:
    path = resolve_path(cfg["paths"]["results"]) / f"per_sample_{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "image_name", "nme", "mae_px", *ATTRIBUTES])
        for i in range(len(nme)):
            w.writerow([i, test["image_names"][i], f"{nme[i]:.6f}", f"{mae[i]:.4f}",
                        *test["attributes"][i].tolist()])


def compare(cfg: dict[str, Any], results: dict[str, dict], nmes: dict[str, np.ndarray]) -> dict[str, Any]:
    base, prop = cfg["eval"]["baseline"], cfg["eval"]["proposed"]
    n_boot = int(cfg["eval"]["bootstrap_samples"])
    out: dict[str, Any] = {"baseline": base, "proposed": prop, "vs_baseline": {}}
    if base not in results:
        return out
    b = results[base]
    for name, r in results.items():
        out["vs_baseline"][name] = {
            "nme_improvement_pct": relative_improvement(b["overall"]["nme"], r["overall"]["nme"]),
            "failure_rate_change_pts": (r["overall"]["failure_rate"] - b["overall"]["failure_rate"]) * 100,
            "bootstrap_nme_diff": paired_bootstrap(nmes[base], nmes[name], n_boot, cfg["seed"]),
        }
    if prop in results:
        test = load_split(cfg, "test")
        p = results[prop]
        out["overall"] = {
            "baseline_nme": b["overall"]["nme"], "proposed_nme": p["overall"]["nme"],
            "relative_improvement_pct": relative_improvement(b["overall"]["nme"], p["overall"]["nme"]),
            "bootstrap_nme_diff": paired_bootstrap(nmes[base], nmes[prop], n_boot, cfg["seed"]),
        }
        out["subsets"] = {}
        for j, a in enumerate(ATTRIBUTES):
            mask = test["attributes"][:, j] == 1
            out["subsets"][a] = {
                "baseline_nme": b["subsets"][a]["nme"], "proposed_nme": p["subsets"][a]["nme"],
                "relative_improvement_pct": relative_improvement(b["subsets"][a]["nme"], p["subsets"][a]["nme"]),
                "bootstrap_nme_diff": paired_bootstrap(nmes[base][mask], nmes[prop][mask], n_boot, cfg["seed"]),
            }
    return out


def write_tables(cfg: dict[str, Any], metrics: dict[str, Any]) -> None:
    res_dir = resolve_path(cfg["paths"]["results"])
    vs = metrics["comparison"].get("vs_baseline", {})
    with (res_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "label", "nme_pct", "nme_median_pct", "failure_rate_pct", "auc",
                    "mae_px", "gpu_latency_ms", "cpu_latency_ms", "nme_improvement_vs_baseline_pct",
                    "best_epoch", "parameters"])
        for name, r in metrics["models"].items():
            o, lat = r["overall"], r["latency"]
            w.writerow([name, r["label"], f"{o['nme'] * 100:.3f}", f"{o['nme_median'] * 100:.3f}",
                        f"{o['failure_rate'] * 100:.2f}", f"{o['auc']:.4f}", f"{o['mae_px']:.2f}",
                        f"{lat.get('gpu_ms_median', float('nan')):.2f}", f"{lat['cpu_ms_median']:.2f}",
                        f"{vs.get(name, {}).get('nme_improvement_pct', float('nan')):.2f}",
                        r["best_epoch"], r["parameters"]])
    with (res_dir / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "subset", "faces", "nme_pct", "failure_rate_pct", "auc"])
        for name, r in metrics["models"].items():
            rows = [("full", r["overall"])] + [(a, r["subsets"][a]) for a in ATTRIBUTES]
            for subset, s in rows:
                w.writerow([name, subset, s["n"], f"{s['nme'] * 100:.3f}",
                            f"{s['failure_rate'] * 100:.2f}", f"{s['auc']:.4f}"])


def write_figures(cfg: dict[str, Any], metrics: dict[str, Any], nmes: dict[str, np.ndarray]) -> None:
    res_dir = resolve_path(cfg["paths"]["results"])
    names = list(metrics["models"])
    labels = [metrics["models"][n]["label"] for n in names]
    thr = float(cfg["eval"]["failure_threshold"])
    ov = [metrics["models"][n]["overall"] for n in names]
    plotting.bar_panels(
        res_dir / "nme_comparison.png", names, labels,
        [("NME (%) - lower is better", [o["nme"] * 100 for o in ov], "{:.2f}"),
         (f"Failure rate @{thr * 100:.0f}% (%) - lower is better", [o["failure_rate"] * 100 for o in ov], "{:.2f}"),
         (f"AUC @{thr * 100:.0f}% - higher is better", [o["auc"] for o in ov], "{:.3f}")],
        "WFLW test set (2,500 faces)")
    groups = [g.replace(" ", "\n", 1) for g in ["Full test set"] + [ATTRIBUTE_LABELS[a] for a in ATTRIBUTES]]
    vals = np.array([[metrics["models"][n]["overall"]["nme"] * 100] +
                     [metrics["models"][n]["subsets"][a]["nme"] * 100 for a in ATTRIBUTES] for n in names])
    plotting.grouped_bars(res_dir / "subgroup_analysis.png", groups, names, labels, vals,
                          "NME (%) - lower is better",
                          "NME by WFLW challenge condition (imaging conditions, not demographic groups)")
    curves = []
    for n, lab in zip(names, labels):
        xs, ys = ced_curve(nmes[n], thr)
        curves.append((n, f"{lab} (AUC {metrics['models'][n]['overall']['auc']:.3f})", xs, ys))
    plotting.ced_plot(res_dir / "ced_curve.png", curves, thr)
    hists = []
    for n, lab in zip(names, labels):
        hp = res_dir / f"history_{n}.json"
        if hp.is_file():
            hists.append((n, lab, load_json(hp)["history"]))
    if hists:
        plotting.training_curves(res_dir / "training_loss.png", hists)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate APAC-Face models on WFLW test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        build_cache(cfg)
    except DatasetNotFoundError as e:
        print(e)
        raise SystemExit(1)
    device = get_device(args.device)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"])
    names = args.experiments or list(cfg["experiments"])
    found = [n for n in names if (ckpt_dir / f"{n}.pth").is_file()]
    for n in set(names) - set(found):
        log.warning("Skipping %s: no checkpoint at %s", n, ckpt_dir / f"{n}.pth")
    if not found:
        raise SystemExit("No trained checkpoints found. Train first: python -m src.train --experiment baseline")

    results, nmes = {}, {}
    for n in found:
        results[n], nmes[n] = evaluate_experiment(cfg, n, device)
    metrics = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"name": "WFLW", "split": "test", "faces": int(len(next(iter(nmes.values()))))},
        "protocol": {
            "normalisation": "inter-ocular distance (outer eye corners, points 60 and 72)",
            "failure_threshold": float(cfg["eval"]["failure_threshold"]),
            "crop": "square box around the ground-truth landmark extent x box_scale (standard WFLW protocol)",
            "coordinates": "errors computed in original image pixels",
        },
        "device": describe_device(device),
        "disclaimer": DISCLAIMER,
        "models": results,
    }
    metrics["comparison"] = compare(cfg, results, nmes)
    res_dir = resolve_path(cfg["paths"]["results"])
    save_json(metrics, res_dir / "metrics.json")
    write_tables(cfg, metrics)
    write_figures(cfg, metrics, nmes)
    comp = metrics["comparison"].get("overall")
    if comp:
        bs = comp["bootstrap_nme_diff"]
        log.info("Proposed vs baseline: NME %.3f%% -> %.3f%% (%.2f%% relative), 95%% CI of NME drop "
                 "[%.3f, %.3f] pts, p=%.4f", comp["baseline_nme"] * 100, comp["proposed_nme"] * 100,
                 comp["relative_improvement_pct"], bs["ci_low"] * 100, bs["ci_high"] * 100, bs["p_value"])
    log.info("Wrote metrics.json, comparison.csv, subgroup_metrics.csv and figures to %s", res_dir)


if __name__ == "__main__":
    main()
