"""Landmark accuracy metrics: NME, MAE, failure rate, CED/AUC and paired bootstrap tests.

Conventions (WFLW benchmark):
    NME  = mean over the 98 points of ||pred - gt||_2 / d, with d = inter-ocular distance
           (outer eye corners, points 60 and 72).
    FR@t = fraction of faces whose NME exceeds t (t = 0.10 by default).
    AUC@t = area under the cumulative error distribution on [0, t], normalised to [0, 1].
Invalid predictions (NaN/inf) get NME = NaN: they are excluded from mean errors but counted as
failures, so a model cannot look better by producing garbage.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .landmarks import interocular_distance


def _valid_rows(pred: np.ndarray) -> np.ndarray:
    return np.isfinite(pred).all(axis=(-2, -1))


def per_sample_nme(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    norm = interocular_distance(gt)
    err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nme = err / norm
    nme[~_valid_rows(pred) | ~(norm > 0)] = np.nan
    return nme


def per_sample_mae(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Mean absolute coordinate error in pixels (averaged over x and y of all points)."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    mae = np.abs(pred - gt).mean(axis=(-2, -1))
    mae[~_valid_rows(pred)] = np.nan
    return mae


def failure_rate(nme: np.ndarray, threshold: float) -> float:
    nme = np.asarray(nme, np.float64)
    if nme.size == 0:
        return float("nan")
    failed = ~np.isfinite(nme) | (nme > threshold)
    return float(failed.mean())


def ced_curve(nme: np.ndarray, threshold: float, steps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    nme = np.asarray(nme, np.float64)
    xs = np.linspace(0.0, threshold, steps)
    safe = np.where(np.isfinite(nme), nme, np.inf)
    ys = (safe[None, :] <= xs[:, None]).mean(axis=1) if nme.size else np.zeros_like(xs)
    return xs, ys


def auc(nme: np.ndarray, threshold: float, steps: int = 1000) -> float:
    xs, ys = ced_curve(nme, threshold, steps)
    return float(np.trapezoid(ys, xs) / threshold)


def summarize(nme: np.ndarray, mae: np.ndarray | None, threshold: float) -> dict[str, Any]:
    nme = np.asarray(nme, np.float64)
    valid = nme[np.isfinite(nme)]
    out = {
        "n": int(nme.size),
        "n_invalid": int(nme.size - valid.size),
        "nme": float(valid.mean()) if valid.size else float("nan"),
        "nme_median": float(np.median(valid)) if valid.size else float("nan"),
        "nme_std": float(valid.std()) if valid.size else float("nan"),
        "failure_rate": failure_rate(nme, threshold),
        "auc": auc(nme, threshold),
    }
    if mae is not None:
        m = np.asarray(mae, np.float64)
        m = m[np.isfinite(m)]
        out["mae_px"] = float(m.mean()) if m.size else float("nan")
    return out


def relative_improvement(baseline: float, proposed: float) -> float:
    """((baseline - proposed) / baseline) * 100. Positive = proposed has lower error."""
    if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(proposed):
        return float("nan")
    return float((baseline - proposed) / baseline * 100.0)


def paired_bootstrap(baseline: np.ndarray, proposed: np.ndarray, n_boot: int = 2000,
                     seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """Bootstrap CI for mean(baseline - proposed) over faces both models predicted validly."""
    a, b = np.asarray(baseline, np.float64), np.asarray(proposed, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    diff = a[ok] - b[ok]
    if diff.size == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    # two-sided bootstrap p-value for H0: mean difference = 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p)), "n": int(diff.size)}


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0,
                      ci: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)
