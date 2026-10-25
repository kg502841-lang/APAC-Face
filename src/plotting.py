"""Static report figures (matplotlib) shared by evaluate.py and apac_eval.py.

Colour follows the model, never its rank: baseline = blue, augmented = orange, proposed = aqua
(validated colour-blind-safe as a set). Text uses ink colours, never series colours.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {"baseline": SERIES[0], "augmented": SERIES[1], "proposed": SERIES[2], "robust": SERIES[3],
                "heatmap": SERIES[4]}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def model_color(name: str, order: Sequence[str]) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    extra = [n for n in order if n not in MODEL_COLORS]
    return SERIES[(5 + extra.index(name)) % len(SERIES)]


def _setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _grid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def bar_panels(path: Path, models: list[str], labels: list[str],
               panels: list[tuple[str, list[float], str]], title: str) -> Path:
    """One small bar chart per metric (no dual axes). panels = (title, values, value format)."""
    _setup()
    many = len(models) > 4
    fig, axes = plt.subplots(1, len(panels), figsize=(max(3.6, 0.95 * len(models)) * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    colors = [model_color(m, models) for m in models]
    for ax, (ptitle, values, fmt) in zip(axes, panels):
        x = np.arange(len(models))
        vals = np.array(values, dtype=float)
        ax.bar(x, vals, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        top = np.nanmax(vals) if np.isfinite(vals).any() else 1
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + top * 0.02, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(x, labels, rotation=0, fontsize=8.5 if many else 10)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(ptitle, loc="left")
        _grid(ax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return _save(fig, path)


def grouped_bars(path: Path, groups: list[str], models: list[str], labels: list[str],
                 values: np.ndarray, ylabel: str, title: str,
                 errors: np.ndarray | None = None) -> Path:
    """values[m, g] for model m and group g. errors: optional (2, M, G) lower/upper deltas."""
    _setup()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(groups) + 2), 4.2))
    m_count = len(models)
    width = 0.8 / m_count
    x = np.arange(len(groups))
    for i, (m, lab) in enumerate(zip(models, labels)):
        pos = x - 0.4 + width * (i + 0.5)
        yerr = errors[:, i, :] if errors is not None else None
        ax.bar(pos, values[i], width=width, color=model_color(m, models), edgecolor=SURFACE,
               linewidth=2, label=lab, yerr=yerr, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncols=m_count, bbox_to_anchor=(0, 1.0))
    top = np.nanmax(values + (errors[1] if errors is not None else 0))
    ax.set_ylim(0, top * 1.2)
    _grid(ax)
    return _save(fig, path)


def ced_plot(path: Path, curves: list[tuple[str, str, np.ndarray, np.ndarray]], threshold: float) -> Path:
    """curves = (model, legend label, x, y)."""
    _setup()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    order = [c[0] for c in curves]
    for name, label, xs, ys in curves:
        ax.plot(xs * 100, ys * 100, color=model_color(name, order), linewidth=2, label=label)
    ax.set_xlim(0, threshold * 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("NME (% of inter-ocular distance)")
    ax.set_ylabel("Faces with error at or below x (%)")
    ax.set_title("Cumulative error distribution - WFLW test", loc="left")
    ax.legend(loc="lower right")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return _save(fig, path)


def canonical_val_nme(hist: dict) -> np.ndarray:
    """Validation NME on annotated (canonical) crops - comparable across experiments, even for runs
    that select checkpoints on detector-seeded validation crops."""
    canon = np.array(hist.get("val_nme_canonical") or [], dtype=float)
    if canon.size == len(hist["val_nme"]) and np.isfinite(canon).all():
        return canon
    return np.array(hist["val_nme"], dtype=float)


def training_curves(path: Path, histories: list[tuple[str, str, dict]]) -> Path:
    """Two separate panels: losses (train dashed, val solid) and validation NME per epoch."""
    _setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    order = [h[0] for h in histories]
    for name, label, hist in histories:
        c = model_color(name, order)
        ax1.plot(hist["epoch"], hist["train_loss"], color=c, linewidth=1.5, linestyle="--")
        ax1.plot(hist["epoch"], hist["val_loss"], color=c, linewidth=2, label=label)
        curve = canonical_val_nme(hist)
        ax2.plot(hist["epoch"], curve * 100, color=c, linewidth=2, label=label)
        best = int(np.nanargmin(hist["val_nme"]))  # the selection metric decides the saved epoch
        ax2.scatter([hist["epoch"][best]], [curve[best] * 100], s=36, color=c,
                    edgecolor=SURFACE, linewidth=2, zorder=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training loss (dashed) and validation loss (solid)", loc="left")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation NME, annotated crop (%)")
    ax2.set_title("Validation NME - dot marks the saved checkpoint", loc="left")
    for ax in (ax1, ax2):
        ax.legend(loc="upper right")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return _save(fig, path)
