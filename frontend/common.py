"""Shared Streamlit helpers: styling, API client with in-process fallback, results loading, charts."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, load_json, resolve_path  # noqa: E402

API_URL = os.environ.get("APAC_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Chart colours follow the model (validated colour-blind-safe set); chrome uses the brand accents.
MODEL_COLORS = {"baseline": "#2a78d6", "augmented": "#eb6834", "proposed": "#1baf7a"}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0f1b3d", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

CSS = """
<style>
:root { --navy:#0f1b3d; --ink-2:#475069; --blue:#2563eb; --violet:#7c3aed; --card:#ffffff;
        --line:#e5e8f0; --soft:#f4f6fb; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
h1, h2, h3, h4 { color: var(--navy); letter-spacing: -0.01em; }
.apac-hero { padding: 1.6rem 1.8rem; border-radius: 18px; border: 1px solid var(--line);
  background: linear-gradient(120deg, rgba(37,99,235,0.08), rgba(124,58,237,0.08)); margin-bottom: 1.2rem; }
.apac-hero h1 { margin: 0 0 .3rem 0; font-size: 2.0rem; }
.apac-hero p { margin: 0; color: var(--ink-2); font-size: 1.02rem; }
.apac-eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  background: linear-gradient(90deg, var(--blue), var(--violet)); -webkit-background-clip: text;
  background-clip: text; color: transparent; margin-bottom: .35rem; }
.apac-card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1rem 1.15rem; height: 100%; box-shadow: 0 1px 2px rgba(15,27,61,0.04); }
.apac-card h4 { margin: 0 0 .4rem 0; font-size: 1.0rem; }
.apac-card p, .apac-card li { color: var(--ink-2); font-size: .93rem; margin-bottom: .25rem; }
.apac-stat { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: .9rem 1.1rem; }
.apac-stat .label { color: var(--ink-2); font-size: .8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .04em; }
.apac-stat .value { color: var(--navy); font-size: clamp(1.05rem, 1.8vw, 1.6rem); font-weight: 700;
  line-height: 1.25; overflow-wrap: normal; word-break: keep-all; }
.apac-stat .sub { color: var(--ink-2); font-size: .82rem; }
.apac-chip { display: inline-block; padding: .2rem .65rem; margin: .15rem .25rem .15rem 0; border-radius: 999px;
  background: var(--soft); border: 1px solid var(--line); color: var(--navy); font-size: .82rem; font-weight: 500; }
.apac-flow { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; }
.apac-flow .step { padding: .55rem .8rem; border-radius: 10px; border: 1px solid var(--line); background: var(--soft);
  color: var(--navy); font-size: .86rem; font-weight: 600; }
.apac-flow .arrow { color: var(--violet); font-weight: 700; }
.apac-banner { border-left: 4px solid var(--violet); background: rgba(124,58,237,0.06); padding: .7rem 1rem;
  border-radius: 8px; color: var(--navy); font-size: .92rem; margin: .4rem 0 1rem 0; }
.apac-banner.blue { border-left-color: var(--blue); background: rgba(37,99,235,0.06); }
section[data-testid="stSidebar"] { background: #f7f8fc; border-right: 1px solid var(--line); }
</style>
"""


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="apac-hero"><div class="apac-eyebrow">{eyebrow}</div>'
                f"<h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def stat(label: str, value: str, sub: str = "") -> None:
    st.markdown(f'<div class="apac-stat"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="apac-card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


def banner(text: str, tone: str = "violet") -> None:
    st.markdown(f'<div class="apac-banner {"blue" if tone == "blue" else ""}">{text}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- config / results


@st.cache_data(show_spinner=False)
def config() -> dict[str, Any]:
    return load_config(os.environ.get("APAC_CONFIG"))


def results_dir() -> Path:
    return resolve_path(config()["paths"]["results"])


def model_label(name: str) -> str:
    return config()["experiments"].get(name, {}).get("label", name)


def fmt_p(p: float) -> str:
    """Bootstrap p-values of 0 only mean 'below the resolution of the bootstrap'."""
    n = int(config()["eval"]["bootstrap_samples"])
    return f"< {1 / n:.4f}" if p == 0 else f"{p:.4f}"


def p_text(p: float) -> str:
    """'p < 0.0005' or 'p = 0.1890' for use in sentences."""
    v = fmt_p(p)
    return f"p {v}" if v.startswith("<") else f"p = {v}"


def model_color(name: str, order: list[str] | None = None) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    extras = [n for n in (order or [name]) if n not in MODEL_COLORS]
    return EXTRA_COLORS[extras.index(name) % len(EXTRA_COLORS)] if name in extras else EXTRA_COLORS[0]


def load_result_json(name: str) -> dict[str, Any] | None:
    path = results_dir() / name
    return load_json(path) if path.is_file() else None


def load_result_csv(name: str) -> pd.DataFrame | None:
    path = results_dir() / name
    return pd.read_csv(path) if path.is_file() else None


# --------------------------------------------------------------------------- inference client


def api_health() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


@st.cache_resource(show_spinner="Loading models in-process ...")
def local_predictors() -> dict[str, Any]:
    from src.inference import load_predictors
    return load_predictors(config())


def available_models() -> tuple[list[str], str]:
    """Model names and where inference runs ('api' or 'local')."""
    health = api_health()
    if health and health.get("models_loaded"):
        return health["models_loaded"], "api"
    return list(local_predictors()), "local"


def predict(image_bytes: bytes, filename: str, model: str, backend: str) -> dict[str, Any]:
    """Returns the /predict JSON schema whether the API or the in-process model answered."""
    if backend == "api":
        r = requests.post(f"{API_URL}/predict", params={"model": model},
                          files={"file": (filename, image_bytes, _mime(filename))}, timeout=60)
        if not r.ok:
            raise RuntimeError(_api_error(r))
        return r.json()
    from src.inference import decode_image
    predictors = local_predictors()
    if model not in predictors:
        raise RuntimeError(f"Model '{model}' is not available locally")
    return predictors[model].predict(decode_image(image_bytes)).to_dict()


def _mime(filename: str) -> str:
    return "image/png" if filename.lower().endswith(".png") else "image/jpeg"


def _api_error(r: requests.Response) -> str:
    try:
        return f"API error {r.status_code}: {r.json().get('detail')}"
    except ValueError:
        return f"API error {r.status_code}: {r.text[:200]}"


def to_png_bytes(image_rgb: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- charts


def style_figure(fig: go.Figure, height: int = 360, y_title: str = "", x_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor=GRID),
        bargap=0.25, bargroupgap=0.08,
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=AXIS, tickfont=dict(color=INK_2))
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=INK_2))
    return fig


def bar_by_model(names: list[str], values: list[float], title: str, fmt: str, y_title: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[model_label(n) for n in names], y=values,
        marker=dict(color=[model_color(n, names) for n in names], line=dict(color=SURFACE, width=2)),
        text=[fmt.format(v) for v in values], textposition="outside", textfont=dict(color=INK),
        hovertemplate="%{x}<br>" + title + ": %{text}<extra></extra>", cliponaxis=False,
    ))
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)), showlegend=False)
    return style_figure(fig, 320, y_title)
