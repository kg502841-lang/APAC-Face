import common
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.metrics import ced_curve

common.hero("Measured results", "Model Comparison",
            "Every number on this page is computed by `python -m src.evaluate` on the 2,500-face WFLW "
            "test set, which was never used for training or checkpoint selection.")

metrics = common.load_result_json("metrics.json")
if not metrics:
    st.warning("No evaluation results yet. After training, run:\n\n```\npython -m src.evaluate\n```",
               icon=":material/warning:")
    st.stop()

cfg = common.config()
models = metrics["models"]
names = list(models)
base, prop = cfg["eval"]["baseline"], cfg["eval"]["proposed"]
comp = metrics.get("comparison", {})
thr = metrics["protocol"]["failure_threshold"]

if base in models and prop in models:
    b, p = models[base]["overall"], models[prop]["overall"]
    ov = comp.get("overall", {})
    bs = ov.get("bootstrap_nme_diff", {})
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        common.stat("Baseline NME", f"{b['nme'] * 100:.2f}%", "inter-ocular normalised")
    with k2:
        common.stat("Proposed NME", f"{p['nme'] * 100:.2f}%", "inter-ocular normalised")
    with k3:
        imp = ov.get("relative_improvement_pct", float("nan"))
        ci = (f"95% CI of NME drop {bs['ci_low'] * 100:.2f} to {bs['ci_high'] * 100:.2f} pts"
              if bs else "")
        common.stat("Relative improvement", f"{imp:+.1f}%", ci)
    with k4:
        common.stat(f"Failure rate @{thr * 100:.0f}%", f"{p['failure_rate'] * 100:.2f}%",
                    f"baseline {b['failure_rate'] * 100:.2f}%")
    with k5:
        lat = models[prop]["latency"]
        gpu = lat.get("gpu_ms_median")
        common.stat("Inference latency", f"{gpu:.1f} ms" if gpu else f"{lat['cpu_ms_median']:.1f} ms",
                    ("GPU, " if gpu else "") + f"CPU {lat['cpu_ms_median']:.1f} ms · batch 1")
    if bs:
        verdict = ("significant at the 5% level" if bs["p_value"] < 0.05
                   else "not significant at the 5% level")
        st.caption(f"Paired bootstrap over {bs['n']:,} test faces: mean NME reduction "
                   f"{bs['mean_diff'] * 100:.3f} points, p {common.fmt_p(bs['p_value'])} - {verdict}. "
                   "Each model was trained once (seed 42), so this covers test-set sampling, not seed variance.")

st.write("")
c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(common.bar_by_model(names, [models[n]["overall"]["nme"] * 100 for n in names],
                                        "NME (%) · lower is better", "{:.2f}", "NME (%)"),
                    width="stretch")
with c2:
    st.plotly_chart(common.bar_by_model(names, [models[n]["overall"]["failure_rate"] * 100 for n in names],
                                        f"Failure rate @{thr * 100:.0f}% · lower is better", "{:.2f}", "%"),
                    width="stretch")
with c3:
    st.plotly_chart(common.bar_by_model(names, [models[n]["overall"]["auc"] for n in names],
                                        f"AUC @{thr * 100:.0f}% · higher is better", "{:.3f}", "AUC"),
                    width="stretch")

st.markdown("#### Cumulative error distribution")
fig = go.Figure()
for n in names:
    df = common.load_result_csv(f"per_sample_{n}.csv")
    if df is None:
        continue
    xs, ys = ced_curve(df["nme"].to_numpy(), thr, steps=200)
    fig.add_trace(go.Scatter(x=xs * 100, y=ys * 100, mode="lines", name=common.model_label(n),
                             line=dict(color=common.model_color(n, names), width=2),
                             hovertemplate="NME ≤ %{x:.2f}%: %{y:.1f}% of faces<extra>" + common.model_label(n) + "</extra>"))
fig.update_layout(hovermode="x unified")
st.plotly_chart(common.style_figure(fig, 380, "Faces with error ≤ x (%)", "NME (% of inter-ocular distance)"),
                width="stretch")

st.markdown("#### Training curves")
fig = go.Figure()
for n in names:
    hist = common.load_result_json(f"history_{n}.json")
    if not hist:
        continue
    h = hist["history"]
    fig.add_trace(go.Scatter(x=h["epoch"], y=np.array(h["val_nme"]) * 100, mode="lines", name=common.model_label(n),
                             line=dict(color=common.model_color(n, names), width=2),
                             hovertemplate="epoch %{x}: val NME %{y:.2f}%<extra>" + common.model_label(n) + "</extra>"))
fig.update_layout(hovermode="x unified")
st.plotly_chart(common.style_figure(fig, 340, "Validation NME (%)", "Epoch"), width="stretch")

st.markdown("#### Results table")
vs = comp.get("vs_baseline", {})
rows = []
for n in names:
    o, lat = models[n]["overall"], models[n]["latency"]
    rows.append({
        "Model": common.model_label(n),
        "NME (%)": round(o["nme"] * 100, 3),
        "Median NME (%)": round(o["nme_median"] * 100, 3),
        f"FR@{thr * 100:.0f}% (%)": round(o["failure_rate"] * 100, 2),
        f"AUC@{thr * 100:.0f}%": round(o["auc"], 4),
        "MAE (px)": round(o["mae_px"], 2),
        "Improvement vs baseline (%)": round(vs.get(n, {}).get("nme_improvement_pct", float("nan")), 2),
        "GPU latency (ms)": round(lat.get("gpu_ms_median", float("nan")), 2),
        "CPU latency (ms)": round(lat["cpu_ms_median"], 2),
        "Best epoch": models[n]["best_epoch"],
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"Protocol: {metrics['protocol']['normalisation']}; {metrics['protocol']['crop']}. "
           f"Evaluated {metrics['generated']} on {metrics['device']}.")
