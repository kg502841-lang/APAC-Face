import common
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dataset import ATTRIBUTE_LABELS, ATTRIBUTES

common.hero("Robustness", "Robustness Analysis",
            "Two different questions, kept apart on purpose: how accuracy changes under difficult "
            "imaging conditions, and how consistently each model behaves across APAC demographic groups.")

cfg = common.config()
metrics = common.load_result_json("metrics.json")
apac = common.load_result_json("apac_consistency.json")
base, prop = cfg["eval"]["baseline"], cfg["eval"]["proposed"]

tab1, tab2 = st.tabs(["WFLW challenge conditions (accuracy)", "APAC demographic groups (consistency)"])

with tab1:
    common.banner("<b>These are imaging conditions, not demographic groups.</b> WFLW labels each test face "
                  "for large pose, exaggerated expression, extreme illumination, make-up, occlusion and blur. "
                  "Errors are measured against ground-truth landmarks.", tone="blue")
    if not metrics:
        st.warning("Run `python -m src.evaluate` to populate this section.")
    else:
        models = metrics["models"]
        names = list(models)
        groups = ["full"] + list(ATTRIBUTES)
        labels = ["Full test set"] + [ATTRIBUTE_LABELS[a] for a in ATTRIBUTES]
        fig = go.Figure()
        for n in names:
            vals = [models[n]["overall"]["nme"] * 100] + [models[n]["subsets"][a]["nme"] * 100 for a in ATTRIBUTES]
            counts = [models[n]["overall"]["n"]] + [models[n]["subsets"][a]["n"] for a in ATTRIBUTES]
            fig.add_trace(go.Bar(x=labels, y=vals, name=common.model_label(n), customdata=counts,
                                 marker=dict(color=common.model_color(n, names), line=dict(color="#ffffff", width=2)),
                                 hovertemplate="%{x}<br>NME %{y:.2f}% · %{customdata} faces<extra>"
                                               + common.model_label(n) + "</extra>"))
        fig.update_layout(barmode="group")
        st.plotly_chart(common.style_figure(fig, 420, "NME (%) · lower is better"), width="stretch")

        comp = metrics.get("comparison", {}).get("subsets")
        if comp:
            rows = []
            for a in ATTRIBUTES:
                c = comp[a]
                bs = c["bootstrap_nme_diff"]
                rows.append({
                    "Condition": ATTRIBUTE_LABELS[a],
                    "Faces": models[prop]["subsets"][a]["n"],
                    "Baseline NME (%)": round(c["baseline_nme"] * 100, 2),
                    "Proposed NME (%)": round(c["proposed_nme"] * 100, 2),
                    "Relative improvement (%)": round(c["relative_improvement_pct"], 1),
                    "95% CI of NME drop (pts)": f"{bs['ci_low'] * 100:.2f} to {bs['ci_high'] * 100:.2f}",
                    "p-value": common.fmt_p(bs["p_value"]),
                })
            st.markdown("##### Baseline vs proposed per condition")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        rob = {n: models[n]["robustness"] for n in names}
        st.caption("Worst condition per model: " + "; ".join(
            f"{common.model_label(n)} - {ATTRIBUTE_LABELS[r['worst_subset']]} "
            f"({r['worst_subset_nme'] * 100:.2f}%, +{r['gap_worst_minus_full'] * 100:.2f} pts over full set)"
            for n, r in rob.items()))

with tab2:
    common.banner("<b>Label-free consistency, not ground-truth accuracy.</b> The APAC images have no landmark "
                  "annotations. Group labels come from the dataset's file names (annotator-provided metadata) - "
                  "never inferred from appearance. <i>Flip inconsistency</i>: disagreement between the prediction "
                  "on an image and on its mirror. <i>Crop instability</i>: scatter of predictions under small "
                  "shifts of the face box. Lower is better for both. The augmented and proposed models were trained "
                  "with flips and shifts, so their gains here are expected and do not by themselves prove better "
                  "accuracy on APAC faces.")
    if not apac:
        st.warning("Run the APAC consistency analysis to populate this section:\n\n"
                   "```\npython -m src.apac_eval --source \"<folder or zip of {id}_{Group}.jpg>\"\n```")
    else:
        groups = list(apac["groups"])
        names = apac["models"]
        metric = st.radio("Metric", ["Flip inconsistency", "Crop instability"], horizontal=True)
        key, ci_key = (("flip_consistency", "flip_ci") if metric == "Flip inconsistency"
                       else ("crop_stability", "stability_ci"))
        labels = [apac["groups"][g]["label"] + ("" if apac["groups"][g]["apac"] else " (ref.)") for g in groups]
        fig = go.Figure()
        for n in names:
            vals = [apac["groups"][g]["models"][n][key] * 100 for g in groups]
            lo = [apac["groups"][g]["models"][n][ci_key][0] * 100 for g in groups]
            hi = [apac["groups"][g]["models"][n][ci_key][1] * 100 for g in groups]
            fig.add_trace(go.Bar(
                x=labels, y=vals, name=common.model_label(n),
                marker=dict(color=common.model_color(n, names), line=dict(color="#ffffff", width=2)),
                error_y=dict(type="data", symmetric=False, array=[h - v for h, v in zip(hi, vals)],
                             arrayminus=[v - l for v, l in zip(vals, lo)], color="#52514e", thickness=1, width=3),
                customdata=list(zip(lo, hi)),
                hovertemplate="%{x}<br>%{y:.2f}% (95% CI %{customdata[0]:.2f}-%{customdata[1]:.2f})<extra>"
                              + common.model_label(n) + "</extra>"))
        fig.update_layout(barmode="group")
        st.plotly_chart(common.style_figure(fig, 420, f"{metric} (% of inter-ocular) · lower is better"),
                        width="stretch")

        k = st.columns(len(names))
        for col, n in zip(k, names):
            s = apac["summary"][n]
            with col:
                worst = apac["groups"][s["apac_worst_group"]]["label"] if s["apac_worst_group"] else "-"
                common.stat(f"{common.model_label(n)} · APAC gap",
                            f"{s['apac_gap_flip_consistency'] * 100:.2f} pts",
                            f"flip inconsistency, max-min across APAC groups · worst: {worst}")

        rows = []
        for g in groups:
            e = apac["groups"][g]
            row = {"Group": e["label"], "APAC": "yes" if e["apac"] else "reference", "Images": e["images"],
                   "Face detected (%)": round(e["face_detection_rate"] * 100, 1)}
            for n in names:
                row[f"{common.model_label(n)} flip (%)"] = round(e["models"][n]["flip_consistency"] * 100, 2)
                row[f"{common.model_label(n)} crop (%)"] = round(e["models"][n]["crop_stability"] * 100, 2)
            c = apac.get("comparison", {}).get(g)
            if c:
                row["Flip improvement, proposed vs baseline (%)"] = round(c["flip_improvement_pct"], 1)
                row["p-value"] = common.fmt_p(c["flip_bootstrap"]["p_value"])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Source: {apac['source']} · {sum(apac['groups'][g]['images'] for g in groups):,} images · "
                   f"generated {apac['generated']}. Face detection rate is for the OpenCV Haar detector; when it "
                   "misses, the image is treated as a centred face crop.")
