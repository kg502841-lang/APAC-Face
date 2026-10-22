import json

import common
import numpy as np
import pandas as pd
import streamlit as st

from src.inference import decode_image
from src.landmarks import draw_landmarks, interocular_distance

common.hero("Live inference", "Landmark Detection",
            "Upload a face photo. The largest detected face is cropped, the trained model predicts 98 "
            "landmarks, and the crop is refined twice around the previous prediction.")

try:
    models, backend = common.available_models()
except Exception as e:  # noqa: BLE001
    models, backend = [], "local"
    st.error(f"Could not load models: {e}")

if not models:
    st.warning("No trained model found in `checkpoints/`. Train one first:\n\n"
               "```\npython -m src.train --experiment proposed\n```", icon=":material/warning:")
    st.stop()

default = common.config()["inference"]["default_model"]
c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    model = st.selectbox("Model", models, index=models.index(default) if default in models else 0,
                         format_func=common.model_label)
with c2:
    show_mesh = st.toggle("Draw face mesh", value=True)
with c3:
    compare_all = st.toggle("Run every model", value=False, disabled=len(models) < 2,
                            help="Overlay each model's prediction on the same image")
st.caption(f"Inference via {'FastAPI backend' if backend == 'api' else 'in-process model (API offline)'}")

upload = st.file_uploader("Upload a facial image", type=["jpg", "jpeg", "png"])
if upload is None:
    common.banner("Predictions only: uploaded photos have no ground-truth landmarks, so no accuracy or NME "
                  "is shown here. Measured accuracy lives on the Model Comparison page.", tone="blue")
    st.stop()

data = upload.getvalue()
try:
    image = decode_image(data)
except ValueError as e:
    st.error(f"Could not read this image: {e}")
    st.stop()

run_models = models if compare_all else [model]
results = {}
with st.spinner("Detecting face and predicting landmarks ..."):
    for name in run_models:
        try:
            results[name] = common.predict(data, upload.name, name, backend)
        except Exception as e:  # noqa: BLE001
            st.error(f"{common.model_label(name)}: {e}")
if not results:
    st.stop()

main = results.get(model) or next(iter(results.values()))
for w in main.get("warnings", []):
    st.warning(w, icon=":material/info:")

lm = np.array(main["landmarks"], dtype=np.float32)
annotated = draw_landmarks(image, lm, draw_mesh=show_mesh)

k1, k2, k3, k4 = st.columns(4)
with k1:
    common.stat("Predicted landmarks", str(main["num_landmarks"]), common.model_label(main["model"]))
with k2:
    common.stat("Model time", f"{main['model_ms']:.1f} ms", "network forward passes")
with k3:
    common.stat("End-to-end time", f"{main['total_ms']:.1f} ms", "detection + crop + model")
with k4:
    common.stat("Face detector", "Detected" if main["face_detected"] else "No face found",
                f"{main['faces_found']} face(s) in image")

st.write("")
left, right = st.columns(2)
with left:
    st.markdown("**Original**")
    st.image(image, width="stretch")
with right:
    st.markdown(f"**{common.model_label(main['model'])} prediction**")
    st.image(annotated, width="stretch")

stem = upload.name.rsplit(".", 1)[0]
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Download annotated PNG", common.to_png_bytes(annotated),
                       file_name=f"{stem}_{main['model']}_landmarks.png", mime="image/png",
                       width="stretch")
table = pd.DataFrame(lm, columns=["x", "y"]).rename_axis("landmark").reset_index()
with d2:
    st.download_button("Download coordinates (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"{stem}_{main['model']}_landmarks.csv", mime="text/csv",
                       width="stretch")
with d3:
    st.download_button("Download full response (JSON)", json.dumps(main, indent=2).encode(),
                       file_name=f"{stem}_{main['model']}.json", mime="application/json",
                       width="stretch")

if compare_all and len(results) > 1:
    st.markdown("#### Every model on this image")
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.markdown(f"**{common.model_label(name)}** · {res['model_ms']:.1f} ms")
            st.image(draw_landmarks(image, np.array(res["landmarks"], np.float32), draw_mesh=show_mesh),
                     width="stretch")
    names = list(results)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            la, lb = np.array(results[a]["landmarks"]), np.array(results[b]["landmarks"])
            iod = float(interocular_distance((la + lb) / 2))
            rows.append({"Model A": common.model_label(a), "Model B": common.model_label(b),
                         "Disagreement (% of inter-ocular)": round(
                             float(np.linalg.norm(la - lb, axis=-1).mean() / iod * 100), 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Disagreement measures how far two models' predictions are apart. It is not an error against "
               "ground truth - this image has none.")

with st.expander("Landmark coordinates (pixels in the uploaded image)"):
    st.dataframe(table, hide_index=True, width="stretch", height=300)
