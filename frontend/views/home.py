import common
import streamlit as st

cfg = common.config()
metrics = common.load_result_json("metrics.json")
summary = common.load_result_json("dataset_summary.json")

common.hero("Cross-cutting computer vision",
            "APAC-Face | Facial Landmark Intelligence",
            "A 98-point facial landmark detector trained on WFLW, compared against a baseline, and "
            "analysed for robustness across challenge conditions and APAC demographic groups.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    common.stat("Landmarks", "98", "WFLW topology per face")
with c2:
    faces = sum(v["faces"] for v in summary["counts"].values()) if summary else 10000
    common.stat("Training data", f"{faces:,}", "WFLW faces (7,500 train / 2,500 test)")
with c3:
    common.stat("Backbone", "MobileNetV2", "ImageNet-pretrained, ~3.0M params")
with c4:
    prop = cfg["eval"]["proposed"]
    if metrics and prop in metrics["models"]:
        nme = metrics["models"][prop]["overall"]["nme"] * 100
        common.stat("Proposed test NME", f"{nme:.2f}%", "WFLW test, inter-ocular")
    else:
        common.stat("Proposed test NME", "Pending", "run python -m src.evaluate")

st.write("")
left, right = st.columns([1.15, 1])
with left:
    common.card("Problem statement", """
<p>Existing facial landmark detectors are trained predominantly on Caucasian faces and can lose
accuracy on East, South and Southeast Asian facial morphology. The goal is a landmark detector that
performs accurately across these populations, with the improvement over a baseline quantified.</p>
<p><b>What this project measures honestly:</b> accuracy (NME) on WFLW, which has ground truth;
robustness across WFLW imaging conditions; and label-free consistency per APAC group on images whose
demographic labels come from dataset metadata - never inferred from faces.</p>""")
with right:
    common.card("Dataset", """
<ul>
<li><b>WFLW</b> - 10,000 faces from WIDER Face, 98 manual landmarks, 7,500 train / 2,500 test.</li>
<li>Attribute flags: large pose, expression, illumination, make-up, occlusion, blur.</li>
<li>Validation split (10%) is made by source photo, so no photo appears in two splits.</li>
<li><b>APAC images</b> - FairFace-style crops labelled East Asian, Southeast Asian, Indian (and
reference groups) by file name; no landmark ground truth.</li>
</ul>""")

st.write("")
st.markdown("#### Model architecture")
st.markdown("""
<div class="apac-flow">
  <span class="step">Image upload</span><span class="arrow">→</span>
  <span class="step">Face detection (OpenCV Haar + calibration)</span><span class="arrow">→</span>
  <span class="step">Square crop 224×224</span><span class="arrow">→</span>
  <span class="step">MobileNetV2 features</span><span class="arrow">→</span>
  <span class="step">Global average pool</span><span class="arrow">→</span>
  <span class="step">Linear 1280→512 · ReLU · Dropout</span><span class="arrow">→</span>
  <span class="step">Linear 512→196</span><span class="arrow">→</span>
  <span class="step">98 × (x, y) in [0, 1]</span><span class="arrow">→</span>
  <span class="step">Re-crop & refine</span>
</div>""", unsafe_allow_html=True)

st.write("")
a, b, c = st.columns(3)
with a:
    common.card("Baseline", "<p>ImageNet MobileNetV2 fine-tuned on WFLW. No augmentation, uniform "
                            "sampling. Same optimiser, schedule and epochs as every other model.</p>")
with b:
    common.card("Augmented (ablation)", "<p>Adds landmark-consistent flips (with left/right index "
                                        "remapping), rotation, scale, translation, brightness/contrast, "
                                        "blur, grayscale and synthetic occlusion.</p>")
with c:
    common.card("Proposed", "<p>Augmentation plus attribute-balanced sampling, which up-weights rare "
                            "challenge conditions (large pose, occlusion, ...) during training.</p>")

st.write("")
st.markdown("#### Technology stack")
stack = ["Python 3.13", "PyTorch", "Torchvision", "OpenCV", "NumPy", "Pandas", "Matplotlib", "Plotly",
         "FastAPI", "Uvicorn", "Streamlit", "pytest"]
st.markdown("".join(f'<span class="apac-chip">{s}</span>' for s in stack), unsafe_allow_html=True)
