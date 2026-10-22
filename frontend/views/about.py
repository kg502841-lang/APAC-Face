import common
import streamlit as st

common.hero("Project", "About APAC-Face",
            "Methodology, architecture, data, limitations and next steps.")

common.card("Problem statement", """
<p><b>APAC-Diverse Facial Landmark Detection Model.</b> Existing facial landmark detectors are trained
predominantly on Caucasian faces and lose accuracy on East, South, and Southeast Asian facial morphology.
Build a landmark detection model that performs accurately across these populations and quantify the
improvement over current baselines.</p>""")
st.write("")

left, right = st.columns(2)
with left:
    common.card("Methodology", """
<ol>
<li>Parse WFLW (98 landmarks + 6 attribute flags per face); split validation by source photo.</li>
<li>Cache square face crops around the landmark extent (×1.2) with a 1.5× margin so augmentation
never sees empty borders.</li>
<li>Fine-tune ImageNet MobileNetV2 with a regression head (AdamW, smooth-L1, plateau LR schedule,
early stopping on validation NME, mixed precision).</li>
<li>Three configurations with identical optimiser and budget: <b>baseline</b> (no augmentation),
<b>augmented</b> (ablation) and <b>proposed</b> (augmentation + attribute-balanced sampling).</li>
<li>Evaluate on the untouched WFLW test set: NME, failure rate, AUC, MAE, latency, per-condition
errors and a paired bootstrap test of the improvement.</li>
<li>Measure label-free consistency per APAC group on images whose labels come from dataset metadata.</li>
</ol>""")
with right:
    common.card("Architecture", """
<ul>
<li>Input 224×224×3, ImageNet normalisation.</li>
<li>MobileNetV2 feature extractor (inverted residual blocks, 1280 channels).</li>
<li>Global average pooling → Linear 1280→512 → ReLU → Dropout 0.2 → Linear 512→196.</li>
<li>Output reshaped to 98 × (x, y), normalised to [0, 1]; head bias initialised to the mean face.</li>
<li>Inference: OpenCV Haar face detector, box calibrated on WFLW, then two refinement passes that
re-crop around the previous prediction exactly like the training crop (tuned on WFLW validation:
5.98% → 5.11% NME from detector boxes).</li>
<li>Live mode: after the first detection the face is tracked by cropping around the previous frame's
landmarks; the detector re-checks every 15 frames and landmarks are smoothed between frames.</li>
</ul>""")

st.write("")
left, right = st.columns(2)
with left:
    common.card("Dataset", """
<ul>
<li><b>WFLW</b> (Wu et al., CVPR 2018): 10,000 faces, 98 manual landmarks; 7,500 train / 2,500 test;
pose, expression, illumination, make-up, occlusion and blur flags. Research use only - not redistributed.</li>
<li><b>APAC images</b>: FairFace-style 224×224 face crops labelled by file name (East Asian, Southeast
Asian, Indian, plus White, Black, Latino/Hispanic, Middle Eastern as reference groups). No landmarks.</li>
</ul>""")
with right:
    common.card("Limitations", """
<ul>
<li>WFLW has no ethnicity labels: its results show robustness to imaging conditions, not accuracy on
APAC populations.</li>
<li>The APAC analysis is label-free (flip and crop consistency). It detects instability, not a
consistent bias, so it cannot prove per-population accuracy.</li>
<li>Demographic labels in the APAC set are annotator-perceived categories, coarse by nature.</li>
<li>Direct coordinate regression with global pooling is simpler (and less accurate) than heatmap
models such as HRNet or STAR.</li>
<li>The Haar detector is frontal-only; profile faces may be missed and fall back to a centre crop.</li>
</ul>""")

st.write("")
common.card("Future scope", """
<ul>
<li>Annotate a consented APAC test set (e.g. 50-100 faces per group in the 98-point format) to measure true
per-population NME; the evaluation code accepts WFLW-format annotation files.</li>
<li>Semi-supervised adaptation on unlabelled APAC faces (equivariance / flip-consistency losses).</li>
<li>Heatmap or hybrid heads, Wing / adaptive-wing loss, a stronger detector (e.g. YuNet / RetinaFace).</li>
<li>Multiple training seeds with mean ± std reporting; ONNX export for edge deployment.</li>
</ul>""")
