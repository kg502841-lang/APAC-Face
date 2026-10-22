import common
import cv2
import numpy as np
import streamlit as st

from src.inference import decode_image
from src.landmarks import draw_landmarks

common.hero("Real time", "Live Camera",
            "98 landmarks on a live webcam feed. After the first detection the face is tracked from frame to "
            "frame using the previous landmarks, and the detector re-checks every few frames.")

cfg = common.config()
live_cfg = cfg.get("live", {})
tab_video, tab_snap = st.tabs(["Live video", "Snapshot (browser camera)"])

with tab_video:
    common.banner("Uses the webcam of the <b>computer running the dashboard</b>, which suits a local demo. "
                  "Frames are processed in memory and never saved. Opening the dashboard from another "
                  "computer? Use the Snapshot tab, which uses that browser's camera.", tone="blue")
    try:
        predictors = common.local_predictors()
    except Exception as e:  # noqa: BLE001
        predictors = {}
        st.error(f"Could not load models: {e}")
    if not predictors:
        st.warning("No trained model found in `checkpoints/`. Train one first:\n\n"
                   "```\npython -m src.train --experiment proposed\n```")
        st.stop()

    names = list(predictors)
    default = cfg["inference"]["default_model"]
    c1, c2, c3, c4 = st.columns([1.3, 0.8, 1, 1])
    with c1:
        model = st.selectbox("Model", names, index=names.index(default) if default in names else 0,
                             format_func=common.model_label, key="live_model")
    with c2:
        camera = st.number_input("Camera index", min_value=0, max_value=9,
                                 value=int(live_cfg.get("camera_index", 0)), step=1)
    with c3:
        mirror = st.toggle("Mirror", value=bool(live_cfg.get("mirror", True)), help="Selfie view")
    with c4:
        mesh = st.toggle("Face mesh", value=True, key="live_mesh")
    smoothing = st.slider("Smoothing (steadier, slightly laggier)", 0.0, 0.9,
                          float(live_cfg.get("smoothing", 0.4)), 0.1)
    run = st.toggle("Start camera", value=False)

    frame_slot = st.empty()
    stats_slot = st.empty()
    if not run:
        frame_slot.info("Camera is off. Switch on **Start camera** to begin; switch it off to release the camera.",
                        icon=":material/videocam_off:")
    else:
        from src.live import FpsMeter, LiveTracker, open_capture, render

        predictor = predictors[model]
        tracker = LiveTracker(predictor, live_cfg.get("redetect_every", 15),
                              live_cfg.get("max_missed_detections", 2), smoothing)
        try:
            cap = open_capture(int(camera), int(live_cfg.get("width", 640)), int(live_cfg.get("height", 480)))
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        fps, frames, with_face = FpsMeter(), 0, 0
        try:
            while True:  # ends when the toggle changes: Streamlit stops this run and `finally` frees the camera
                ok, bgr = cap.read()
                if not ok:
                    st.error("The camera stopped delivering frames.")
                    break
                if mirror:
                    bgr = cv2.flip(bgr, 1)
                frame = np.ascontiguousarray(bgr[:, :, ::-1])
                result = tracker.process(frame)
                frame_slot.image(render(frame, result, fps.tick(), common.model_label(model), mesh),
                                 width="stretch", output_format="JPEG")
                frames += 1
                with_face += result.landmarks is not None
                if frames % 15 == 0:
                    stats_slot.caption(f"{frames:,} frames · face found in {with_face / frames:.0%} · "
                                       f"{fps.fps:.1f} FPS · model {result.model_ms:.1f} ms per frame")
        finally:
            cap.release()

with tab_snap:
    common.banner("Takes a photo with <b>this browser's</b> camera and runs the selected model on it. "
                  "The photo is sent only to your local APAC-Face server.", tone="blue")
    try:
        snap_models, backend = common.available_models()
    except Exception as e:  # noqa: BLE001
        snap_models, backend = [], "local"
        st.error(f"Could not load models: {e}")
    if not snap_models:
        st.warning("No trained model found in `checkpoints/`.")
        st.stop()
    snap_model = st.selectbox("Model", snap_models,
                              index=snap_models.index(cfg["inference"]["default_model"])
                              if cfg["inference"]["default_model"] in snap_models else 0,
                              format_func=common.model_label, key="snap_model")
    # The camera widget asks the browser for camera permission as soon as it renders, so it only
    # appears once the user opts in.
    shot = st.camera_input("Take a photo") if st.toggle("Use this browser's camera", value=False) else None
    if shot is not None:
        data = shot.getvalue()
        try:
            image = decode_image(data)
            with st.spinner("Predicting landmarks ..."):
                res = common.predict(data, "camera.jpg", snap_model, backend)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process the photo: {e}")
            st.stop()
        for w in res.get("warnings", []):
            st.warning(w, icon=":material/info:")
        k1, k2, k3 = st.columns(3)
        with k1:
            common.stat("Predicted landmarks", str(res["num_landmarks"]), common.model_label(res["model"]))
        with k2:
            common.stat("Model time", f"{res['model_ms']:.1f} ms", "network forward passes")
        with k3:
            common.stat("Face detector", "Detected" if res["face_detected"] else "No face found",
                        f"{res['faces_found']} face(s) in photo")
        annotated = draw_landmarks(image, np.array(res["landmarks"], np.float32))
        st.image(annotated, width="stretch")
        st.download_button("Download annotated PNG", common.to_png_bytes(annotated),
                           file_name="camera_landmarks.png", mime="image/png")
