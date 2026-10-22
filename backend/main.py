"""FastAPI inference service for APAC-Face.

    uvicorn backend.main:app --host 127.0.0.1 --port 8000

Endpoints
    GET  /health    service status, loaded models, missing checkpoints
    GET  /models    metadata of every loaded model
    POST /predict   98 landmarks for the largest face in an uploaded JPG/PNG
    POST /compare   run several models on the same upload
    GET  /metrics   measured evaluation results (results/metrics.json, apac_consistency.json)

Uploaded images have no ground truth, so the API returns predictions only - never an accuracy.
Environment: APAC_CONFIG (config path), APAC_DEVICE (auto|cuda|cpu).
"""
from __future__ import annotations

import base64
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.inference import (LandmarkPredictor, annotate, available_checkpoints, decode_image,
                           encode_png, load_predictors)
from src.landmarks import interocular_distance
from src.utils import get_logger, load_config, load_json, resolve_path

log = get_logger("api")

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
NO_GT_NOTE = "Uploaded images have no ground-truth landmarks, so no accuracy metric is computed."


class State:
    cfg: dict[str, Any] = {}
    predictors: dict[str, LandmarkPredictor] = {}
    load_error: str | None = None
    lock = threading.Lock()  # one forward pass at a time on the shared GPU


state = State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.cfg = load_config(os.environ.get("APAC_CONFIG"))
    try:
        state.predictors = load_predictors(state.cfg, os.environ.get("APAC_DEVICE", "auto"))
        state.load_error = None
        log.info("Loaded models: %s", list(state.predictors) or "none")
    except Exception as e:  # noqa: BLE001 - surfaced through /health
        log.exception("Model loading failed")
        state.predictors, state.load_error = {}, str(e)
    yield
    state.predictors.clear()


app = FastAPI(title="APAC-Face API", version=__version__, lifespan=lifespan,
              description="98-point facial landmark detection (WFLW topology).")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _default_model() -> str | None:
    preferred = state.cfg.get("inference", {}).get("default_model")
    if preferred in state.predictors:
        return preferred
    return next(iter(state.predictors), None)


def _predictor(name: str | None) -> LandmarkPredictor:
    if not state.predictors:
        raise HTTPException(503, "No trained model is loaded. Train one with "
                                 "`python -m src.train --experiment proposed`, then restart the API."
                            + (f" Load error: {state.load_error}" if state.load_error else ""))
    name = name or _default_model()
    if name not in state.predictors:
        raise HTTPException(404, f"Unknown model '{name}'. Available: {sorted(state.predictors)}")
    return state.predictors[name]


def _read_upload(file: UploadFile) -> np.ndarray:
    ext = Path(file.filename or "").suffix.lower()
    if (file.content_type or "").lower() not in ALLOWED_TYPES and ext not in ALLOWED_EXT:
        raise HTTPException(415, "Unsupported file type - upload a JPG or PNG image")
    limit = int(state.cfg.get("inference", {}).get("max_upload_mb", 10)) * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image larger than {limit // (1024 * 1024)} MB")
    try:
        return decode_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _run(predictor: LandmarkPredictor, image: np.ndarray, return_image: bool, draw_mesh: bool) -> dict:
    with state.lock:
        pred = predictor.predict(image)
    out = pred.to_dict()
    if return_image:
        out["annotated_png_base64"] = base64.b64encode(
            encode_png(annotate(image, pred, draw_mesh=draw_mesh))).decode("ascii")
    return out


@app.get("/health")
def health() -> dict[str, Any]:
    names = list(state.cfg.get("experiments", {}))
    found = available_checkpoints(state.cfg) if state.cfg else {}
    first = next(iter(state.predictors.values()), None)
    return {
        "status": "ok" if state.predictors else "degraded",
        "version": __version__,
        "models_loaded": list(state.predictors),
        "default_model": _default_model(),
        "missing_checkpoints": [n for n in names if n not in found],
        "device": str(first.device) if first else None,
        "face_detector": ("opencv-haar" if first and first.detector.available else "unavailable"),
        "detector_calibration": first.detector.calibration.get("source") if first else None,
        "load_error": state.load_error,
    }


@app.get("/models")
def models() -> dict[str, Any]:
    out = {}
    for name, p in state.predictors.items():
        exp = state.cfg["experiments"].get(name, {})
        out[name] = {"label": exp.get("label", name), "description": exp.get("description", ""),
                     "best_epoch": p.meta.get("epoch"), "val_nme": p.meta.get("val_nme"),
                     "input_size": p.size, "created": p.meta.get("created")}
    return out


@app.post("/predict")
def predict(file: UploadFile = File(...), model: str | None = Query(None),
            return_image: bool = Query(False), draw_mesh: bool = Query(True)) -> dict[str, Any]:
    image = _read_upload(file)
    result = _run(_predictor(model), image, return_image, draw_mesh)
    result["note"] = NO_GT_NOTE
    return result


@app.post("/compare")
def compare(file: UploadFile = File(...), models: list[str] | None = Query(None),
            return_image: bool = Query(False)) -> dict[str, Any]:
    image = _read_upload(file)
    names = models or list(state.predictors)
    preds = {n: _run(_predictor(n), image, return_image, True) for n in names}
    disagreement = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            la, lb = np.array(preds[a]["landmarks"]), np.array(preds[b]["landmarks"])
            iod = float(interocular_distance((la + lb) / 2))
            disagreement[f"{a}|{b}"] = float(np.linalg.norm(la - lb, axis=-1).mean() / iod) if iod > 0 else None
    return {
        "predictions": preds,
        "inter_model_disagreement": disagreement,
        "note": NO_GT_NOTE + " Disagreement is the mean landmark distance between two models / "
                             "inter-ocular distance; it is not an error against ground truth.",
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    res = resolve_path(state.cfg.get("paths", {}).get("results", "results"))
    wflw, apac = res / "metrics.json", res / "apac_consistency.json"
    if not wflw.is_file():
        raise HTTPException(404, "No evaluation results yet - run `python -m src.evaluate`")
    return {"wflw": load_json(wflw), "apac": load_json(apac) if apac.is_file() else None}
