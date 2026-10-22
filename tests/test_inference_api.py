import io

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src.inference import LandmarkPredictor, decode_image
from src.model import build_model, save_inference_checkpoint
from src.utils import load_config


def png_bytes(h=240, w=200) -> bytes:
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_decode_image_validation():
    assert decode_image(png_bytes()).shape == (240, 200, 3)
    for bad in (b"", b"not an image", png_bytes(10, 10)):
        with pytest.raises(ValueError):
            decode_image(bad)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    cfg = {"model": {"weights": None, "hidden_dim": 32, "dropout": 0.0},
           "data": {"input_size": 224, "box_scale": 1.2}}
    path = tmp_path / "checkpoints" / "proposed.pth"
    save_inference_checkpoint(path, build_model(cfg["model"], pretrained=False), cfg, "proposed")
    return path


def test_predictor_returns_98_finite_points_without_a_face(tiny_checkpoint):
    p = LandmarkPredictor(tiny_checkpoint, device="cpu")
    pred = p.predict(decode_image(png_bytes()))
    assert pred.landmarks.shape == (98, 2) and np.isfinite(pred.landmarks).all()
    assert not pred.face_detected and pred.warnings
    d = pred.to_dict()
    assert d["num_landmarks"] == 98 and len(d["landmarks"]) == 98


def _client(tmp_path, monkeypatch, with_model: bool, tiny_checkpoint=None) -> TestClient:
    cfg = load_config()
    ckpt_dir = tiny_checkpoint.parent if with_model else tmp_path / "empty_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["checkpoints"] = str(ckpt_dir)
    cfg["paths"]["results"] = str(tmp_path / "results")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("APAC_CONFIG", str(cfg_path))
    monkeypatch.setenv("APAC_DEVICE", "cpu")
    from backend.main import app
    return TestClient(app)


def test_api_without_models_degrades_gracefully(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, with_model=False) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded" and health["models_loaded"] == []
        assert set(health["missing_checkpoints"]) == set(load_config()["experiments"])
        r = client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 503
        assert client.get("/metrics").status_code == 404


def test_api_predict_and_validation(tmp_path, monkeypatch, tiny_checkpoint):
    with _client(tmp_path, monkeypatch, with_model=True, tiny_checkpoint=tiny_checkpoint) as client:
        assert client.get("/health").json()["models_loaded"] == ["proposed"]
        r = client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")},
                        params={"return_image": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["num_landmarks"] == 98 and body["model"] == "proposed"
        assert "annotated_png_base64" in body and "accuracy" not in body
        r = client.post("/predict", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
        assert r.status_code == 415
        r = client.post("/predict", files={"file": ("x.png", b"garbage", "image/png")})
        assert r.status_code == 400
        r = client.post("/predict", params={"model": "nope"},
                        files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 404
        r = client.post("/compare", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200 and set(r.json()["predictions"]) == {"proposed"}
