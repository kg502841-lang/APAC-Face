import cv2
import numpy as np
import pytest

from src.inference import LandmarkPredictor
from src.live import LiveTracker, draw_hud, run
from src.model import build_model, save_inference_checkpoint

FACE_RECT = (110, 70, 100, 100)


@pytest.fixture
def predictor(tmp_path, face_shape):
    cfg = {"model": {"weights": None, "hidden_dim": 32, "dropout": 0.0},
           "data": {"input_size": 224, "box_scale": 1.2}}
    model = build_model(cfg["model"], pretrained=False)
    model.init_mean_shape(face_shape / 224)  # an untrained model that outputs a plausible face
    path = tmp_path / "proposed.pth"
    save_inference_checkpoint(path, model, cfg, "proposed")
    return LandmarkPredictor(path, device="cpu")


def frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (240, 320, 3), dtype=np.uint8)


def test_searching_without_a_face(predictor, monkeypatch):
    monkeypatch.setattr(predictor.detector, "detect", lambda img: [])
    r = LiveTracker(predictor).process(frame())
    assert r.state == "searching" and r.landmarks is None


def test_detect_then_track_then_drop(predictor, monkeypatch):
    faces = {"rects": [FACE_RECT]}
    monkeypatch.setattr(predictor.detector, "detect", lambda img: faces["rects"])
    t = LiveTracker(predictor, redetect_every=3, max_missed_detections=1, smoothing=0.0)
    states = [t.process(frame(i)).state for i in range(3)]
    assert states == ["detected", "tracking", "tracking"]
    faces["rects"] = []  # the face leaves the picture
    states = [t.process(frame(i)).state for i in range(4)]
    assert states == ["tracking", "tracking", "tracking", "searching"]
    assert t.landmarks is None


def test_reseeds_when_detector_finds_the_face_elsewhere(predictor, monkeypatch):
    faces = {"rects": [FACE_RECT]}
    monkeypatch.setattr(predictor.detector, "detect", lambda img: faces["rects"])
    t = LiveTracker(predictor, redetect_every=2, smoothing=0.0)
    t.process(frame())
    t.process(frame())
    faces["rects"] = [(10, 10, 60, 60)]
    r = t.process(frame())
    assert r.state == "detected"
    assert r.landmarks[:, 0].mean() < 110


def test_landmarks_are_98_finite_points(predictor, monkeypatch):
    monkeypatch.setattr(predictor.detector, "detect", lambda img: [FACE_RECT])
    t = LiveTracker(predictor, smoothing=0.5)
    for i in range(4):
        r = t.process(frame(i))
    assert r.landmarks.shape == (98, 2) and np.isfinite(r.landmarks).all()


def test_invalid_smoothing():
    with pytest.raises(ValueError):
        LiveTracker(predictor=None, smoothing=1.0)


def test_headless_video_run_records_output(predictor, tmp_path, monkeypatch):
    monkeypatch.setattr(predictor.detector, "detect", lambda img: [FACE_RECT])
    src = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(8):
        w.write(frame(i))
    w.release()
    out = tmp_path / "annotated.mp4"
    stats = run(predictor, str(src), {"mirror": False}, headless=True, record=str(out))
    assert stats["frames"] == 8 and stats["face_rate"] == 1.0
    assert out.is_file() and out.stat().st_size > 0


def test_hud_keeps_frame_shape():
    img = frame()
    assert draw_hud(img, ["a", "b"]).shape == img.shape


def test_one_euro_filter_smooths_jitter_without_much_lag():
    from src.live import OneEuroFilter
    rng = np.random.default_rng(0)
    base = rng.uniform(50, 150, (98, 2))
    f = OneEuroFilter(min_cutoff=1.0, beta=10.0)
    assert np.allclose(f(base, 0.0, 100.0), base)                      # first frame passes through
    still = [f(base + rng.normal(0, 2, base.shape), t / 30, 100.0) for t in range(1, 60)]
    assert np.std(np.stack(still)[20:], axis=0).mean() < 0.6 * 2.0     # >= 40% less jitter (raw std 2 px)
    f.reset()
    moving = [f(base + [t * 5.0, 0], t / 30, 100.0) for t in range(60)]  # 150 px/s = 1.5 faces/s
    assert abs(moving[-1][0, 0] - (base[0, 0] + 59 * 5.0)) < 10        # follows with little lag


def test_make_filter_choices():
    from src.live import EmaFilter, NoFilter, OneEuroFilter, make_filter
    assert isinstance(make_filter({}), OneEuroFilter)
    assert isinstance(make_filter({"filter": "ema"}), EmaFilter)
    assert isinstance(make_filter({"filter": "none"}), NoFilter)
    with pytest.raises(ValueError):
        make_filter({"filter": "kalman"})
