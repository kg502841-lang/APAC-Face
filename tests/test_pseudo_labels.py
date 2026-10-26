import numpy as np

from src.inference import LandmarkPredictor
from src.model import build_model, save_inference_checkpoint
from src.pseudo_labels import JITTER, copy_photos, teacher_predict


def test_copy_reads_the_source_and_never_changes_it(tmp_path):
    src = tmp_path / "Dataset1"
    src.mkdir()
    (src / "000001_East_Asian.jpg").write_bytes(b"jpeg-1")
    (src / "notes.txt").write_text("not a photo")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in src.iterdir()}
    dest = tmp_path / "project" / "images"
    assert copy_photos(src, dest) == 1
    assert copy_photos(src, dest) == 1                         # re-running is harmless
    assert (dest / "000001_East_Asian.jpg").read_bytes() == b"jpeg-1" and not (dest / "notes.txt").exists()
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in src.iterdir()} == before


def test_teacher_averages_ten_views(tmp_path, face_shape):
    cfg = {"model": {"weights": None, "hidden_dim": 32, "dropout": 0.0},
           "data": {"input_size": 224, "box_scale": 1.2}}
    model = build_model(cfg["model"], pretrained=False)
    model.init_mean_shape(face_shape / 224)
    save_inference_checkpoint(tmp_path / "m.pth", model, cfg, "unit")
    pred = LandmarkPredictor(tmp_path / "m.pth", device="cpu")
    image = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    mean, spread, single = teacher_predict(pred, image, (150.0, 150.0, 200.0))
    assert mean.shape == (98, 2) and single.shape == (98, 2) and np.isfinite(mean).all()
    assert spread >= 0 and len(JITTER) * 2 == 10
