import numpy as np
import pytest
import torch

from src.model import (LandmarkRegressor, PretrainedWeightsError, build_model,
                       load_inference_checkpoint, save_inference_checkpoint)

CFG = {"model": {"backbone": "mobilenet_v2", "weights": None, "hidden_dim": 64, "dropout": 0.1},
       "data": {"input_size": 224, "box_scale": 1.2}}


def test_forward_shape():
    model = build_model(CFG["model"], pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 98, 2)


def test_mean_shape_initialisation(face_shape):
    model = build_model(CFG["model"], pretrained=False).eval()
    mean = face_shape / 224
    model.init_mean_shape(mean)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))[0].numpy()
    assert np.abs(out - mean).max() < 0.05


def test_bad_pretrained_weights_fail_loudly():
    with pytest.raises(PretrainedWeightsError):
        LandmarkRegressor(weights="NOT_A_REAL_WEIGHT_SET")


def test_checkpoint_roundtrip(tmp_path):
    model = build_model(CFG["model"], pretrained=False).eval()
    path = tmp_path / "m.pth"
    save_inference_checkpoint(path, model, CFG, "unit", extra={"epoch": 3})
    loaded, meta = load_inference_checkpoint(path, torch.device("cpu"))
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x))
    assert meta["experiment"] == "unit" and meta["epoch"] == 3
    assert meta["preprocess"] == {"input_size": 224, "box_scale": 1.2}


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inference_checkpoint(tmp_path / "nope.pth", torch.device("cpu"))


def test_heatmap_head_outputs_coordinates_and_confidence(tmp_path):
    cfg = {"model": {"weights": None, "head": "heatmap", "decoder_dim": 32},
           "data": {"input_size": 224, "box_scale": 1.2}}
    model = build_model(cfg["model"], pretrained=False).eval()
    with torch.no_grad():
        coords, conf = model.forward_with_confidence(torch.randn(2, 3, 224, 224))
    assert tuple(coords.shape) == (2, 98, 2) and tuple(conf.shape) == (2, 98)
    assert coords.min() >= 0 and coords.max() <= 1 and (conf > 0).all() and (conf <= 1).all()
    path = tmp_path / "hm.pth"
    save_inference_checkpoint(path, model, cfg, "unit")
    loaded, meta = load_inference_checkpoint(path, torch.device("cpu"))
    assert meta["model_cfg"]["head"] == "heatmap"
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x))
