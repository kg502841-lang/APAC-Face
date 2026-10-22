import numpy as np
import pytest

from src.landmarks import EYES_68, INNER_68, MEDIAPIPE_TO_68, WFLW_TO_68, to_68
from src.mediapipe_baseline import (apply_offset, fit_box_calibration, fit_offset, mediapipe_seed_box, score)
from src.metrics import per_sample_nme


def test_68_point_maps_are_valid():
    assert len(WFLW_TO_68) == len(set(WFLW_TO_68)) == 68
    assert len(MEDIAPIPE_TO_68) == len(set(MEDIAPIPE_TO_68)) == 68
    assert max(WFLW_TO_68) < 98 and max(MEDIAPIPE_TO_68) < 468
    # the 68-point outer eye corners are WFLW's NME normaliser points
    assert (WFLW_TO_68[EYES_68[0]], WFLW_TO_68[EYES_68[1]]) == (60, 72)
    assert INNER_68[0] == 17 and len(INNER_68) == 51


def test_to_68_shapes(face_shape):
    assert to_68(face_shape, "wflw").shape == (68, 2)
    assert to_68(np.zeros((3, 478, 2)), "mediapipe").shape == (3, 68, 2)
    with pytest.raises(KeyError):
        to_68(face_shape, "dlib")


def test_nme_outer_eyes_and_point_subset(face_shape):
    gt = to_68(face_shape, "wflw")[None]
    pred = gt.copy()
    pred[0, :17] += 10.0  # only the jaw is wrong
    full = per_sample_nme(pred, gt, outer_eyes=EYES_68)[0]
    inner = per_sample_nme(pred, gt, outer_eyes=EYES_68, points=INNER_68)[0]
    iod = np.linalg.norm(gt[0, 36] - gt[0, 45])
    assert inner == 0
    np.testing.assert_allclose(full, (17 * np.hypot(10, 10) / 68) / iod, rtol=1e-6)


def _rotate(points, angle, center):
    c, s = np.cos(angle), np.sin(angle)
    return (points - center) @ np.array([[c, s], [-s, c]]) + center


def test_offset_is_learned_in_an_eye_aligned_frame(face_shape):
    rng = np.random.default_rng(0)
    base_mp = to_68(face_shape, "wflw")
    true_offset = rng.normal(0, 0.02, size=(68, 2))
    true_offset[list(EYES_68)] = 0  # eye corners define the frame
    mp, gt = [], []
    for _ in range(40):  # random rotation / scale / translation of the same face
        angle, scale, shift = rng.uniform(-0.6, 0.6), rng.uniform(0.5, 2.0), rng.uniform(-50, 50, 2)
        m = _rotate(base_mp * scale, angle, base_mp.mean(0) * scale) + shift
        mp.append(m)
        gt.append(apply_offset(m[None], true_offset)[0])
    mp, gt = np.stack(mp), np.stack(gt)
    np.testing.assert_allclose(fit_offset(mp, gt), true_offset, atol=1e-6)
    np.testing.assert_allclose(apply_offset(mp, fit_offset(mp, gt)), gt, atol=1e-4)


def test_box_calibration_round_trip(face_shape):
    rng = np.random.default_rng(1)
    mesh = rng.uniform(40, 200, size=(478, 2))
    calib = fit_box_calibration(mesh[None], face_shape[None], box_scale=1.2)
    from src.landmarks import landmark_box
    np.testing.assert_allclose(mediapipe_seed_box(mesh, calib), landmark_box(face_shape, 1.2), rtol=1e-6)


def test_score_reports_failures_and_regions(face_shape):
    gt = np.repeat(to_68(face_shape, "wflw")[None], 4, axis=0)
    pred = gt.copy()
    pred[1] += 30.0           # one clearly failed face
    pred[2] = np.nan          # one missed face
    s = score(pred, gt, thr=0.10)
    assert s["faces"] == 4 and s["failure_rate"] == 0.5
    assert set(s["regions"]) == {"Jawline", "Eyebrows", "Nose", "Eyes", "Mouth"}
    assert s["mean_error_px"] > 0
