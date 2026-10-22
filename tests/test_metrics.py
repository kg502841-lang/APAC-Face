import numpy as np

from src.metrics import (auc, failure_rate, paired_bootstrap, per_sample_mae, per_sample_nme,
                         relative_improvement, summarize)


def test_nme_zero_for_perfect_prediction(face_shape):
    gt = face_shape[None]
    assert per_sample_nme(gt, gt)[0] == 0


def test_nme_normalises_by_interocular(face_shape):
    gt = face_shape[None]
    iod = np.linalg.norm(face_shape[60] - face_shape[72])
    pred = gt + np.array([3.0, 4.0], np.float32)  # every point off by 5 px
    np.testing.assert_allclose(per_sample_nme(pred, gt), [5.0 / iod], rtol=1e-6)
    np.testing.assert_allclose(per_sample_mae(pred, gt), [3.5], rtol=1e-6)


def test_invalid_predictions_are_failures_not_hidden(face_shape):
    gt = np.stack([face_shape, face_shape])
    pred = gt.copy()
    pred[1, 5, 0] = np.nan
    nme = per_sample_nme(pred, gt)
    assert nme[0] == 0 and np.isnan(nme[1])
    assert failure_rate(nme, 0.1) == 0.5
    s = summarize(nme, per_sample_mae(pred, gt), 0.1)
    assert s["n_invalid"] == 1 and s["nme"] == 0.0


def test_auc_bounds():
    assert np.isclose(auc(np.zeros(10), 0.1), 1.0, atol=1e-3)
    assert auc(np.full(10, 0.5), 0.1) == 0.0


def test_relative_improvement_matches_definition():
    assert np.isclose(relative_improvement(5.0, 4.0), 20.0)
    assert np.isnan(relative_improvement(0.0, 1.0))


def test_paired_bootstrap_detects_consistent_gain():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.04, 0.08, 500)
    better = base - 0.005
    r = paired_bootstrap(base, better, n_boot=500)
    assert np.isclose(r["mean_diff"], 0.005) and r["ci_low"] > 0 and r["p_value"] < 0.01
    same = paired_bootstrap(base, base, n_boot=200)
    assert same["mean_diff"] == 0
