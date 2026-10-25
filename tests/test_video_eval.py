import numpy as np

from src.landmarks import to_68
from src.video_eval import FPS, MOTION, SIZE, STILL, make_sequence, sequence_metrics, summarise


def test_sequence_ground_truth_follows_the_face(face_shape):
    crop = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    frames, gts = make_sequence(crop, face_shape.astype(np.float32), np.random.default_rng(1))
    assert len(frames) == STILL + MOTION and frames[0].shape == (SIZE, SIZE, 3)
    assert gts.shape == (STILL + MOTION, 98, 2)
    assert np.allclose(gts[0], gts[STILL - 1])               # still part: the face does not move
    assert np.abs(gts[STILL + MOTION // 4] - gts[0]).max() > 5  # moving part: it does
    assert FPS == 30


def test_metrics_score_jitter_accuracy_and_failures(face_shape):
    gt = np.repeat(to_68(face_shape[None].astype(np.float32), "wflw"), STILL + MOTION, axis=0)
    perfect = sequence_metrics(gt.copy(), gt)
    assert perfect["still_jitter"] < 1e-5 and perfect["all_nme"] == 0 and perfect["failures"] == 0
    noisy = gt + np.random.default_rng(0).normal(0, 1.0, gt.shape)
    noisy[-10:] = np.nan                                      # the face is lost for 10 frames
    m = sequence_metrics(noisy, gt)
    assert m["still_jitter"] > 0 and np.isfinite(m["motion_nme"])
    assert np.isclose(m["failures"], 10 / (STILL + MOTION))
    s = summarise([perfect, m, {**m, "all_nme": 0.5}])
    assert np.isclose(s["lost_tracks"], 1 / 3) and "motion_nme_median" in s and "failures_median" not in s
