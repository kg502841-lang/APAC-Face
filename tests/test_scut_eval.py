import numpy as np

from src.scut_eval import COMMON_68, SCUT_TO_68, gap_bootstrap, nme_common, read_pts, scut_68


def test_reads_binary_pts_and_rejects_empty_files(tmp_path):
    pts = np.random.default_rng(0).uniform(0, 350, (86, 2)).astype("<f4")
    good = tmp_path / "AF1.pts"
    good.write_bytes(np.int32(86).tobytes() + pts.tobytes())
    assert np.allclose(read_pts(good), pts)
    empty = tmp_path / "CM152.pts"
    empty.write_bytes(np.int32(0).tobytes())   # the one empty file in the real release
    assert read_pts(empty) is None


def test_mapping_is_one_to_one_on_30_points():
    assert len(COMMON_68) == 30 and len(set(SCUT_TO_68.values())) == 30
    assert {36, 45}.issubset(COMMON_68)                      # outer eye corners for normalisation
    assert not set(range(17)) & set(COMMON_68)                # jaw left out
    assert not set(range(60, 68)) & set(COMMON_68)            # inner lip left out
    pts = np.arange(172, dtype=np.float32).reshape(86, 2)
    out = scut_68(pts)
    assert np.array_equal(out[36], pts[46]) and np.isnan(out[0]).all()


def test_nme_on_common_points_and_gap():
    gt = scut_68(np.random.default_rng(1).uniform(50, 300, (86, 2)).astype(np.float32))[None].repeat(4, 0)
    pred = gt.copy()
    pred[:, list(COMMON_68)] += [3.0, 4.0]                    # 5 px off on every scored point
    iod = np.linalg.norm(gt[0, 36] - gt[0, 45])
    assert np.allclose(nme_common(pred, gt), 5 / iod)
    g = gap_bootstrap(np.full(50, 0.03), np.full(20, 0.04), 200, 0)
    assert np.isclose(g["gap"], -0.01) and g["ci_low"] <= g["gap"] <= g["ci_high"]
