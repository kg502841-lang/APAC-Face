import numpy as np

from src.landmarks import (FLIP_PERMUTATION, MIRROR_PAIRS, NUM_LANDMARKS, TOPOLOGY, draw_landmarks,
                           flip_landmarks, interocular_distance, landmark_box)


def test_flip_permutation_is_an_involution_over_all_points():
    assert sorted(FLIP_PERMUTATION.tolist()) == list(range(NUM_LANDMARKS))
    assert (FLIP_PERMUTATION[FLIP_PERMUTATION] == np.arange(NUM_LANDMARKS)).all()


def test_only_centre_line_points_map_to_themselves():
    fixed = {i for i in range(NUM_LANDMARKS) if FLIP_PERMUTATION[i] == i}
    # chin, nose bridge, nose tip/base centre, lip centres (outer top/bottom, inner top/bottom)
    assert fixed == {16, 51, 52, 53, 54, 57, 79, 85, 90, 94}
    assert len({i for p in MIRROR_PAIRS for i in p}) == 2 * len(MIRROR_PAIRS)


def test_flip_twice_is_identity_and_keeps_left_on_the_left(face_shape):
    flipped = flip_landmarks(face_shape, width=224)
    np.testing.assert_allclose(flip_landmarks(flipped, width=224), face_shape, atol=1e-5)
    for a, b in MIRROR_PAIRS:  # semantic index a is still the image-left point after flipping
        assert flipped[a, 0] < flipped[b, 0]


def test_topology_covers_points_0_to_95():
    covered = sorted(i for idx, _ in TOPOLOGY.values() for i in idx)
    assert covered == list(range(96))


def test_interocular_and_box(face_shape):
    expected = np.linalg.norm(face_shape[60] - face_shape[72])
    assert np.isclose(interocular_distance(face_shape), expected)
    cx, cy, side = landmark_box(face_shape, 1.2)
    span = np.ptp(face_shape, axis=0).max()
    assert np.isclose(side, span * 1.2)
    assert face_shape[:, 0].min() < cx < face_shape[:, 0].max()


def test_draw_landmarks_returns_copy_and_marks_pixels(face_shape):
    img = np.zeros((224, 224, 3), np.uint8)
    out = draw_landmarks(img, face_shape)
    assert out.shape == img.shape and img.sum() == 0 and out.sum() > 0
    # invalid landmarks are ignored instead of crashing
    assert draw_landmarks(img, np.full((98, 2), np.nan)).sum() == 0
