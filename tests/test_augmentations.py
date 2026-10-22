import numpy as np

from src.augmentations import (AugmentationParams, LandmarkAugmenter, apply_affine, crop_matrix,
                               invert_affine, to_tensor)
from src.landmarks import MIRROR_PAIRS


def geometric_only(**overrides) -> AugmentationParams:
    p = AugmentationParams(brightness=0, contrast=0, blur_prob=0, grayscale_prob=0, occlusion_prob=0)
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def test_crop_matrix_maps_box_to_output_frame():
    m = crop_matrix((150.0, 80.0, 100.0), 224)
    corners = np.array([[100.0, 30.0], [200.0, 130.0], [150.0, 80.0]])
    out = apply_affine(corners, m)
    np.testing.assert_allclose(out, [[-0.5, -0.5], [223.5, 223.5], [111.5, 111.5]], atol=1e-4)
    np.testing.assert_allclose(apply_affine(out, invert_affine(m)), corners, atol=1e-3)


def test_geometric_augmentation_moves_pixels_and_landmarks_together():
    """Paint a bright dot under a few landmarks; after augmentation the dots must sit under the
    transformed landmarks - including flips, whose indices are remapped."""
    rng = np.random.default_rng(3)
    lm = np.zeros((98, 2), np.float32)
    lm[:] = (168, 168)
    picks = {0: (100, 120), 32: (236, 120), 60: (130, 140), 72: (206, 140), 54: (168, 190), 16: (168, 250)}
    for i, xy in picks.items():
        lm[i] = xy
    img = np.zeros((336, 336, 3), np.uint8)
    for x, y in picks.values():
        img[y - 2:y + 3, x - 2:x + 3] = 255
    aug = LandmarkAugmenter(geometric_only(flip_prob=0.5), 224, enabled=True)
    box = (167.5, 167.5, 224.0)
    for _ in range(20):
        out_img, out_lm = aug(img, lm, box, rng)
        for i in picks:
            x, y = np.round(out_lm[i]).astype(int)
            if 2 <= x < 222 and 2 <= y < 222:
                assert out_img[y - 1:y + 2, x - 1:x + 2].max() > 100, f"landmark {i} lost its pixel"


def test_flip_remaps_left_and_right(face_shape):
    aug = LandmarkAugmenter(geometric_only(flip_prob=1.0, rotation_deg=0, scale_range=0, translate=0),
                            224, enabled=True)
    img = np.zeros((224, 224, 3), np.uint8)
    _, out = aug(img, face_shape, (111.5, 111.5, 224.0), np.random.default_rng(0))
    for a, b in MIRROR_PAIRS:
        assert out[a, 0] < out[b, 0]
    np.testing.assert_allclose(out[16, 0], 223 - face_shape[16, 0], atol=1e-3)


def test_disabled_augmenter_is_a_deterministic_crop(face_shape):
    aug = LandmarkAugmenter(AugmentationParams(), 224, enabled=False)
    img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    a_img, a_lm = aug(img, face_shape, (111.5, 111.5, 224.0), np.random.default_rng(1))
    b_img, b_lm = aug(img, face_shape, (111.5, 111.5, 224.0), np.random.default_rng(2))
    np.testing.assert_array_equal(a_img, b_img)
    np.testing.assert_allclose(a_lm, face_shape, atol=1e-3)
    np.testing.assert_allclose(b_lm, a_lm)


def test_photometric_never_moves_landmarks(face_shape):
    aug = LandmarkAugmenter(AugmentationParams(flip_prob=0, rotation_deg=0, scale_range=0, translate=0,
                                               occlusion_prob=1.0, blur_prob=1.0), 224, enabled=True)
    _, lm = aug(np.full((224, 224, 3), 128, np.uint8), face_shape, (111.5, 111.5, 224.0),
                np.random.default_rng(0))
    np.testing.assert_allclose(lm, face_shape, atol=1e-3)


def test_to_tensor_layout():
    t = to_tensor(np.zeros((224, 224, 3), np.uint8))
    assert tuple(t.shape) == (3, 224, 224)
    assert t.dtype.is_floating_point
