"""WFLW 98-point landmark definition: mirror permutation, drawing topology and geometry helpers.

Index layout (image-left / image-right as seen by the viewer):
    0-32   face contour (16 = chin)
    33-41  eyebrow, image-left   (33-37 upper edge outer->inner, 38-41 lower edge inner->outer)
    42-50  eyebrow, image-right  (42-46 upper edge inner->outer, 47-50 lower edge outer->inner)
    51-54  nose bridge (top->tip)          55-59 nose base (left->right)
    60-67  eye, image-left  (60 outer corner, 64 inner corner)
    68-75  eye, image-right (68 inner corner, 72 outer corner)
    76-87  outer lip (76 / 82 mouth corners)   88-95 inner lip
    96, 97 pupils (image-left, image-right)
"""
from __future__ import annotations

import cv2
import numpy as np

NUM_LANDMARKS = 98
EYE_OUTER_LEFT = 60   # outer corner of the image-left eye
EYE_OUTER_RIGHT = 72  # outer corner of the image-right eye

# (image-left index, image-right index) pairs that swap under a horizontal flip.
MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    *[(i, 32 - i) for i in range(16)],                                   # contour
    (33, 46), (34, 45), (35, 44), (36, 43), (37, 42),                    # eyebrow upper edges
    (38, 50), (39, 49), (40, 48), (41, 47),                              # eyebrow lower edges
    (55, 59), (56, 58),                                                  # nose base
    (60, 72), (61, 71), (62, 70), (63, 69), (64, 68),                    # eyes, upper lids
    (65, 75), (66, 74), (67, 73),                                        # eyes, lower lids
    (76, 82), (77, 81), (78, 80), (83, 87), (84, 86),                    # outer lip
    (88, 92), (89, 91), (93, 95),                                        # inner lip
    (96, 97),                                                            # pupils
)


def _build_flip_permutation() -> np.ndarray:
    perm = np.arange(NUM_LANDMARKS)
    for a, b in MIRROR_PAIRS:
        perm[a], perm[b] = b, a
    return perm


FLIP_PERMUTATION = _build_flip_permutation()

# Polylines used to draw the face mesh: name -> (indices, closed).
TOPOLOGY: dict[str, tuple[tuple[int, ...], bool]] = {
    "contour": (tuple(range(0, 33)), False),
    "eyebrow_left": (tuple(range(33, 42)), True),
    "eyebrow_right": (tuple(range(42, 51)), True),
    "nose_bridge": (tuple(range(51, 55)), False),
    "nose_base": (tuple(range(55, 60)), False),
    "eye_left": (tuple(range(60, 68)), True),
    "eye_right": (tuple(range(68, 76)), True),
    "lip_outer": (tuple(range(76, 88)), True),
    "lip_inner": (tuple(range(88, 96)), True),
}
PUPILS = (96, 97)

REGION_COLORS: dict[str, tuple[int, int, int]] = {  # RGB
    "contour": (56, 189, 248),
    "eyebrow_left": (167, 139, 250),
    "eyebrow_right": (167, 139, 250),
    "nose_bridge": (52, 211, 153),
    "nose_base": (52, 211, 153),
    "eye_left": (96, 165, 250),
    "eye_right": (96, 165, 250),
    "lip_outer": (244, 114, 182),
    "lip_inner": (244, 114, 182),
    "pupils": (250, 204, 21),
}


# --------------------------------------------------------------------------- 68-point common layout
# The iBUG / 300W 68-point layout is the common ground for comparing models with different topologies
# (0-16 jaw, 17-26 brows, 27-35 nose, 36-47 eyes, 48-67 mouth; 36 and 45 are the outer eye corners).

# Standard WFLW-98 -> 68 subset (every other contour point; 3 of 4 lid points per eye).
WFLW_TO_68: tuple[int, ...] = (
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,   # jaw
    33, 34, 35, 36, 37, 42, 43, 44, 45, 46,                          # brows (upper edges)
    51, 52, 53, 54, 55, 56, 57, 58, 59,                              # nose
    60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75,                  # eyes
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,                  # outer lip
    88, 89, 90, 91, 92, 93, 94, 95,                                  # inner lip
)

# MediaPipe face-mesh vertex -> 68-point layout (same map as the Faceify web app's DatasetManager).
MEDIAPIPE_TO_68: tuple[int, ...] = (
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 81, 13, 311, 308, 402, 14, 178,
)
EYES_68 = (36, 45)
INNER_68 = tuple(range(17, 68))  # everything except the jaw line, where annotation conventions differ most
REGIONS_68 = (("Jawline", 0, 17), ("Eyebrows", 17, 27), ("Nose", 27, 36), ("Eyes", 36, 48), ("Mouth", 48, 68))


def to_68(landmarks: np.ndarray, source: str) -> np.ndarray:
    """Reduce (..., 98, 2) WFLW or (..., >=468, 2) MediaPipe landmarks to the 68-point layout."""
    index = {"wflw": WFLW_TO_68, "mediapipe": MEDIAPIPE_TO_68}[source]
    return np.asarray(landmarks)[..., list(index), :]


def flip_landmarks(landmarks: np.ndarray, width: float) -> np.ndarray:
    """Mirror landmarks (..., 98, 2) horizontally inside an image of the given width."""
    out = np.array(landmarks, dtype=np.float32, copy=True)
    out[..., 0] = (width - 1) - out[..., 0]
    return out[..., FLIP_PERMUTATION, :]


def interocular_distance(landmarks: np.ndarray) -> np.ndarray:
    """Outer-eye-corner distance, the standard WFLW NME normaliser."""
    lm = np.asarray(landmarks)
    return np.linalg.norm(lm[..., EYE_OUTER_LEFT, :] - lm[..., EYE_OUTER_RIGHT, :], axis=-1)


def landmark_box(landmarks: np.ndarray, box_scale: float) -> tuple[float, float, float]:
    """Square crop box (cx, cy, side) around the landmark extent - the training crop convention."""
    lm = np.asarray(landmarks, dtype=np.float64)
    x0, y0 = lm.min(axis=0)
    x1, y1 = lm.max(axis=0)
    side = max(x1 - x0, y1 - y0) * box_scale
    return float((x0 + x1) / 2), float((y0 + y1) / 2), float(max(side, 1.0))


def draw_landmarks(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    draw_mesh: bool = True,
    point_radius: int | None = None,
) -> np.ndarray:
    """Return a copy of the image with the 98-point mesh and points drawn on it."""
    canvas = np.ascontiguousarray(image_rgb.copy())
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape != (NUM_LANDMARKS, 2) or not np.isfinite(lm).all():
        return canvas
    face_size = float(np.ptp(lm, axis=0).max())
    radius = point_radius or max(1, int(round(face_size / 110)))
    thickness = max(1, int(round(face_size / 220)))
    shift = 4  # sub-pixel precision for cv2 drawing
    pts = np.round(lm * (1 << shift)).astype(np.int32)

    if draw_mesh:
        for name, (idx, closed) in TOPOLOGY.items():
            cv2.polylines(canvas, [pts[list(idx)]], closed, REGION_COLORS[name], thickness,
                          cv2.LINE_AA, shift)
    for name, (idx, _) in TOPOLOGY.items():
        for i in idx:
            cv2.circle(canvas, tuple(pts[i]), radius << shift, REGION_COLORS[name], -1,
                       cv2.LINE_AA, shift)
    for i in PUPILS:
        cv2.circle(canvas, tuple(pts[i]), (radius + 1) << shift, REGION_COLORS["pupils"], -1,
                   cv2.LINE_AA, shift)
    return canvas
