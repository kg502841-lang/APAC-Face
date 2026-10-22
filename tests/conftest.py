import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def face_shape() -> np.ndarray:
    """A plausible, left/right-symmetric 98-point face (image-left points have smaller x)."""
    from src.landmarks import MIRROR_PAIRS, NUM_LANDMARKS

    rng = np.random.default_rng(0)
    lm = np.zeros((NUM_LANDMARKS, 2), np.float32)
    for a, b in MIRROR_PAIRS:
        dx, y = rng.uniform(10, 70), rng.uniform(40, 200)
        lm[a] = (112 - dx, y)
        lm[b] = (112 + dx, y)
    paired = {i for p in MIRROR_PAIRS for i in p}
    for i in range(NUM_LANDMARKS):
        if i not in paired:
            lm[i] = (112, rng.uniform(40, 200))
    return lm
