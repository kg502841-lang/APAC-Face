import numpy as np
import pytest

from src.dataset import (FIELDS_PER_LINE, WFLWSample, balanced_sample_weights, parse_annotation_file,
                         parse_annotation_line, split_train_val, wflw_paths, DatasetNotFoundError)
from src.utils import resolve_path


def make_line(name="12--Group/12_Group_1.jpg", attrs=(1, 0, 0, 0, 1, 0)) -> str:
    coords = " ".join(f"{100 + i * 0.5:.3f}" for i in range(196))
    return f"{coords} 10 20 300 330 {' '.join(map(str, attrs))} {name}"


def test_parse_line_fields():
    s = parse_annotation_line(make_line())
    assert len(make_line().split()) == FIELDS_PER_LINE == 207
    assert s.landmarks.shape == (98, 2) and s.landmarks.dtype == np.float32
    assert s.landmarks[0].tolist() == [100.0, 100.5]
    assert s.rect.tolist() == [10, 20, 300, 330]
    assert s.attributes.tolist() == [1, 0, 0, 0, 1, 0]
    assert s.image_name == "12--Group/12_Group_1.jpg"


def test_parse_line_rejects_malformed():
    with pytest.raises(ValueError):
        parse_annotation_line("1 2 3 image.jpg")
    with pytest.raises(ValueError):
        parse_annotation_line(make_line(attrs=(2, 0, 0, 0, 0, 0)))


def test_parse_file_skips_bad_lines(tmp_path):
    f = tmp_path / "ann.txt"
    f.write_text(make_line() + "\n\nbroken line\n" + make_line("a/b.jpg") + "\n", encoding="utf-8")
    samples = parse_annotation_file(f)
    assert [s.image_name for s in samples] == ["12--Group/12_Group_1.jpg", "a/b.jpg"]


def test_split_keeps_faces_of_one_photo_together():
    lm, rect, attrs = np.zeros((98, 2), np.float32), np.zeros(4, np.float32), np.zeros(6, np.int8)
    samples = [WFLWSample(f"img{i // 3}.jpg", lm, rect, attrs) for i in range(300)]
    train, val = split_train_val(samples, 0.1, seed=1)
    assert len(train) + len(val) == 300
    assert not {s.image_name for s in train} & {s.image_name for s in val}
    assert 0.05 < len(val) / 300 < 0.15


def test_balanced_weights_upweight_rare_conditions():
    attrs = np.zeros((100, 6), np.int8)
    attrs[:10, 0] = 1  # 10% large pose
    w = balanced_sample_weights(attrs, [0, 1], power=1.0)
    assert np.isclose(w.mean(), 1.0)
    assert w[0] > w[50]
    assert np.allclose(balanced_sample_weights(attrs, [0], power=0.0), 1.0)


def test_missing_dataset_gives_setup_instructions(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="WFLW_annotations"):
        wflw_paths(tmp_path / "nowhere")


@pytest.mark.skipif(not (resolve_path("data/WFLW/WFLW_annotations")).exists(), reason="WFLW not installed")
def test_real_wflw_annotations_parse():
    paths = wflw_paths("data/WFLW")
    samples = parse_annotation_file(paths.test_list)
    assert len(samples) == 2500
    assert (paths.images / samples[0].image_name).is_file()
