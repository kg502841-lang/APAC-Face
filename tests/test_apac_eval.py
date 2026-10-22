import zipfile

import cv2
import numpy as np
import pytest

from src.apac_eval import ImageSource, group_of, parse_annotated_set, sample_by_group


def test_group_comes_from_file_name_only():
    assert group_of("train/000016_Southeast_Asian.jpg") == "Southeast_Asian"
    assert group_of("000001_Indian.JPG") == "Indian"
    assert group_of("holiday_photo.jpg") is None
    assert group_of("000002_Black.txt") is None


def test_sampling_is_per_group_and_reproducible():
    names = [f"train/{i:06d}_{g}.jpg" for i, g in enumerate(["East_Asian", "Indian", "White"] * 50)]
    a = sample_by_group(names, per_group=10, seed=3)
    b = sample_by_group(names, per_group=10, seed=3)
    assert a == b and set(a) == {"East_Asian", "Indian", "White"}
    assert all(len(v) == 10 and all(group_of(n) == g for n in v) for g, v in a.items())


def test_zip_source_reads_without_extracting(tmp_path):
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    zpath = tmp_path / "faces.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("train/000000_East_Asian.jpg", buf.tobytes())
        z.writestr("train/readme.txt", "x")
    src = ImageSource(zpath)
    assert src.names() == ["train/000000_East_Asian.jpg"]
    assert src.read("train/000000_East_Asian.jpg") == buf.tobytes()


def test_missing_source_explains_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="Group"):
        ImageSource(tmp_path / "missing.zip")


def test_annotated_set_accepts_short_and_full_wflw_lines(tmp_path):
    coords = " ".join(["1.5"] * 196)
    f = tmp_path / "ann.txt"
    f.write_text(f"{coords} a/000001_Indian.jpg\n{coords} 0 0 9 9 0 0 0 0 0 0 b/000002_East_Asian.jpg\n",
                 encoding="utf-8")
    rows = parse_annotated_set(f)
    assert [r[0] for r in rows] == ["a/000001_Indian.jpg", "b/000002_East_Asian.jpg"]
    assert rows[0][1].shape == (98, 2)
    f.write_text("1 2 3 x.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_annotated_set(f)
