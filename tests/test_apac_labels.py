import json
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from src.apac_labels import inter_annotator, load_labels, save_label, validate_label
from src.serve_web import ISOLATION_HEADERS, IsolatedHandler

IDS = {"000001_East_Asian", "000002_White"}


def label(**kw):
    pts = np.random.default_rng(0).uniform(50, 400, (68, 2))
    return {"id": "000001_East_Asian", "annotator": "ann_1", "points": pts.tolist(), "occluded": [False] * 68,
            "seconds": 120, "moved": 68, **kw}


def test_validate_accepts_a_full_label_and_a_skip():
    rec = validate_label(label(), IDS)
    assert len(rec["points"]) == 68 and rec["moved"] == 68 and not rec["skipped"]
    skip = validate_label({"id": "000002_White", "annotator": "ann_1", "skipped": True, "skip_reason": "weird"}, IDS)
    assert skip["skipped"] and skip["skip_reason"] == "other" and "points" not in skip


def test_start_shape_is_recorded_and_unknown_starts_fall_back():
    assert validate_label(label(start="mediapipe"), IDS)["start"] == "mediapipe"
    assert validate_label(label(), IDS)["start"] == "mean_shape"
    assert validate_label(label(start="our_model"), IDS)["start"] == "mean_shape"   # never a model-of-ours start


@pytest.mark.parametrize("bad", [
    {"annotator": "../etc"}, {"annotator": ""}, {"id": "../../x"}, {"id": "999_Unknown"},
    {"points": [[1, 2]] * 67}, {"points": [[1, float("nan")]] * 68}, {"occluded": [False] * 3},
])
def test_validate_rejects_bad_labels(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_label(label(**bad), IDS)


def test_save_load_and_inter_annotator(tmp_path):
    a = validate_label(label(), IDS)
    b = validate_label(label(annotator="ann_2"), IDS)
    b["points"] = (np.asarray(b["points"]) + 1.0).tolist()   # second annotator, 1.41 px away everywhere
    save_label(tmp_path, a)
    save_label(tmp_path, b)
    labels = load_labels(tmp_path)
    assert set(labels["000001_East_Asian"]) == {"ann_1", "ann_2"}
    ia = inter_annotator(labels)
    iod = np.linalg.norm(np.asarray(a["points"])[36] - np.asarray(a["points"])[45])
    assert ia["faces"] == 1 and np.isclose(ia["nme"], np.sqrt(2) / iod, rtol=1e-3)


@pytest.fixture
def server(tmp_path):
    web = tmp_path / "web"
    (web / "labeling").mkdir(parents=True)
    (web / "labeling" / "manifest.json").write_text(json.dumps({"items": [{"id": i} for i in IDS]}))
    (web / "index.html").write_text("<html></html>")

    class Handler(IsolatedHandler):
        labels_root = tmp_path / "data" / "apac_labels"
        manifest_path = web / "labeling" / "manifest.json"

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(web)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()


def test_server_isolation_headers_and_label_round_trip(server):
    url, root = server
    with urllib.request.urlopen(f"{url}/index.html") as r:
        assert all(r.headers[k] == v for k, v in ISOLATION_HEADERS.items())
    req = urllib.request.Request(f"{url}/api/labels", data=json.dumps(label()).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        assert json.load(r)["record"]["annotator"] == "ann_1"
    assert (root / "data" / "apac_labels" / "ann_1" / "000001_East_Asian.json").is_file()
    with urllib.request.urlopen(f"{url}/api/labels?annotator=ann_1") as r:
        d = json.load(r)
    assert "000001_East_Asian" in d["labels"] and d["labelled_by"]["000001_East_Asian"] == ["ann_1"]
    bad = urllib.request.Request(f"{url}/api/labels", data=json.dumps(label(id="nope")).encode(), method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(bad)
    assert e.value.code == 400
