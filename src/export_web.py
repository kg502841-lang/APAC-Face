"""Export what the Faceify web app (web/) needs to run the trained model in the browser.

    python -m src.export_web --apac-source "D:/data/APAC asian images.zip"

Committed outputs (no dataset content):
    web/models/apac_face.onnx          the deployed model (inference.web_model) for ONNX Runtime Web
    web/models/model_meta.json         preprocessing, landmark maps, MediaPipe box calibration
    web/data/benchmark_results.json    measured results shown on the page
Git-ignored outputs (dataset content, regenerate locally):
    web/benchmark/wflw/*.jpg + web/data/wflw_benchmark.json   WFLW test faces with REAL ground truth
    web/benchmark/apac/*.jpg + web/data/apac_gallery.json     APAC photos (no landmark ground truth)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .apac_eval import ImageSource, group_of, sample_by_group
from .augmentations import IMAGENET_MEAN, IMAGENET_STD
from .dataset import ATTRIBUTES, build_cache, load_split
from .landmarks import MEDIAPIPE_TO_68, PUPILS, TOPOLOGY, WFLW_TO_68
from .model import load_inference_checkpoint
from .utils import get_logger, load_config, load_json, resolve_path, save_json

log = get_logger("export_web")

MODEL_FILE = "apac_face.onnx"

REGIONS = {  # group key in file names -> region label used by the Faceify gallery
    "East_Asian": ("East Asia", True), "Indian": ("South Asia", True),
    "Southeast_Asian": ("Southeast Asia", True), "Middle_Eastern": ("Middle East", False),
    "Latino_Hispanic": ("Latino / Hispanic", False), "Black": ("Black / African", False),
    "White": ("Caucasian / White", False),
}


def export_onnx(cfg: dict[str, Any], experiment: str, out: Path) -> dict[str, Any]:
    ckpt = resolve_path(cfg["paths"]["checkpoints"]) / f"{experiment}.pth"
    model, meta = load_inference_checkpoint(ckpt, torch.device("cpu"))
    size = int(meta["preprocess"]["input_size"])
    dummy = torch.zeros(1, 3, size, size)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["image"], output_names=["landmarks"], opset_version=17,
                  dynamic_axes={"image": {0: "batch"}, "landmarks": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:  # very old torch without the dynamo switch
        torch.onnx.export(model, dummy, str(out), **kwargs)

    # Verify the exported graph reproduces PyTorch exactly enough to be the same model.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(["landmarks"], {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX export mismatch: max |torch - onnx| = {max_diff:.2e}")
    log.info("ONNX %s: %.1f MB, max |torch - onnx| = %.1e", out.name, out.stat().st_size / 1e6, max_diff)
    return {"experiment": experiment, "input_size": size, "box_scale": float(meta["preprocess"]["box_scale"]),
            "val_nme": meta.get("val_nme"), "epoch": meta.get("epoch"), "onnx_max_abs_diff": max_diff,
            "onnx_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def model_meta(cfg: dict[str, Any], onnx_info: dict[str, Any]) -> dict[str, Any]:
    calib_path = resolve_path(cfg["paths"]["checkpoints"]) / "mediapipe_calibration.json"
    calib = load_json(calib_path) if calib_path.is_file() else None
    if calib is None:
        log.warning("No MediaPipe box calibration yet - run python -m src.mediapipe_baseline first")
    exp = cfg["experiments"][onnx_info["experiment"]]
    return {
        "name": f"APAC-Face {exp.get('label', onnx_info['experiment'])}",
        "model_file": MODEL_FILE, "input_name": "image", "output_name": "landmarks",
        **onnx_info,
        "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
        "num_landmarks": 98,
        # Crops seeded from MediaPipe's face location are not re-cropped: on WFLW validation extra
        # passes raised NME (1 pass 5.89%, 2 passes 5.92%, 4 passes 5.99%).
        "mediapipe_seed_refine_iterations": 0,
        "live": {k: cfg.get("live", {}).get(k) for k in ("smoothing", "redetect_every")},
        "wflw_to_68": list(WFLW_TO_68), "mediapipe_to_68": list(MEDIAPIPE_TO_68),
        "topology": {name: {"points": list(idx), "closed": closed} for name, (idx, closed) in TOPOLOGY.items()},
        "pupils": list(PUPILS),
        "mediapipe_box_calibration": None if calib is None else
        {k: calib[k] for k in ("side_ratio", "dx", "dy")},
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def benchmark_results(cfg: dict[str, Any]) -> dict[str, Any]:
    res = resolve_path(cfg["paths"]["results"])
    out: dict[str, Any] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if (res / "mediapipe_comparison.json").is_file():
        mp = load_json(res / "mediapipe_comparison.json")
        out["mediapipe"] = {k: mp[k] for k in ("protocol", "detection_rate", "faces_detected", "faces_total",
                                               "methods", "subsets", "comparison", "mediapipe_version")}
    if (res / "metrics.json").is_file():
        m = load_json(res / "metrics.json")
        out["wflw98"] = {
            "protocol": m["protocol"], "comparison": m["comparison"].get("overall"),
            "models": {k: {"label": v["label"], "overall": v["overall"], "latency": v["latency"],
                           "subsets": v["subsets"]} for k, v in m["models"].items()},
        }
    if (res / "apac_consistency.json").is_file():
        a = load_json(res / "apac_consistency.json")
        out["apac_consistency"] = {
            "disclaimer": a["disclaimer"], "models": a["models"],
            "groups": {g: {"label": e["label"], "apac": e["apac"], "images": e["images"],
                           "face_detection_rate": e["face_detection_rate"],
                           "models": {m: {"flip": v["flip_consistency"], "crop": v["crop_stability"]}
                                      for m, v in e["models"].items()}}
                       for g, e in a["groups"].items()},
        }
    return out


def export_wflw_samples(cfg: dict[str, Any], web: Path, per_condition: int, seed: int) -> int:
    """Real WFLW test faces (cached crops) with their manual 98-point annotations."""
    test = load_split(cfg, "test")
    rng = np.random.default_rng(seed)
    attrs = test["attributes"]
    chosen: list[int] = []
    pools = [("Typical", np.flatnonzero(attrs.sum(axis=1) == 0))] + [
        (a, np.flatnonzero(attrs[:, j] == 1)) for j, a in enumerate(ATTRIBUTES)]
    for _, pool in pools:
        picks = [int(i) for i in rng.permutation(pool) if int(i) not in chosen][:per_condition]
        chosen.extend(picks)
    img_dir = web / "benchmark" / "wflw"
    img_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_path(cfg["paths"]["cache_dir"])
    samples = []
    for i in sorted(chosen):
        name = f"wflw_test_{i:04d}.jpg"
        shutil.copyfile(cache_dir / test["files"][i], img_dir / name)
        gt = test["landmarks"][i]  # in crop pixels (OpenCV convention)
        flags = [a for j, a in enumerate(ATTRIBUTES) if attrs[i, j] == 1]
        samples.append({
            "id": f"WFLW-{i:04d}", "key": f"wflw_test_{i:04d}", "file": name,
            "imagePath": f"./benchmark/wflw/{name}", "source": "WFLW test",
            "region": "WFLW ground truth", "conditions": flags or ["typical"],
            "name": f"WFLW test face #{i}", "hasGroundTruth": True,
            "gt98": np.round(gt, 2).tolist(), "gt68": np.round(gt[list(WFLW_TO_68)], 2).tolist(),
        })
    save_json({"source": "WFLW test set (manual annotations, research use only)", "count": len(samples),
               "samples": samples}, web / "data" / "wflw_benchmark.json")
    return len(samples)


def export_apac_gallery(src_path: str, web: Path, per_group: int, seed: int) -> int:
    src = ImageSource(src_path)
    picked = sample_by_group(src.names(), per_group, seed)
    img_dir = web / "benchmark" / "apac"
    img_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for group, names in picked.items():
        region, is_apac = REGIONS.get(group, (group.replace("_", " "), False))
        for n in names:
            file = Path(n).name
            (img_dir / file).write_bytes(src.read(n))
            samples.append({"id": f"FAC-{file.split('_')[0]}", "key": Path(file).stem, "file": file,
                            "imagePath": f"./benchmark/apac/{file}", "region": region, "is_apac": is_apac,
                            "group": group_of(file), "name": f"{region} #{file.split('_')[0]}",
                            "source": src.path.name, "hasGroundTruth": False})
    save_json({"source": src.path.name, "count": len(samples), "samples": samples},
              web / "data" / "apac_gallery.json")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the model and data for the Faceify web app")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default=None, help="model to export (default: inference.web_model)")
    parser.add_argument("--apac-source", default=None, help="folder or .zip of {id}_{Group}.jpg photos")
    parser.add_argument("--wflw-per-condition", type=int, default=10)
    parser.add_argument("--apac-per-group", type=int, default=30)
    args = parser.parse_args()
    cfg = load_config(args.config)
    web = resolve_path(cfg["paths"].get("web", "web"))
    experiment = args.experiment or cfg["inference"].get("web_model") or cfg["eval"]["proposed"]

    info = export_onnx(cfg, experiment, web / "models" / MODEL_FILE)
    save_json(model_meta(cfg, info), web / "models" / "model_meta.json")
    save_json({**benchmark_results(cfg), "web_model": experiment}, web / "data" / "benchmark_results.json")
    build_cache(cfg)
    n = export_wflw_samples(cfg, web, args.wflw_per_condition, int(cfg["seed"]))
    log.info("WFLW benchmark gallery: %d faces with real ground truth", n)
    apac_src = args.apac_source or cfg["paths"]["apac_source"]
    try:
        log.info("APAC gallery: %d photos", export_apac_gallery(apac_src, web, args.apac_per_group, int(cfg["seed"])))
    except FileNotFoundError as e:
        log.warning("APAC gallery skipped: %s", str(e).splitlines()[0])
    log.info("Done. Serve the app: python -m http.server 8080 --directory web  (or /web on the API)")


if __name__ == "__main__":
    main()
