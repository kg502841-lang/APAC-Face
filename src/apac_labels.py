"""Hand-labelled APAC test set: prepare faces for the labelling tool, then measure real per-group accuracy.

The APAC archive has demographic group labels (from its file names) but no landmarks. This module closes
that gap with human annotation, never with model output:

  prepare   samples faces per group, copies them to web/labeling/images/ and writes
            web/labeling/manifest.json. Each face gets a NEUTRAL starting shape: the mean WFLW training
            face (68 points) placed in a box around the face. Only the face location comes from
            MediaPipe; no model's landmark prediction is shown, so annotators are not nudged toward
            any method being evaluated.
  status    labelling progress per group and annotator.
  evaluate  scores Google MediaPipe and every APAC-Face checkpoint against the human labels on the
            68-point iBUG layout: NME per group, the gap between each APAC group and the reference
            group, paired bootstrap tests, and inter-annotator agreement when faces were labelled twice.

    python -m src.apac_labels prepare --per-group 50
    python -m src.serve_web                 # then open http://127.0.0.1:8080/label.html
    python -m src.apac_labels status
    python -m src.apac_labels evaluate

Labels are saved by the labelling page through src/serve_web.py to data/apac_labels/<annotator>/<id>.json
(git-ignored, like every dataset file).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .apac_eval import ImageSource, sample_by_group
from .inference import decode_image, load_predictors
from .landmarks import EYES_68, landmark_box, to_68
from .metrics import paired_bootstrap
from .utils import get_logger, load_config, load_json, resolve_path, save_json

log = get_logger("apac_labels")

NUM_POINTS = 68
LABELS_DIR = "data/apac_labels"
LABELING_WEB_DIR = "labeling"            # inside paths.web
ANNOTATOR_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
SKIP_REASONS = ("no_face", "multiple_faces", "too_occluded", "too_blurry", "not_a_photo", "other")
# Where a face's points start in the labelling tool. "mean_shape" is neutral; "mediapipe" (its 68 points with the
# labelling-style offset removed) is about twice as fast to correct, and any anchoring it causes favours
# MediaPipe, so a win for APAC-Face measured on such labels is conservative. Every label records its start.
START_SOURCES = ("mean_shape", "mediapipe")
# An average face matches almost no real face, so a label that started there with few points moved was not
# finished (e.g. saved by accident). Such labels are left out of the evaluation and counted.
MIN_MOVED_FROM_MEAN = 50


# ----------------------------------------------------------------------------------------------- prepare

def mean_shape_68(cfg: dict[str, Any]) -> np.ndarray:
    """Mean WFLW training face on the 68-point layout, in units of the training crop box:
    point = (box centre) + side * mean_shape."""
    from .dataset import load_split
    train = load_split(cfg, "train")
    box_scale = float(cfg["data"]["box_scale"])
    shapes = []
    for lm in train["landmarks"]:
        cx, cy, side = landmark_box(lm, box_scale)
        shapes.append((to_68(lm, "wflw") - (cx, cy)) / side)
    return np.mean(shapes, axis=0).astype(np.float32)


def face_box(image: np.ndarray, landmarker: Any, calib: dict[str, float] | None) -> tuple[tuple[float, float, float], str]:
    """Training-style crop box around the face: from MediaPipe's face location when it finds the face,
    otherwise the centre of the image (the APAC archive holds centred face crops)."""
    if landmarker is not None and calib is not None:
        mesh = landmarker.detect(image)
        if mesh is not None:
            from .mediapipe_baseline import mediapipe_seed_box
            return tuple(float(v) for v in mediapipe_seed_box(mesh, calib)), "mediapipe_box"
    h, w = image.shape[:2]
    return ((w - 1) / 2, (h - 1) / 2 + 0.05 * h, 0.62 * min(h, w)), "centre_box"


def prepare(cfg: dict[str, Any], source: str, per_group: int, seed: int, groups: list[str] | None) -> Path:
    src = ImageSource(source)
    by_group = sample_by_group(src.names(), per_group, seed)
    if groups:
        by_group = {g: v for g, v in by_group.items() if g in groups}
    web = resolve_path(cfg["paths"]["web"]) / LABELING_WEB_DIR
    img_dir = web / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    mean = mean_shape_68(cfg)

    landmarker, calib = None, None
    calib_path = resolve_path(cfg["paths"]["checkpoints"]) / "mediapipe_calibration.json"
    try:
        from .mediapipe_baseline import MediaPipeLandmarker
        landmarker = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"])
        calib = load_json(calib_path)
    except Exception as e:  # mediapipe is optional: fall back to centred boxes
        log.warning("MediaPipe unavailable (%s): starting shapes use a centred box", e)

    labels = cfg.get("apac", {}).get("group_labels", {})
    items, located = [], 0
    for g, names in by_group.items():
        for name in names:
            data = src.read(name)
            image = decode_image(data)
            (cx, cy, side), how = face_box(image, landmarker, calib)
            located += how == "mediapipe_box"
            file = Path(name).name
            (img_dir / file).write_bytes(data)
            init = (np.array([cx, cy]) + side * mean).round(2)
            items.append({"id": Path(file).stem, "file": f"images/{file}", "group": g,
                          "group_label": labels.get(g, g.replace("_", " ")),
                          "width": int(image.shape[1]), "height": int(image.shape[0]),
                          "init": init.tolist(), "init_source": f"mean WFLW face in {how}"})
    if landmarker is not None:
        landmarker.close()
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "source": src.path.name, "seed": seed,
        "per_group": per_group, "num_points": NUM_POINTS, "layout": "iBUG 68-point (WFLW 98 -> 68 subset)",
        "reference_group": "White", "skip_reasons": list(SKIP_REASONS),
        "mean_shape": mean.round(5).tolist(), "items": items,
    }
    save_json(manifest, web / "manifest.json")
    log.info("Prepared %d faces (%s); face located by MediaPipe for %d, centred box for %d -> %s",
             len(items), ", ".join(f"{g} {len(v)}" for g, v in by_group.items()), located,
             len(items) - located, web / "manifest.json")
    return web / "manifest.json"


def add_mediapipe_starts(cfg: dict[str, Any]) -> int:
    """Store MediaPipe's offset-corrected 68 points as an optional starting shape for every manifest face
    (`init_mp`, null where MediaPipe finds no face). Faces and ids are unchanged, so existing labels stay valid."""
    from .mediapipe_baseline import MediaPipeLandmarker, apply_offset

    web = resolve_path(cfg["paths"]["web"]) / LABELING_WEB_DIR
    manifest = load_manifest(cfg)
    calib = load_json(resolve_path(cfg["paths"]["checkpoints"]) / "mediapipe_calibration.json")
    offset = np.asarray(calib["offset_68"], np.float32)
    landmarker = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"])
    found = 0
    for it in manifest["items"]:
        mesh = landmarker.detect(decode_image((web / it["file"]).read_bytes()))
        if mesh is None:
            it["init_mp"] = None
            continue
        it["init_mp"] = apply_offset(to_68(mesh, "mediapipe").astype(np.float32)[None], offset)[0].round(2).tolist()
        found += 1
    landmarker.close()
    manifest["start_sources"] = list(START_SOURCES)
    save_json(manifest, web / "manifest.json")
    return found


# ------------------------------------------------------------------------------------------ label files

def validate_label(label: dict[str, Any], manifest_ids: set[str]) -> dict[str, Any]:
    """Check one label posted by the labelling page; returns the cleaned record or raises ValueError."""
    annotator = str(label.get("annotator", ""))
    item_id = str(label.get("id", ""))
    if not ANNOTATOR_RE.match(annotator):
        raise ValueError("annotator must be 1-40 letters, digits, '-' or '_'")
    if not ID_RE.match(item_id) or item_id not in manifest_ids:
        raise ValueError(f"unknown image id {item_id!r}")
    skipped = bool(label.get("skipped", False))
    start = str(label.get("start", "mean_shape"))
    out = {"id": item_id, "annotator": annotator, "skipped": skipped,
           "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds": float(label.get("seconds", 0) or 0),
           "start": start if start in START_SOURCES else "mean_shape"}
    if skipped:
        reason = str(label.get("skip_reason", "other"))
        out["skip_reason"] = reason if reason in SKIP_REASONS else "other"
        return out
    pts = np.asarray(label.get("points"), dtype=np.float64)
    if pts.shape != (NUM_POINTS, 2) or not np.isfinite(pts).all():
        raise ValueError(f"points must be {NUM_POINTS} finite (x, y) pairs")
    occluded = label.get("occluded") or [False] * NUM_POINTS
    if len(occluded) != NUM_POINTS:
        raise ValueError(f"occluded must have {NUM_POINTS} flags")
    out["points"] = pts.round(2).tolist()
    out["occluded"] = [bool(v) for v in occluded]
    out["moved"] = int(label.get("moved", 0) or 0)  # how many points the annotator actually moved
    return out


def save_label(labels_root: Path, record: dict[str, Any]) -> Path:
    path = labels_root / record["annotator"] / f"{record['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(path)
    return path


def load_labels(labels_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """{image id: {annotator: record}}"""
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if labels_root.is_dir():
        for p in sorted(labels_root.glob("*/*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[rec["id"]][rec["annotator"]] = rec
    return out


def load_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(cfg["paths"]["web"]) / LABELING_WEB_DIR / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{path} missing - run: python -m src.apac_labels prepare")
    return load_json(path)


def status(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = load_manifest(cfg)
    labels = load_labels(resolve_path(LABELS_DIR))
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "labelled": 0, "skipped": 0, "double": 0})
    for it in manifest["items"]:
        g = groups[it["group"]]
        g["total"] += 1
        recs = labels.get(it["id"], {})
        done = [r for r in recs.values() if not r["skipped"]]
        g["labelled"] += bool(done)
        g["skipped"] += bool(recs) and not done
        g["double"] += len(done) >= 2
    annotators = sorted({a for recs in labels.values() for a in recs})
    for name, g in sorted(groups.items()):
        log.info("%-18s %3d / %3d labelled, %d skipped, %d labelled twice", name, g["labelled"], g["total"],
                 g["skipped"], g["double"])
    log.info("Annotators: %s", ", ".join(annotators) or "none yet")
    return {"groups": dict(groups), "annotators": annotators}


# ------------------------------------------------------------------------------------------- evaluation

def inter_annotator(labels: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Disagreement between two annotators of the same face (NME of one against the other): the noise
    floor below which differences between models are not meaningful."""
    vals = []
    for recs in labels.values():
        done = [np.asarray(r["points"]) for r in recs.values() if not r["skipped"]]
        if len(done) >= 2:
            a, b = done[0], done[1]
            iod = np.linalg.norm((a[EYES_68[0]] + b[EYES_68[0]]) / 2 - (a[EYES_68[1]] + b[EYES_68[1]]) / 2)
            vals.append(float(np.linalg.norm(a - b, axis=-1).mean() / iod))
    return {"faces": len(vals), "nme": float(np.mean(vals)) if vals else None,
            "nme_median": float(np.median(vals)) if vals else None}


def evaluate(cfg: dict[str, Any], annotator: str | None, device: str = "auto") -> dict[str, Any]:
    from .mediapipe_baseline import MediaPipeLandmarker, apply_offset, mediapipe_seed_box, score

    manifest = load_manifest(cfg)
    labels = load_labels(resolve_path(LABELS_DIR))
    web = resolve_path(cfg["paths"]["web"]) / LABELING_WEB_DIR
    ref_group = manifest.get("reference_group", "White")
    thr = float(cfg["eval"]["failure_threshold"])
    n_boot = int(cfg["eval"]["bootstrap_samples"])
    calib = load_json(resolve_path(cfg["paths"]["checkpoints"]) / "mediapipe_calibration.json")
    offset = np.asarray(calib["offset_68"], np.float32)
    predictors = load_predictors(cfg, device)
    landmarker = MediaPipeLandmarker(cfg["paths"]["mediapipe_model"])

    rows: list[dict[str, Any]] = []
    gts, grp, mp_found, starts, incomplete = [], [], [], [], []
    preds: dict[str, list[np.ndarray]] = defaultdict(list)
    for it in manifest["items"]:
        recs = {a: r for a, r in labels.get(it["id"], {}).items() if not r["skipped"]}
        if not recs:
            continue
        rec = recs.get(annotator) if annotator else recs[sorted(recs)[0]]
        if rec is None:
            continue
        if rec.get("start", "mean_shape") == "mean_shape" and rec.get("moved", 0) < MIN_MOVED_FROM_MEAN:
            incomplete.append(f"{rec['annotator']}/{it['id']} ({rec.get('moved', 0)} points moved)")
            continue
        image = decode_image((web / it["file"]).read_bytes())
        gt = np.asarray(rec["points"], np.float32)
        mesh = landmarker.detect(image)
        nan68 = np.full((NUM_POINTS, 2), np.nan, np.float32)
        if mesh is not None:
            mp68 = to_68(mesh, "mediapipe").astype(np.float32)
            preds["mediapipe"].append(mp68)
            preds["mediapipe_calibrated"].append(apply_offset(mp68[None], offset)[0])
            box = mediapipe_seed_box(mesh, calib)       # exactly what the web app does: one pass
            for m, p in predictors.items():
                preds[m].append(to_68(p.predict_boxes(image, [box])[0], "wflw"))
        else:
            preds["mediapipe"].append(nan68)
            preds["mediapipe_calibrated"].append(nan68)
            for m, p in predictors.items():               # Python-app fallback: Haar box + refinement
                preds[m].append(to_68(p.predict(image).landmarks, "wflw"))
        gts.append(gt)
        grp.append(it["group"])
        mp_found.append(mesh is not None)
        starts.append(rec.get("start", "mean_shape"))
        rows.append({"id": it["id"], "group": it["group"], "annotator": rec["annotator"],
                     "start": starts[-1], "mediapipe_found": mesh is not None})
    landmarker.close()
    if incomplete:
        log.warning("Left out %d unfinished label(s) - redo them in the tool ('Review my saved faces'): %s",
                    len(incomplete), ", ".join(incomplete))
    if not gts:
        raise SystemExit("No finished labels yet - label faces in http://127.0.0.1:8080/label.html")

    gts_a, grp_a, found, start_a = np.stack(gts), np.array(grp), np.array(mp_found), np.array(starts)
    methods = ["mediapipe", "mediapipe_calibrated", *predictors]
    labels_of = {"mediapipe": "Google MediaPipe", "mediapipe_calibrated": "MediaPipe, offset-corrected"}
    labels_of |= {m: cfg["experiments"].get(m, {}).get("label", m) for m in predictors}
    deployed = cfg["inference"].get("web_model", cfg["inference"]["default_model"])

    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "faces": int(len(gts_a)),
        "mediapipe_found": int(found.sum()), "reference_group": ref_group, "deployed_model": deployed,
        "protocol": ("Human 68-point labels (iBUG layout) on APAC images; NME = mean point error / "
                     "outer-eye-corner distance of the label. APAC-Face is located by MediaPipe's face "
                     "detection like the web app; every method is scored on the faces MediaPipe found."),
        "inter_annotator": inter_annotator(labels), "methods": {m: labels_of[m] for m in methods}, "groups": {},
        "excluded_unfinished": incomplete,
    }
    scored: dict[str, dict[str, Any]] = {}
    for g in sorted(set(grp_a)):
        mask = (grp_a == g) & found
        if not mask.any():
            continue
        scored[g] = {m: score(np.stack(preds[m])[mask], gts_a[mask], thr) for m in methods}
        report["groups"][g] = {"faces": int(mask.sum()),
                               "methods": {m: {k: v for k, v in s.items() if not k.startswith("_")}
                                           for m, s in scored[g].items()}}
        for ref in ("mediapipe", "mediapipe_calibrated", "baseline"):
            if ref in scored[g] and deployed in scored[g] and ref != deployed:
                report["groups"][g].setdefault("comparison", {})[f"{deployed}_vs_{ref}"] = paired_bootstrap(
                    scored[g][ref]["_nme"], scored[g][deployed]["_nme"], n_boot, cfg["seed"])
    # The same per-group NMEs split by starting shape: if a MediaPipe start anchored annotators, MediaPipe
    # looks better on those labels than on the mean-shape ones.
    report["by_start"] = {}
    for s in sorted(set(start_a)):
        for g in sorted(set(grp_a)):
            mask = (grp_a == g) & found & (start_a == s)
            if mask.any():
                report["by_start"].setdefault(s, {})[g] = {
                    "faces": int(mask.sum()),
                    "nme": {m: float(np.nanmean(score(np.stack(preds[m])[mask], gts_a[mask], thr)["_nme"]))
                            for m in methods}}
    if ref_group in scored:
        report["gap_vs_reference"] = {
            g: {m: scored[g][m]["nme"] - scored[ref_group][m]["nme"] for m in methods}
            for g in scored if g != ref_group}
    for (i, r) in enumerate(rows):
        for m in methods:
            p = preds[m][i]
            iod = np.linalg.norm(gts_a[i, EYES_68[0]] - gts_a[i, EYES_68[1]])
            r[f"nme_{m}"] = float(np.linalg.norm(p - gts_a[i], axis=-1).mean() / iod) if np.isfinite(p).all() else ""

    res = resolve_path(cfg["paths"]["results"])
    save_json(report, res / "apac_labelled_accuracy.json")
    with (res / "apac_labelled_per_face.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return report


# ------------------------------------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-labelled APAC test set: prepare, status, evaluate")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="sample faces and write web/labeling/manifest.json")
    p.add_argument("--source", default=None, help="folder or .zip of {id}_{Group}.jpg images")
    p.add_argument("--per-group", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026, help="differs from apac_eval's seed on purpose")
    p.add_argument("--groups", nargs="*", default=None, help="limit to these groups (default: all)")
    p.add_argument("--force", action="store_true", help="replace an existing manifest")
    sub.add_parser("status", help="labelling progress")
    sub.add_parser("add-starts", help="add MediaPipe starting shapes to the existing manifest (ids unchanged)")
    e = sub.add_parser("evaluate", help="per-group accuracy against the human labels")
    e.add_argument("--annotator", default=None, help="use this annotator's labels (default: first by name)")
    e.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "prepare":
        web = resolve_path(cfg["paths"]["web"]) / LABELING_WEB_DIR
        if (web / "manifest.json").is_file() and not args.force:
            raise SystemExit(f"{web / 'manifest.json'} exists; labels refer to it. Use --force to replace it.")
        if args.force and (web / "images").is_dir():
            shutil.rmtree(web / "images")
        prepare(cfg, args.source or cfg["paths"]["apac_source"], args.per_group, args.seed, args.groups)
    elif args.cmd == "status":
        status(cfg)
    elif args.cmd == "add-starts":
        n = add_mediapipe_starts(cfg)
        log.info("MediaPipe starting shapes added for %d faces (others keep the mean-shape start)", n)
    else:
        report = evaluate(cfg, args.annotator, args.device)
        ia = report["inter_annotator"]
        log.info("%d labelled faces (%d found by MediaPipe). Inter-annotator NME: %s", report["faces"],
                 report["mediapipe_found"], f"{ia['nme'] * 100:.2f}% on {ia['faces']} faces" if ia["faces"] else "n/a")
        for g, e in report["groups"].items():
            cells = "  ".join(f"{m} {v['nme'] * 100:.2f}%" for m, v in e["methods"].items())
            log.info("%-18s n=%-3d %s", g, e["faces"], cells)
            for ref, b in e.get("comparison", {}).items():
                log.info("    %s: NME drop %.2f pts (95%% CI %.2f to %.2f, p=%.4f)", ref, b["mean_diff"] * 100,
                         b["ci_low"] * 100, b["ci_high"] * 100, b["p_value"])
        for s, groups in report["by_start"].items():
            log.info("labels started from %s: %s", s, "  ".join(
                f"{g} n={v['faces']} deployed {v['nme'][report['deployed_model']] * 100:.2f}% vs MediaPipe "
                f"{v['nme']['mediapipe'] * 100:.2f}%" for g, v in groups.items()))
        log.info("Wrote results/apac_labelled_accuracy.json and apac_labelled_per_face.csv")


if __name__ == "__main__":
    main()
