"""Live (video / webcam) landmark detection with face tracking.

Per frame, instead of running the face detector every time:
  * while a face is tracked, the crop comes from the previous frame's landmarks - the same
    landmark-box convention the model was trained on - so one forward pass per frame suffices;
  * every `redetect_every` frames the Haar detector checks the frame again: it re-seeds the track
    when the face moved away from the tracked box and drops the track after
    `max_missed_detections` consecutive misses (face left the picture);
  * landmarks are smoothed with an exponential moving average to reduce frame-to-frame jitter.

    python -m src.inference --webcam                      # OpenCV window: q quit, m mesh, s snapshot
    python -m src.inference --video clip.mp4 --record out.mp4 --headless
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .augmentations import Box
from .inference import LandmarkPredictor
from .landmarks import draw_landmarks, landmark_box
from .utils import get_logger

log = get_logger("live")


@dataclass
class LiveResult:
    landmarks: np.ndarray | None   # (98, 2) in frame pixels, None while searching
    box: Box | None
    state: str                     # "detected" (new track), "tracking" or "searching"
    model_ms: float
    total_ms: float


class LiveTracker:
    def __init__(self, predictor: LandmarkPredictor, redetect_every: int = 15,
                 max_missed_detections: int = 2, smoothing: float = 0.4):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.predictor = predictor
        self.redetect_every = max(1, int(redetect_every))
        self.max_missed = max(0, int(max_missed_detections))
        self.smoothing = float(smoothing)
        self.reset()

    def reset(self) -> None:
        self.landmarks: np.ndarray | None = None
        self.frames_since_detect = 0
        self.missed = 0

    def _detected_box(self, frame: np.ndarray) -> Box | None:
        faces = self.predictor.detector.detect(frame)
        return self.predictor.detector.to_crop_box(faces[0]) if faces else None

    def process(self, frame_rgb: np.ndarray) -> LiveResult:
        t0 = time.perf_counter()
        seed: Box | None = None
        if self.landmarks is None or self.frames_since_detect >= self.redetect_every:
            self.frames_since_detect = 0
            det = self._detected_box(frame_rgb)
            if det is None:
                self.missed += 1
                if self.landmarks is None or self.missed > self.max_missed:
                    self.landmarks = None
                    return LiveResult(None, None, "searching", 0.0, (time.perf_counter() - t0) * 1000)
            else:
                self.missed = 0
                if self.landmarks is None or _drifted(det, self._track_box()):
                    seed = det

        if seed is not None:  # new track: calibrated detector box plus one refinement pass
            pred = self.predictor.predict(frame_rgb, box=seed, refine_iterations=1)
            lm, state = pred.landmarks, "detected"
        else:                 # tracking: crop from the previous landmarks, single pass
            pred = self.predictor.predict(frame_rgb, box=self._track_box(), refine_iterations=0)
            lm = pred.landmarks
            if self.smoothing > 0:
                lm = self.smoothing * self.landmarks + (1 - self.smoothing) * lm
            state = "tracking"
        self.frames_since_detect += 1

        if not np.isfinite(lm).all() or not _inside(lm, frame_rgb.shape):
            self.reset()
            return LiveResult(None, None, "searching", pred.model_ms, (time.perf_counter() - t0) * 1000)
        self.landmarks = lm.astype(np.float32)
        return LiveResult(self.landmarks, pred.box, state, pred.model_ms, (time.perf_counter() - t0) * 1000)

    def _track_box(self) -> Box:
        return landmark_box(self.landmarks, self.predictor.box_scale)


def _drifted(det: Box, track: Box) -> bool:
    """The detector's face is not the one being tracked (centres far apart or very different size)."""
    dist = np.hypot(det[0] - track[0], det[1] - track[1])
    ratio = det[2] / max(track[2], 1e-6)
    return dist > 0.35 * det[2] or not 0.6 < ratio < 1.6


def _inside(lm: np.ndarray, shape: tuple[int, ...], margin: float = 0.25) -> bool:
    """Most of the face must be inside the frame; otherwise the track has run away."""
    h, w = shape[:2]
    cx, cy = lm.mean(axis=0)
    return -margin * w < cx < (1 + margin) * w and -margin * h < cy < (1 + margin) * h


def draw_hud(image_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    """Status lines in a translucent box, top-left."""
    out = image_rgb.copy()
    scale = max(0.45, image_rgb.shape[1] / 1400)
    thick = 1 if scale < 0.8 else 2
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0] for t in lines]
    line_h = int(max(s[1] for s in sizes) * 1.8)
    box_w = max(s[0] for s in sizes) + 16
    box_h = line_h * len(lines) + 8
    overlay = out.copy()
    cv2.rectangle(overlay, (6, 6), (6 + box_w, 6 + box_h), (15, 27, 61), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    for i, text in enumerate(lines):
        y = 6 + line_h * (i + 1) - line_h // 4
        cv2.putText(out, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def render(frame_rgb: np.ndarray, result: LiveResult, fps: float, model_name: str,
           draw_mesh: bool = True) -> np.ndarray:
    out = draw_landmarks(frame_rgb, result.landmarks, draw_mesh) if result.landmarks is not None else frame_rgb
    status = {"detected": "face found", "tracking": "tracking", "searching": "searching for a face"}[result.state]
    return draw_hud(out, [f"APAC-Face | {model_name}", f"{fps:4.1f} FPS | {status}",
                          f"model {result.model_ms:5.1f} ms | frame {result.total_ms:5.1f} ms"])


def open_capture(source: int | str, width: int, height: int) -> cv2.VideoCapture:
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # DirectShow opens webcams much faster on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        what = f"camera {source}" if isinstance(source, int) else f"video {source}"
        raise RuntimeError(f"Could not open {what}. Check that it exists and no other app is using it.")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class FpsMeter:
    def __init__(self, alpha: float = 0.1):
        self.alpha, self.fps, self._last = alpha, 0.0, None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps


def run(predictor: LandmarkPredictor, source: int | str, live_cfg: dict[str, Any], *,
        mirror: bool | None = None, headless: bool = False, record: str | None = None,
        max_frames: int | None = None, snapshot_dir: Path | None = None,
        on_frame: Callable[[np.ndarray, LiveResult], None] | None = None) -> dict[str, Any]:
    """Process a webcam or video until it ends, `q`/Esc is pressed or max_frames is reached."""
    is_camera = isinstance(source, int)
    mirror = bool(live_cfg.get("mirror", True)) if mirror is None else mirror
    cap = open_capture(source, int(live_cfg.get("width", 640)), int(live_cfg.get("height", 480)))
    tracker = LiveTracker(predictor, live_cfg.get("redetect_every", 15),
                          live_cfg.get("max_missed_detections", 2), live_cfg.get("smoothing", 0.4))
    fps, writer, draw_mesh = FpsMeter(), None, True
    counts = {"frames": 0, "detected": 0, "tracking": 0, "searching": 0}
    model_ms: list[float] = []
    window = "APAC-Face live (q quit, m mesh, s snapshot)"
    try:
        while max_frames is None or counts["frames"] < max_frames:
            ok, bgr = cap.read()
            if not ok:
                if is_camera:
                    raise RuntimeError("The camera stopped delivering frames")
                break
            if mirror:
                bgr = cv2.flip(bgr, 1)
            frame = np.ascontiguousarray(bgr[:, :, ::-1])
            result = tracker.process(frame)
            counts["frames"] += 1
            counts[result.state] += 1
            if result.landmarks is not None:
                model_ms.append(result.model_ms)
            shown = render(frame, result, fps.tick(), predictor.name, draw_mesh)
            if on_frame:
                on_frame(shown, result)
            if record:
                if writer is None:
                    h, w = shown.shape[:2]
                    rate = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), rate if rate > 1 else 20.0, (w, h))
                writer.write(shown[:, :, ::-1])
            if not headless:
                cv2.imshow(window, shown[:, :, ::-1])
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    draw_mesh = not draw_mesh
                if key == ord("s") and snapshot_dir is not None:
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    path = snapshot_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(str(path), shown[:, :, ::-1])
                    log.info("Snapshot saved to %s", path)
    except KeyboardInterrupt:  # Ctrl-C in the terminal is a normal way to stop
        log.info("Stopped with Ctrl-C")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    stats = {**counts, "mean_fps": fps.fps,
             "median_model_ms": float(np.median(model_ms)) if model_ms else None,
             "face_rate": (counts["detected"] + counts["tracking"]) / max(counts["frames"], 1)}
    log.info("Live session: %d frames, face in %.0f%%, %.1f FPS, model %.1f ms (median)", stats["frames"],
             stats["face_rate"] * 100, stats["mean_fps"], stats["median_model_ms"] or 0.0)
    return stats
