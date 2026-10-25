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
    landmarks: np.ndarray | None   # (98, 2) filtered landmarks in frame pixels, None while searching
    box: Box | None
    state: str                     # "detected" (new track), "tracking" or "searching"
    model_ms: float
    total_ms: float
    raw: np.ndarray | None = None  # the model's unfiltered prediction for this frame


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012) over an array of landmark coordinates.

    A low-pass filter whose cutoff rises with speed: strong smoothing while the face is still
    (kills jitter), little smoothing while it moves (little lag). Speed is measured in face sizes
    per second (`scale` = current face box side), so the settings do not depend on face size.
        cutoff = min_cutoff + beta * |speed|
    """

    def __init__(self, min_cutoff: float = 0.3, beta: float = 20.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float, scale: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x.astype(np.float32)
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True) / max(scale, 1e-6)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.astype(np.float32)


class EmaFilter:
    """Fixed-weight exponential moving average (the earlier live-mode smoothing)."""

    def __init__(self, smoothing: float = 0.4):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None

    def __call__(self, x: np.ndarray, t: float, scale: float) -> np.ndarray:
        x = np.asarray(x, np.float32)
        self.x = x if self.x is None else self.smoothing * self.x + (1 - self.smoothing) * x
        return self.x


class NoFilter:
    def reset(self) -> None:
        pass

    def __call__(self, x: np.ndarray, t: float, scale: float) -> np.ndarray:
        return np.asarray(x, np.float32)


def make_filter(live_cfg: dict[str, Any] | None):
    """`live.filter`: one_euro (default) | ema | none."""
    c = live_cfg or {}
    kind = c.get("filter", "one_euro")
    if kind == "one_euro":
        return OneEuroFilter(c.get("min_cutoff", 0.3), c.get("beta", 20.0), c.get("d_cutoff", 1.0))
    if kind == "ema":
        return EmaFilter(c.get("smoothing", 0.4))
    if kind == "none":
        return NoFilter()
    raise ValueError(f"Unknown live.filter {kind!r} (one_euro | ema | none)")


class LiveTracker:
    """Detect once, then track: each frame is cropped around the previous frame's *raw* prediction
    (the model's own crop convention); the filter only smooths what is returned, so it never feeds
    back into what the model sees."""

    def __init__(self, predictor: LandmarkPredictor, redetect_every: int = 15,
                 max_missed_detections: int = 2, smoothing: float | None = None, landmark_filter=None):
        self.predictor = predictor
        self.redetect_every = max(1, int(redetect_every))
        self.max_missed = max(0, int(max_missed_detections))
        if landmark_filter is None:
            landmark_filter = EmaFilter(smoothing) if smoothing is not None else OneEuroFilter()
        self.filter = landmark_filter
        self.reset()

    def reset(self) -> None:
        self.landmarks: np.ndarray | None = None   # raw prediction of the last frame (drives the crop)
        self.frames_since_detect = 0
        self.missed = 0
        self.filter.reset()

    def _detected_box(self, frame: np.ndarray) -> Box | None:
        faces = self.predictor.detector.detect(frame)
        return self.predictor.detector.to_crop_box(faces[0]) if faces else None

    def process(self, frame_rgb: np.ndarray, timestamp: float | None = None) -> LiveResult:
        """`timestamp` in seconds (defaults to the wall clock; videos pass their frame time)."""
        t0 = time.perf_counter()
        ts = t0 if timestamp is None else float(timestamp)
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
            state = "detected"
            self.filter.reset()
        else:                 # tracking: crop from the previous raw landmarks, single pass
            pred = self.predictor.predict(frame_rgb, box=self._track_box(), refine_iterations=0)
            state = "tracking"
        raw = pred.landmarks
        self.frames_since_detect += 1

        if not np.isfinite(raw).all() or not _inside(raw, frame_rgb.shape):
            self.reset()
            return LiveResult(None, None, "searching", pred.model_ms, (time.perf_counter() - t0) * 1000)
        self.landmarks = raw.astype(np.float32)
        shown = self.filter(self.landmarks, ts, pred.box[2])
        return LiveResult(shown, pred.box, state, pred.model_ms, (time.perf_counter() - t0) * 1000, raw=self.landmarks)

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
                          live_cfg.get("max_missed_detections", 2), landmark_filter=make_filter(live_cfg))
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
