/**
 * Faceify AI Facial Landmark Analysis - APAC-Face Detector (trained model)
 * ========================================================================
 * Runs the trained APAC-Face network (MobileNetV2 regressor, 98 WFLW landmarks) in the browser with
 * ONNX Runtime Web. Nothing leaves the device.
 *
 * Pipeline (identical to the Python inference in src/inference.py):
 *   1. face location: MediaPipe's mesh extent, mapped to the model's training crop with the
 *      calibration fitted on WFLW training faces (model_meta.json -> mediapipe_box_calibration);
 *   2. crop the square box to 224x224 (pixel-centre convention, black outside the image),
 *      normalise with ImageNet statistics, run the network;
 *   3. optionally re-crop around the prediction (`mediapipe_seed_refine_iterations`, 0 by default:
 *      MediaPipe's location is already close and extra passes raised NME on WFLW validation).
 * On video the face is tracked: each frame is cropped around the previous frame's landmarks and
 * MediaPipe re-seeds the track every `redetect_every` frames or when the face moves away.
 */

const ORT_VERSION = "1.20.1";

const WFLW_REGIONS = [
  ["Jawline", 0, 33], ["Right Eyebrow", 33, 42], ["Left Eyebrow", 42, 51], ["Nose", 51, 60],
  ["Right Eye", 60, 68], ["Left Eye", 68, 76], ["Mouth", 76, 96], ["Pupils", 96, 98]
];

/**
 * One Euro filter (Casiez et al., CHI 2012), identical to OneEuroFilter in src/live.py: the cutoff
 * rises with speed, so still faces are smoothed strongly (no jitter) and moving faces lightly (little
 * lag). Speed is measured in face sizes per second, so settings do not depend on face size.
 */
class OneEuroFilter {
  constructor({ min_cutoff = 0.3, beta = 20.0, d_cutoff = 1.0, filter = "one_euro" } = {}) {
    this.minCutoff = min_cutoff;
    this.beta = beta;
    this.dCutoff = d_cutoff;
    this.enabled = filter === "one_euro";
    this.reset();
  }

  reset() {
    this.x = null;
    this.dx = null;
    this.t = null;
  }

  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  apply(pts, t, scale) {
    if (!this.enabled) return pts;
    if (!this.x || this.t === null || t <= this.t) {
      this.x = pts.map(p => ({ x: p.x, y: p.y }));
      this.dx = pts.map(() => ({ x: 0, y: 0 }));
      this.t = t;
      return pts;
    }
    const dt = t - this.t;
    const aD = OneEuroFilter.alpha(this.dCutoff, dt);
    this.x = pts.map((p, i) => {
      const prev = this.x[i];
      const d = this.dx[i];
      d.x = aD * (p.x - prev.x) / dt + (1 - aD) * d.x;
      d.y = aD * (p.y - prev.y) / dt + (1 - aD) * d.y;
      const speed = Math.hypot(d.x, d.y) / Math.max(scale, 1e-6);
      const a = OneEuroFilter.alpha(this.minCutoff + this.beta * speed, dt);
      return { x: a * p.x + (1 - a) * prev.x, y: a * p.y + (1 - a) * prev.y };
    });
    this.t = t;
    return this.x;
  }
}

class APACDetector extends LandmarkDetector {
  constructor(options = {}) {
    super("APAC-Face (Trained, 98-Point)", options);
    this.pointCount = 98;
    this.isConnected = false;
    this.isSimulation = false;
    this.session = null;
    this.meta = null;
    this.connections = [];
    this.track = null;            // { landmarks, frames } while following a face on video
    this.lastFrameLandmarks = null;
    this.loadError = null;
    this.canvas = document.createElement("canvas");
    this.ctx = null;
  }

  async initialize() {
    try {
      const resp = await fetch("./models/model_meta.json");
      if (!resp.ok) throw new Error("models/model_meta.json missing - run python -m src.export_web");
      this.meta = await resp.json();
      if (typeof ort === "undefined") throw new Error("ONNX Runtime Web failed to load from the CDN");
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      this.session = await ort.InferenceSession.create(`./models/${this.meta.model_file}`, {
        executionProviders: ["wasm"], graphOptimizationLevel: "all"
      });
      const size = this.meta.input_size;
      this.canvas.width = size;
      this.canvas.height = size;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
      this.connections = this._buildConnections();
      this.filter = new OneEuroFilter(this.meta.live || {});
      this.name = `APAC-Face (Trained, ${this.meta.num_landmarks}-Point)`;
      this.isReady = this.isConnected = true;
      console.log("[APACDetector] Trained model loaded:", this.meta.model_file);
    } catch (err) {
      this.loadError = `APAC-Face model failed to load: ${err.message || err}`;
      console.error("[APACDetector]", err);
    }
    return this.isReady;
  }

  _buildConnections() {
    const conns = [];
    for (const { points, closed } of Object.values(this.meta.topology)) {
      for (let i = 0; i < points.length - 1; i++) conns.push([points[i], points[i + 1]]);
      if (closed) conns.push([points[points.length - 1], points[0]]);
    }
    return conns;
  }

  /** True when the next video frame needs a MediaPipe face location to (re)start the track. */
  needsSeed() {
    return !this.track || this.track.frames >= (this.meta.live.redetect_every || 15);
  }

  resetTracking() {
    this.track = null;
    this.lastFrameLandmarks = null;
    if (this.filter) this.filter.reset();
  }

  // ------------------------------------------------------------------ geometry

  _landmarkBox(pts) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of pts) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
    const side = Math.max(x1 - x0, y1 - y0) * this.meta.box_scale;
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, side: Math.max(side, 1) };
  }

  _boxFromMediaPipe(mpPts) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of mpPts) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
    const size = Math.max(x1 - x0, y1 - y0);
    const cal = this.meta.mediapipe_box_calibration || { side_ratio: 1.09, dx: 0, dy: 0.08 };
    return { cx: (x0 + x1) / 2 + cal.dx * size, cy: (y0 + y1) / 2 + cal.dy * size, side: cal.side_ratio * size };
  }

  _drifted(seed, track) {
    const dist = Math.hypot(seed.cx - track.cx, seed.cy - track.cy);
    const ratio = seed.side / Math.max(track.side, 1e-6);
    return dist > 0.35 * seed.side || ratio < 0.6 || ratio > 1.6;
  }

  // ------------------------------------------------------------------ inference

  /** Crop `box` to the network input, run the model, return 98 points in image pixels. */
  async _predictBox(input, box) {
    const S = this.meta.input_size;
    const s = S / box.side;
    const tx = (S - 1) / 2 - s * box.cx;   // OpenCV convention: u = s * x + t (pixel centres)
    const ty = (S - 1) / 2 - s * box.cy;
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, S, S);
    // Canvas works in continuous coordinates (pixel i spans [i, i+1]): shift by half a pixel.
    ctx.setTransform(s, 0, 0, s, tx + 0.5 - 0.5 * s, ty + 0.5 - 0.5 * s);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "low";
    ctx.drawImage(input, 0, 0);
    const rgba = ctx.getImageData(0, 0, S, S).data;
    const n = S * S;
    const [m0, m1, m2] = this.meta.mean;
    const [d0, d1, d2] = this.meta.std;
    const chw = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      chw[i] = (rgba[4 * i] / 255 - m0) / d0;
      chw[n + i] = (rgba[4 * i + 1] / 255 - m1) / d1;
      chw[2 * n + i] = (rgba[4 * i + 2] / 255 - m2) / d2;
    }
    const feeds = { [this.meta.input_name]: new ort.Tensor("float32", chw, [1, 3, S, S]) };
    const out = (await this.session.run(feeds))[this.meta.output_name].data;
    const pts = new Array(this.meta.num_landmarks);
    for (let k = 0; k < pts.length; k++) {
      pts[k] = { x: (out[2 * k] * S - tx) / s, y: (out[2 * k + 1] * S - ty) / s };
    }
    return pts;
  }

  /**
   * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputElement
   * @param {{mediapipeLandmarks?: Array<{x:number,y:number}>}} options  MediaPipe points for the face location
   */
  async detect(inputElement, options = {}) {
    const startTime = performance.now();
    const base = { modelName: this.name, connections: this.connections, isSimulation: false };
    if (!this.session) {
      return { ...base, success: false, faceDetected: false, landmarks: [], pointCount: 0, executionTimeMs: 0,
               stability: "Unavailable", error: this.loadError || "Model not loaded" };
    }

    const w = inputElement.videoWidth || inputElement.naturalWidth || inputElement.width;
    const h = inputElement.videoHeight || inputElement.naturalHeight || inputElement.height;
    const isVideo = inputElement instanceof HTMLVideoElement;
    const seedPts = options.mediapipeLandmarks && options.mediapipeLandmarks.length ? options.mediapipeLandmarks : null;
    if (!isVideo) this.resetTracking();

    let box, passes, state;
    const seedBox = seedPts ? this._boxFromMediaPipe(seedPts) : null;
    const checked = Boolean(options.seedChecked || seedPts);  // MediaPipe looked at this frame
    if (isVideo && this.track) {
      const trackBox = this._landmarkBox(this.track.landmarks);
      if (seedBox && this._drifted(seedBox, trackBox)) {      // a different face / big jump: re-seed
        box = seedBox;
        passes = 1;
        state = "detected";
      } else {
        if (checked) {
          this.track.frames = 0;
          this.track.misses = seedBox ? 0 : this.track.misses + 1;
          if (this.track.misses > 2) {                        // face gone for 3 checks in a row
            this.resetTracking();
            return this._noFace(base, startTime);
          }
        }
        box = trackBox;
        passes = 1;
        state = "tracking";
      }
    } else if (seedBox) {
      box = seedBox;
      passes = 1 + (this.meta.mediapipe_seed_refine_iterations || 0);
      state = "detected";
    } else if (!isVideo) {
      // No face location: like the Python app, assume a centred face and refine twice
      // (on WFLW validation two refinements took a rough box from 20.6% to 9.4% NME).
      box = { cx: (w - 1) / 2, cy: (h - 1) / 2, side: Math.min(w, h) * 0.9 };
      passes = 3;
      state = "fallback";
    } else {
      this.resetTracking();
      return this._noFace(base, startTime);
    }

    let pts = null;
    let modelMs = 0;
    for (let p = 0; p < passes; p++) {
      if (p > 0) box = this._landmarkBox(pts);
      const t = performance.now();
      pts = await this._predictBox(inputElement, box);
      modelMs += performance.now() - t;
    }

    if (isVideo) {
      // The next crop follows the RAW prediction; only the displayed points are filtered,
      // so smoothing never feeds back into what the model sees (same as src/live.py).
      const tracking = state === "tracking";
      this.track = { landmarks: pts, frames: tracking ? this.track.frames + 1 : 1,
                     misses: tracking ? this.track.misses : 0 };
      if (!tracking) this.filter.reset();
      pts = this.filter.apply(pts, performance.now() / 1000, box.side);
    }

    const landmarks = pts.map((p, i) => ({
      id: i,
      x: +p.x.toFixed(2),
      y: +p.y.toFixed(2),
      normalized_x: +(p.x / w).toFixed(4),
      normalized_y: +(p.y / h).toFixed(4),
      region: APACDetector.regionName(i)
    }));
    return {
      ...base, success: true, faceDetected: true, landmarks, pointCount: landmarks.length,
      executionTimeMs: Math.round(performance.now() - startTime), modelMs: Math.round(modelMs),
      stability: isVideo ? this._calculateStability(landmarks) : "Single frame", trackState: state,
      statusNotice: state === "fallback"
        ? "No face location from MediaPipe: APAC-Face ran on a centred crop of the image"
        : "Trained APAC-Face model active (ONNX Runtime Web)"
    };
  }

  _noFace(base, startTime) {
    return { ...base, success: true, faceDetected: false, landmarks: [], pointCount: 0,
             executionTimeMs: Math.round(performance.now() - startTime), stability: "No Face", trackState: "searching" };
  }

  /** Frame-to-frame displacement on stable points (eye corners, nose tip, mouth corners). Not accuracy. */
  _calculateStability(current) {
    if (!this.lastFrameLandmarks) {
      this.lastFrameLandmarks = current;
      return "Acquiring Face";
    }
    const idx = [16, 54, 60, 72, 76, 82, 96, 97];
    let total = 0;
    for (const i of idx) {
      total += Math.hypot(current[i].normalized_x - this.lastFrameLandmarks[i].normalized_x,
                          current[i].normalized_y - this.lastFrameLandmarks[i].normalized_y);
    }
    this.lastFrameLandmarks = current;
    const avg = total / idx.length;
    if (avg < 0.003) return "Optimal (Stable)";
    if (avg < 0.008) return "Good";
    if (avg < 0.02) return "Tracking Motion";
    return "High Motion";
  }

  static regionName(i) {
    for (const [name, a, b] of WFLW_REGIONS) if (i >= a && i < b) return name;
    return "Face";
  }

  getConnections() {
    return this.connections;
  }

  getModelInfo() {
    return {
      name: this.name,
      isReady: this.isReady,
      isConnected: this.isConnected,
      isSimulation: false,
      pointCount: 98,
      statusMessage: this.isReady ? "Operational (trained APAC-Face model, ONNX Runtime Web)" : (this.loadError || "Loading"),
      architecture: "MobileNetV2 + regression head, fine-tuned on WFLW (augmentation + balanced sampling)"
    };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { APACDetector };
} else {
  window.APACDetector = APACDetector;
}
